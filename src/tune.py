"""Lab 2 — resumable, budgeted hyperparameter study.

The default mode runs the same tracked study locally for development. ``--remote``
submits each candidate through the cloud adapter, so the exact same search space runs
on discounted managed compute without making the training code provider-specific.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path
from typing import Any

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from cloudlayer.factory import get_adapter
from src import config, costs, data, mlflow_compat, seeds
from src.train import dvc_data_version, git_commit


SEARCH_SPACE: dict[str, list] = {
    "n_estimators": [100, 300],
    "max_depth": [4, 8, 12],
    "min_samples_leaf": [1, 5],
}


def grid(space: dict[str, list]) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ITCS355 Lab 2 — budgeted study")
    p.add_argument("--trials", type=int, default=12, help="minimum 12 for the lab")
    p.add_argument("--budget-thb", type=float, default=150.0)
    p.add_argument("--instance", default="local", help="key into src/costs.py PRICE_TABLE")
    p.add_argument("--spot", action="store_true", help="use discounted managed compute")
    p.add_argument("--remote", action="store_true", help="submit trials through the cloud adapter")
    p.add_argument("--image-uri", default=None, help="digest-pinned image for --remote")
    p.add_argument("--data-uri", default=None, help="S3/GCS/Blob prefix already holding sensors.csv")
    p.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    p.add_argument("--experiment", default="itcs355-lab2")
    p.add_argument("--checkpoint", type=Path, default=Path("reports/tune_checkpoint.json"),
                   help="Resume file. An interruption should cost minutes, not the run.")
    p.add_argument("--poll-seconds", type=float, default=30.0)
    return p.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        state.setdefault("completed", [])
        state.setdefault("results", [])
        state.setdefault("spent_thb", 0.0)
        state.setdefault("in_flight", None)
        return state
    return {"completed": [], "results": [], "spent_thb": 0.0, "in_flight": None}


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]", "-", value).strip("-").lower()[:55]


def _local_trial(
    cfg,
    params: dict[str, Any],
    index: int,
    seed: int,
    experiment: str,
    train_df,
    val_df,
    test_df,
    fingerprint: str,
    data_version: str,
    instance: str,
    spot: bool,
    rate: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    with mlflow.start_run(run_name=f"trial-{index:02d}") as run:
        model = RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
        model.fit(train_df[data.FEATURES], train_df[data.TARGET])

        metrics: dict[str, float] = {}
        for name, part in (("val", val_df), ("test", test_df)):
            proba = model.predict_proba(part[data.FEATURES])[:, 1]
            metrics[f"{name}_roc_auc"] = float(roc_auc_score(part[data.TARGET], proba))
            metrics[f"{name}_pr_auc"] = float(average_precision_score(part[data.TARGET], proba))

        duration_s = time.perf_counter() - started
        trial_cost = duration_s / 3600.0 * rate
        mlflow.log_params({
            **params,
            "seed": seed,
            "instance_type": instance,
            "spot": spot,
            "data_version": data_version,
            "training_job_id": f"local-trial-{index:02d}",
        })
        mlflow.log_metrics({**metrics, "duration_s": duration_s, "cost_thb": trial_cost})
        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_fingerprint": fingerprint,
            "data_version": data_version,
            "training_job_id": f"local-trial-{index:02d}",
            "image_digest": "local-development",
            "split_strategy": "group_by_machine_id",
        })
        mlflow_compat.log_sklearn_model(model)
        return {
            "run_id": run.info.run_id,
            "training_job_id": f"local-trial-{index:02d}",
            "val_roc_auc": metrics["val_roc_auc"],
            "test_roc_auc": metrics["test_roc_auc"],
            "cost_thb": trial_cost,
            "duration_s": duration_s,
            "instance_type": instance,
            "spot": spot,
            **params,
        }


def _remote_trial(
    cfg,
    adapter,
    params: dict[str, Any],
    index: int,
    seed: int,
    experiment: str,
    image_uri: str,
    data_uri: str,
    data_version: str,
    instance: str,
    spot: bool,
    rate: float,
    poll_seconds: float,
    existing_job_id: str | None = None,
    on_submitted=None,
) -> dict[str, Any]:
    job_name = _safe_name(f"{cfg.project_id}-lab2-trial-{index:02d}-{int(time.time())}")
    image_digest = image_uri.split("@", 1)[1] if "@" in image_uri else "unknown"
    container_args = [
        "--n-estimators", str(params["n_estimators"]),
        "--max-depth", str(params["max_depth"]),
        "--min-samples-leaf", str(params["min_samples_leaf"]),
        "--seed", str(seed),
        "--experiment", experiment,
        "--run-name", f"trial-{index:02d}",
        "--data-path", "/opt/ml/input/data/training/sensors.csv",
        "--model-out", "/opt/ml/model/model.joblib",
        "--checkpoint-dir", "/opt/ml/checkpoints",
        "--training-job-id", job_name,
        "--image-digest", image_digest,
        "--git-commit", git_commit(),
        "--data-version", data_version,
        "--instance-type", instance,
    ]
    if spot:
        container_args.append("--spot")

    request = {
        "job_name": job_name,
        "instance_type": instance,
        "spot": spot,
        "poll_seconds": poll_seconds,
        "hourly_rate_thb": rate,
        "image_digest": image_digest,
        "input_s3_uri": data_uri,
        "container_args": container_args,
        "hyperparameters": params,
        "mlflow_tracking_uri": cfg.mlflow_tracking_uri,
        "tags": {"trial": str(index), "seed": str(seed)},
    }
    job_id = existing_job_id or adapter.submit_training(image_uri, request)
    if on_submitted and not existing_job_id:
        on_submitted(job_id)
    result = adapter.wait_training(job_id)
    if result.get("cost_thb") is None and result.get("billable_seconds") is not None:
        result["cost_thb"] = float(result["billable_seconds"]) / 3600.0 * rate

    # The training container records the MLflow run ID. Recover it from the tracking server
    # after the managed job completes so the registry can point back to the exact run.
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    experiment_obj = mlflow.get_experiment_by_name(experiment)
    if experiment_obj:
        runs = mlflow.search_runs(
            experiment_ids=[experiment_obj.experiment_id],
            filter_string=f"tags.training_job_id = '{job_id}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs.empty:
            row = runs.iloc[0]
            result["run_id"] = row["run_id"]
            result["val_roc_auc"] = float(row.get("metrics.val_roc_auc", result.get("val_roc_auc", 0.0)))
            result["test_roc_auc"] = float(row.get("metrics.test_roc_auc", result.get("test_roc_auc", 0.0)))
    result.update({
        **params,
        "training_job_id": job_id,
        "image_uri": image_uri,
        "image_digest": image_digest,
        "instance_type": instance,
        "spot": spot,
    })
    if result.get("run_id") and result.get("cost_thb") is not None:
        tracking_client = mlflow.MlflowClient()
        tracking_client.log_metric(
            str(result["run_id"]), "cost_thb", float(result["cost_thb"])
        )
        if result.get("billable_seconds") is not None:
            tracking_client.log_metric(
                str(result["run_id"]),
                "billable_seconds",
                float(result["billable_seconds"]),
            )
    return result


def _data_prefix(uri: str) -> str:
    return uri.rsplit("/", 1)[0] if "/" in uri.rstrip("/") else uri


def main() -> None:
    args = parse_args()
    if args.trials < 12:
        raise SystemExit("Lab 2 requires at least 12 trials")
    candidates = grid(SEARCH_SPACE)
    if args.trials > len(candidates):
        raise SystemExit(f"search space contains only {len(candidates)} distinct candidates")

    cfg = config.load(strict=False)
    args.instance = costs.resolve_instance(cfg.provider, args.instance)
    seed = seeds.set_all(args.seed)
    df = data.load_raw(cfg.raw_path)
    fingerprint = data.data_fingerprint(cfg.raw_path)
    data_version = dvc_data_version()
    train_df, val_df, test_df = data.split(df, seed=seed)
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)

    try:
        rate = costs.hourly_rate(cfg.provider, args.instance, spot=args.spot)
    except KeyError:
        if args.remote:
            raise
        rate = 0.0

    adapter = None
    data_uri = args.data_uri
    if args.remote:
        if not args.image_uri or "@sha256:" not in args.image_uri:
            raise SystemExit("--remote requires a digest-pinned --image-uri")
        adapter = get_adapter(cfg)
        if not data_uri:
            uploaded = adapter.upload(str(cfg.raw_path), "lab2/data/sensors.csv")
            data_uri = _data_prefix(uploaded)
        if not data_uri:
            raise SystemExit("--remote requires a data URI or a working object-storage adapter")

    state = load_checkpoint(args.checkpoint)
    skipped: list[dict[str, Any]] = []
    for index, params in enumerate(candidates[:args.trials]):
        key = json.dumps(params, sort_keys=True)
        if key in state["completed"]:
            print(f"trial {index}: already done, skipping (resumed from checkpoint)")
            continue
        if state["spent_thb"] >= args.budget_thb:
            skipped.append(params)
            continue

        print(f"trial {index}: starting {params}")
        if args.remote:
            in_flight = state.get("in_flight") or {}
            resume_job_id = (
                in_flight.get("job_id")
                if in_flight.get("key") == key
                else None
            )
            if not resume_job_id:
                state["in_flight"] = {
                    "key": key,
                    "trial": index,
                    "params": params,
                    "submitted_at": time.time(),
                }
                save_checkpoint(args.checkpoint, state)

            def record_submission(job_id: str) -> None:
                state["in_flight"]["job_id"] = job_id
                save_checkpoint(args.checkpoint, state)

            result = _remote_trial(
                cfg, adapter, params, index, seed, args.experiment, args.image_uri, data_uri,
                data_version, args.instance, args.spot, rate, args.poll_seconds,
                existing_job_id=resume_job_id,
                on_submitted=record_submission,
            )
        else:
            result = _local_trial(
                cfg, params, index, seed, args.experiment, train_df, val_df, test_df,
                fingerprint, data_version, args.instance, args.spot, rate,
            )

        result["data_fingerprint"] = fingerprint
        result["data_version"] = data_version
        state["results"].append(result)
        state["completed"].append(key)
        state["spent_thb"] += float(result.get("cost_thb", 0.0))
        state["in_flight"] = None
        save_checkpoint(args.checkpoint, state)
        print(
            f"trial {index}: val_roc_auc={float(result.get('val_roc_auc', 0.0)):.4f} "
            f"cost={float(result.get('cost_thb', 0.0)):.4f} THB "
            f"cumulative={state['spent_thb']:.4f}"
        )

    print(f"\nspent {state['spent_thb']:.4f} of {args.budget_thb} THB")
    if skipped:
        print(f"BUDGET EXHAUSTED — {len(skipped)} configurations not run:")
        for item in skipped:
            print(f"  {item}")
    if args.remote and data_uri:
        print(f"remote input prefix: {data_uri}")


if __name__ == "__main__":
    main()
