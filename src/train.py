"""Provider-neutral training entry point for local and managed jobs.

The managed adapter supplies a data path, job identity, image digest, and data version
as command-line arguments. Keeping those values at the training boundary means the same
container can run locally, in a managed job, or in the Lab 1 reproduction command.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config, data, seeds


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=config.REPO_ROOT,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def dvc_data_version() -> str:
    """Return the content hash recorded by DVC for the raw dataset."""
    descriptor = config.REPO_ROOT / "data" / "raw.dvc"
    if not descriptor.exists():
        return "unavailable"
    match = re.search(
        r"(?m)^\s*-\s*(?:md5|etag|checksum):\s*([0-9a-fA-F]+(?:\.dir)?)\s*$",
        descriptor.read_text(),
    )
    return match.group(1) if match else "unavailable"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ITCS355 reproducible training")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    p.add_argument("--experiment", default="itcs355-lab1")
    p.add_argument("--run-name", default=None)
    p.add_argument("--data-path", type=Path, default=None,
                   help="Input CSV path; defaults to the configured raw dataset.")
    p.add_argument("--model-out", type=Path, default=None,
                   help="Optional joblib output, used by managed training jobs.")
    p.add_argument("--training-job-id", default=None)
    p.add_argument("--image-digest", default=None)
    p.add_argument("--git-commit", default=None)
    p.add_argument("--data-version", default=None)
    p.add_argument("--instance-type", default="local")
    p.add_argument("--estimated-cost-thb", type=float, default=None)
    p.add_argument("--spot", action="store_true")
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="Write final metrics as JSON. Used by `make verify`.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config.load(strict=False)
    seed = seeds.set_all(args.seed)

    raw_path = args.data_path or cfg.raw_path
    df = data.load_raw(raw_path)
    fingerprint = data.data_fingerprint(raw_path)
    data_version = args.data_version or dvc_data_version()
    commit = args.git_commit or git_commit()
    train_df, val_df, test_df = data.split(df, seed=seed)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "seed": seed,
            "n_features": len(data.FEATURES),
            "data_version": data_version,
            "instance_type": args.instance_type,
            "spot": args.spot,
        }
        if args.training_job_id:
            params["training_job_id"] = args.training_job_id
        mlflow.log_params(params)

        tags = {
            "git_commit": commit,
            "data_fingerprint": fingerprint,
            "data_version": data_version,
            "dvc_data_version": data_version,
            "split_strategy": "group_by_machine_id",
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "n_test_rows": len(test_df),
        }
        if args.training_job_id:
            tags["training_job_id"] = args.training_job_id
        if args.image_digest:
            tags["image_digest"] = args.image_digest
        mlflow.set_tags(tags)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_df[data.FEATURES], train_df[data.TARGET])

        metrics: dict[str, float] = {}
        for name, part in (("val", val_df), ("test", test_df)):
            proba = model.predict_proba(part[data.FEATURES])[:, 1]
            metrics[f"{name}_roc_auc"] = float(roc_auc_score(part[data.TARGET], proba))
            metrics[f"{name}_pr_auc"] = float(average_precision_score(part[data.TARGET], proba))
        if args.estimated_cost_thb is not None:
            metrics["cost_thb"] = float(args.estimated_cost_thb)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        if args.model_out:
            args.model_out.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, args.model_out)

        result = {
            "run_id": run.info.run_id,
            "seed": seed,
            "git_commit": commit,
            "data_fingerprint": fingerprint,
            "data_version": data_version,
            "training_job_id": args.training_job_id or "local",
            "image_digest": args.image_digest or "local",
            "instance_type": args.instance_type,
            "spot": args.spot,
            **metrics,
        }
        print(json.dumps(result, indent=2))
        if args.metrics_out:
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
