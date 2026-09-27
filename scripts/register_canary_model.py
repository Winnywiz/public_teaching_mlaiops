"""Train and register a small, deliberately weaker sklearn model for the canary lab."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src import config, data, mlflow_compat, seeds


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-version", required=True)
    parser.add_argument("--min-drop", type=float, default=0.005)
    parser.add_argument("--max-drop", type=float, default=0.04)
    parser.add_argument("--target-drop", type=float, default=0.015)
    parser.add_argument("--out", type=Path, default=Path("reports/lab3-canary-model.json"))
    args = parser.parse_args()
    if args.min_drop <= 0 or args.max_drop < args.min_drop:
        parser.error("require 0 < --min-drop <= --max-drop")

    cfg = config.load(strict=True)
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient()
    stable_ref = f"models:/{cfg.model_registry_name}/{args.stable_version}"
    stable = mlflow.sklearn.load_model(stable_ref)
    frame = data.load_raw(cfg.raw_path)
    train_df, val_df, _ = data.split(frame, seed=seeds.DEFAULT_SEED)
    features = val_df[data.FEATURES]
    labels = val_df[data.TARGET].to_numpy()
    baseline_auc = float(roc_auc_score(labels, stable.predict_proba(features)[:, 1]))

    candidates: list[tuple[float, dict[str, int], RandomForestClassifier]] = []
    seed = seeds.DEFAULT_SEED + 17
    for trees, depth, leaf in (
        (trees, depth, leaf)
        for trees in (5, 10, 20, 40)
        for depth in (1, 2, 3, 4, 5)
        for leaf in (1, 5)
    ):
        params = {"n_estimators": trees, "max_depth": depth, "min_samples_leaf": leaf}
        model = RandomForestClassifier(**params, random_state=seed, n_jobs=-1)
        model.fit(train_df[data.FEATURES], train_df[data.TARGET])
        auc = float(roc_auc_score(labels, model.predict_proba(features)[:, 1]))
        drop = baseline_auc - auc
        if drop >= args.min_drop:
            candidates.append((drop, params, model))

    if not candidates:
        raise SystemExit(
            f"No candidate was at least {args.min_drop:.4f} ROC-AUC below the stable model; "
            "widen --max-drop only if a small-margin candidate remains acceptable."
        )
    in_range = [item for item in candidates if item[0] <= args.max_drop]
    pool = in_range or candidates
    drop, params, candidate = min(pool, key=lambda item: abs(item[0] - args.target_drop))
    if drop > args.max_drop:
        print(f"Warning: smallest available degradation was {drop:.4f}, above max {args.max_drop:.4f}.")

    with mlflow.start_run(run_name="lab3-deliberately-weaker-canary") as run:
        mlflow.log_params({**params, "stable_version": args.stable_version, "candidate_seed": seed})
        mlflow.log_metrics({"stable_roc_auc": baseline_auc, "candidate_roc_auc": baseline_auc - drop,
                            "roc_auc_drop": drop})
        mlflow.set_tag("lab", "3")
        mlflow.set_tag("purpose", "intentional-small-canary-degradation")
        mlflow_compat.log_sklearn_model(candidate, artifact_path="model")
        source = (
            f"azureml://artifacts/ExperimentRun/dcid.{run.info.run_id}/model"
            if cfg.provider == "azure"
            else f"runs:/{run.info.run_id}/model"
        )
        registered = client.create_model_version(
            name=cfg.model_registry_name,
            source=source,
            run_id=run.info.run_id,
            description=(
                f"Lab 3 canary candidate; validation AUC {baseline_auc - drop:.5f} versus "
                f"stable version {args.stable_version} at {baseline_auc:.5f}. Not promoted."
            ),
        )

    version = str(registered.version)
    client.set_model_version_tag(cfg.model_registry_name, version, "lab", "3")
    client.set_model_version_tag(cfg.model_registry_name, version, "purpose", "canary-only-not-promoted")
    result = {
        "model_name": cfg.model_registry_name,
        "stable_version": args.stable_version,
        "candidate_version": version,
        "stable_roc_auc": baseline_auc,
        "candidate_roc_auc": baseline_auc - drop,
        "roc_auc_drop": drop,
        "candidate_params": params,
        "run_id": run.info.run_id,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Registered candidate version {version}; it was not promoted or deployed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
