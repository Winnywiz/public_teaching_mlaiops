"""Register the selected MLflow run and attach the complete Lab 2 lineage.

The MLflow registration is used by ``reload_check.py``. When ``--provider-model-uri``
is supplied, the same lineage is also written to the selected cloud provider registry
through ``CloudAdapter.register_model`` and promoted through its staging mechanism.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow import MlflowClient

from cloudlayer.factory import get_adapter
from src import config
from src.train import dvc_data_version, git_commit


LINEAGE_FIELDS = (
    "git_commit", "data_version", "mlflow_run_id", "training_job_id",
    "image_digest", "seed", "metric_val", "metric_test",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register and promote a Lab 2 model")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--training-job-id", default="local-selected-run")
    parser.add_argument("--image-digest", default="local-development")
    parser.add_argument("--image-uri", default=None)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--metric-val", type=float, default=None)
    parser.add_argument("--metric-test", type=float, default=None)
    parser.add_argument("--provider-model-uri", default=None,
                        help="s3://... model.tar.gz for cloud registry registration")
    parser.add_argument("--stage", default="Staging")
    parser.add_argument("--out", type=Path, default=Path("reports/registered-model.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = config.load(strict=False)
    model_name = args.name or cfg.model_registry_name
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient()

    run = client.get_run(args.run_id)
    metric_val = args.metric_val
    metric_test = args.metric_test
    if metric_val is None:
        metric_val = float(run.data.metrics["val_roc_auc"])
    if metric_test is None:
        metric_test = float(run.data.metrics["test_roc_auc"])

    lineage = {
        "git_commit": run.data.tags.get("git_commit", git_commit()),
        "data_version": run.data.tags.get("data_version", run.data.params.get("data_version", dvc_data_version())),
        "mlflow_run_id": args.run_id,
        "training_job_id": args.training_job_id,
        "image_digest": args.image_digest,
        "seed": str(args.seed),
        "metric_val": f"{metric_val:.8f}",
        "metric_test": f"{metric_test:.8f}",
    }

    if cfg.provider == "azure":
        # Azure ML's MLflow registry does not expose MLflow 3's logged-models
        # search endpoint. Its legacy model-version endpoint accepts the Azure
        # artifact URI for the run's uploaded model directory.
        existing = [
            version for version in client.search_model_versions(f"name='{model_name}'")
            if version.run_id == args.run_id
        ]
        model_version = existing[0] if existing else client.create_model_version(
            name=model_name,
            source=f"azureml://artifacts/ExperimentRun/dcid.{args.run_id}/model",
            run_id=args.run_id,
            description="ITCS355 Lab 2 selected model with complete code/data/job/image lineage.",
        )
    else:
        model_version = mlflow.register_model(f"runs:/{args.run_id}/model", model_name)
    version = str(model_version.version)
    for key in LINEAGE_FIELDS:
        client.set_model_version_tag(model_name, version, key, lineage[key])
    client.update_model_version(
        name=model_name,
        version=version,
        description="ITCS355 Lab 2 selected model with complete code/data/job/image lineage.",
    )
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=args.stage,
            archive_existing_versions=True,
        )
    except Exception as exc:
        # MLflow 3 prefers aliases; keep the explicit stage attempt for older servers.
        client.set_registered_model_alias(model_name, args.stage.lower(), version)
        print(f"stage transition unavailable ({type(exc).__name__}); set alias {args.stage.lower()}")
    else:
        try:
            client.set_registered_model_alias(model_name, args.stage.lower(), version)
        except Exception:
            pass

    result = {
        "mlflow_model_name": model_name,
        "mlflow_version": version,
        "mlflow_uri": f"models:/{model_name}/{version}",
        "stage": args.stage,
        "lineage": lineage,
    }

    if args.provider_model_uri:
        adapter = get_adapter(cfg)
        if not hasattr(adapter, "set_registry_metadata"):
            raise RuntimeError("selected adapter does not expose Lab 2 registry metadata")
        provider_metadata = {**lineage, "image_uri": args.image_uri or ""}
        adapter.set_registry_metadata(provider_metadata)
        provider_version = adapter.register_model(args.provider_model_uri, model_name)
        provider_ref = f"{model_name}:{provider_version}"
        promoted = adapter.promote_model(provider_ref, args.stage)
        result["provider_model_version"] = provider_version
        result["provider_promotion"] = promoted

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
