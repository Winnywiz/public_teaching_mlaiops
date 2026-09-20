"""Submit one Lab 2 training job through the selected cloud adapter.

Example:
    python scripts/train_remote.py --image-uri "$TRAINING_IMAGE_URI" \
        --instance ml.m5.large --spot

The local machine only submits and polls the job. The managed container receives the
data channel and writes its model artifact to the provider's output location.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config, costs, data, seeds
from src.train import dvc_data_version, git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit one managed Lab 2 training job")
    parser.add_argument("--image-uri", required=True, help="digest-pinned training image")
    parser.add_argument("--instance", default="ml.m5.large")
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    parser.add_argument("--experiment", default="itcs355-lab2")
    parser.add_argument("--run-name", default="managed-single-run")
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--data-uri", default=None, help="existing object-storage prefix")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=Path("reports/remote-training.json"))
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]", "-", value).strip("-").lower()[:55]


def data_prefix(uri: str) -> str:
    return uri.rsplit("/", 1)[0] if "/" in uri.rstrip("/") else uri


def main() -> int:
    args = parse_args()
    if "@sha256:" not in args.image_uri:
        raise SystemExit("--image-uri must be digest-pinned as repository@sha256:...")

    cfg = config.load(strict=True)
    args.instance = costs.resolve_instance(cfg.provider, args.instance)
    adapter = get_adapter(cfg)
    seed = seeds.set_all(args.seed)
    raw_path = cfg.raw_path
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} is missing; run `make data` or `dvc pull` first")
    fingerprint = data.data_fingerprint(raw_path)
    data_version = dvc_data_version()
    image_digest = args.image_uri.split("@", 1)[1]
    job_name = safe_name(args.job_name or f"{cfg.project_id}-lab2-{int(time.time())}")
    data_uri = args.data_uri
    if not data_uri:
        uploaded = adapter.upload(str(raw_path), "lab2/data/sensors.csv")
        data_uri = data_prefix(uploaded)

    rate = costs.hourly_rate(cfg.provider, args.instance, spot=args.spot)
    container_args = [
        "--n-estimators", str(args.n_estimators),
        "--max-depth", str(args.max_depth),
        "--min-samples-leaf", str(args.min_samples_leaf),
        "--seed", str(seed),
        "--experiment", args.experiment,
        "--run-name", args.run_name,
        "--data-path", "/opt/ml/input/data/training/sensors.csv",
        "--model-out", "/opt/ml/model/model.joblib",
        "--checkpoint-dir", "/opt/ml/checkpoints",
        "--training-job-id", job_name,
        "--image-digest", image_digest,
        "--git-commit", git_commit(),
        "--data-version", data_version,
        "--instance-type", args.instance,
    ]
    if args.spot:
        container_args.append("--spot")

    job_id = adapter.submit_training(args.image_uri, {
        "job_name": job_name,
        "instance_type": args.instance,
        "spot": args.spot,
        "poll_seconds": args.poll_seconds,
        "hourly_rate_thb": rate,
        "image_digest": image_digest,
        "input_s3_uri": data_uri,
        "container_args": container_args,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "seed": seed,
        },
        "mlflow_tracking_uri": cfg.mlflow_tracking_uri,
        "tags": {"seed": str(seed), "data_version": data_version},
    })
    result = adapter.wait_training(job_id)
    result.update({
        "git_commit": git_commit(),
        "data_version": data_version,
        "data_fingerprint": fingerprint,
        "seed": seed,
        "image_digest": image_digest,
        "instance_type": args.instance,
        "spot": args.spot,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
