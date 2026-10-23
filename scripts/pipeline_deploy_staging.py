"""Deploy the registered pipeline result to the isolated, Lab 5-tagged staging app."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudlayer.factory import get_adapter  # noqa: E402
from src import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--instance", default="0.5cpu/1Gi")
    args = parser.parse_args()
    cfg = config.load(strict=True)
    if cfg.provider != "azure":
        raise SystemExit("Lab 5 staging deployment is configured for Azure Container Apps")
    metadata = json.loads(args.registration.read_text(encoding="utf-8"))
    model_ref = metadata.get("mlflow_uri")
    if not model_ref:
        raise SystemExit("registration output has no exact mlflow_uri to deploy")
    adapter = get_adapter(cfg)
    deploy = getattr(adapter, "deploy_lab5_staging", None)
    if deploy is None:
        raise SystemExit("the configured adapter does not implement Lab 5 staging deployment")
    endpoint = deploy(model_ref, args.instance)
    print(f"Lab 5 staging endpoint deployed: {endpoint}")
    print(f"model version: {model_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
