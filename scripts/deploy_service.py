"""Deploy an exact model-registry version to the configured managed endpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="exact registry version; never use an alias")
    parser.add_argument("--endpoint", default="itcs355-lab3")
    parser.add_argument("--instance", default="0.5cpu/1Gi")
    args = parser.parse_args()

    if not args.version.strip():
        parser.error("--version is required; pass the selected Lab 2 registry version")
    cfg = config.load(strict=True)
    if cfg.provider != "azure":
        raise SystemExit("Lab 3's tested managed-serving adapter is Azure Container Apps; set CLOUD_PROVIDER=azure.")
    model_ref = f"models:/{cfg.model_registry_name}/{args.version}"
    endpoint = get_adapter(cfg).deploy(model_ref, args.endpoint, args.instance)
    print(f"endpoint={endpoint}")
    print(f"model={model_ref}")
    print("Next: verify /ready and run `make smoke ENDPOINT=<endpoint-name-or-url>`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
