"""Explicit Azure Container Apps environment, identity, and teardown operations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.azure import AzureAdapter
from src import config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("create-environment", "create-identity", "teardown"))
    args = parser.parse_args()
    cfg = config.load(strict=True)
    if cfg.provider != "azure":
        raise SystemExit("Set CLOUD_PROVIDER=azure in the private cloud.env before Azure operations.")
    adapter = AzureAdapter(cfg)
    if args.operation == "create-environment":
        print(adapter.create_container_apps_environment())
    elif args.operation == "create-identity":
        print(adapter.create_containerapp_identity())
    else:
        deleted = adapter.teardown(cfg.tags(3))
        for item in deleted:
            print(item)
        if not deleted:
            print("No tagged Lab 3 resources remained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
