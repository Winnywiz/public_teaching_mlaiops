"""Delete exact-tagged Lab resources and verify managed Lab 5 resources are gone.

    python scripts/teardown_verify.py --lab 5

Deletion is asynchronous. Lab 5 resources are selected only by the exact course/student/lab
tags; this script never deletes the shared workspace, registry, storage account, or ACR.
Run it again after 24 hours and verify the Azure console and cost export by hand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", type=int, required=True)
    args = ap.parse_args()

    cfg = config.load()
    if args.lab == 5 and cfg.provider != "azure":
        raise SystemExit("Lab 5's tagged teardown verifier is implemented for Azure; use CLOUD_PROVIDER=azure")
    tags = cfg.tags(args.lab)
    adapter = get_adapter(cfg)

    print(f"cleaning resources tagged exactly {tags}")
    changed = adapter.teardown(tags)
    for resource in changed:
        print(f"  requested: {resource}")

    remaining = []
    if args.lab == 5 and hasattr(adapter, "verify_lab5_teardown"):
        remaining = adapter.verify_lab5_teardown(tags)
    if remaining:
        print("\nTagged resources are still present or deleting:")
        for resource in remaining:
            print(f"  {resource}")
        print("\nRe-run after 24 hours. Confirm in Azure Portal and capture the empty resource list.")
        return 1

    if args.lab == 5:
        print("\nPASS  no active Lab 5 schedule, app, environment, compute, or running job remains under those tags")
    else:
        print("\nPASS  the provider teardown completed its verification")
    print("Still check Azure Portal and Cost Management: untagged resources and shared storage "
          "are outside this verifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
