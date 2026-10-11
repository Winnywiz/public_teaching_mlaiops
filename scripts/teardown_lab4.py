"""Disable and remove only this repository's tagged Lab 4 Azure ML schedule."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCHEDULE_NAME = "itcs355-lab4-weekly-drift"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from src import config

    cfg = config.load(strict=False)
    if cfg.provider.lower() != "azure":
        print("No Azure provider configured; no Lab 4 schedule exists in this checkout.")
        return 0

    workspace = os.environ.get("AZURE_ML_WORKSPACE")
    if not workspace:
        print("AZURE_ML_WORKSPACE is unset; no Lab 4 schedule lookup or deletion was attempted.")
        return 0

    az = shutil.which("az") or shutil.which("az.cmd")
    if not az:
        print("Azure CLI was not found; Lab 4 schedule teardown was not performed.", file=sys.stderr)
        return 2

    common = [
        "--name", SCHEDULE_NAME,
        "--resource-group", os.environ.get("AZURE_RESOURCE_GROUP", cfg.project_id),
        "--workspace-name", workspace,
    ]
    subscription = os.environ.get("AZURE_SUBSCRIPTION_ID")
    if subscription:
        common.extend(["--subscription", subscription])
    shown = subprocess.run(
        [az, "ml", "schedule", "show", *common, "--output", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode:
        message = (shown.stderr or shown.stdout).lower()
        if "not found" in message or "could not be found" in message or "resource not found" in message:
            print(f"Lab 4 schedule {SCHEDULE_NAME!r} is absent; nothing to delete.")
            return 0
        print("Could not verify the exact Lab 4 schedule; nothing was deleted.", file=sys.stderr)
        return 2

    try:
        schedule = json.loads(shown.stdout)
    except json.JSONDecodeError:
        print("Azure CLI returned invalid schedule metadata; nothing was deleted.", file=sys.stderr)
        return 2
    tags = schedule.get("tags", {}) or {}
    if schedule.get("name") != SCHEDULE_NAME or tags.get("course") != "itcs355" or str(tags.get("lab")) != "4":
        print("Schedule name or course/lab tags do not match; nothing was deleted.", file=sys.stderr)
        return 2

    disabled = subprocess.run(
        [az, "ml", "schedule", "disable", *common, "--output", "none"],
        capture_output=True,
        text=True,
        check=False,
    )
    if disabled.returncode:
        print("Could not disable the verified Lab 4 schedule; it was not deleted.", file=sys.stderr)
        return 2
    deleted = subprocess.run(
        [az, "ml", "schedule", "delete", *common, "--yes", "--output", "none"],
        capture_output=True,
        text=True,
        check=False,
    )
    if deleted.returncode:
        print("Schedule was disabled but deletion failed; confirm it remains disabled in Azure.", file=sys.stderr)
        return 2
    print(f"Disabled and deleted the exact Lab 4 schedule {SCHEDULE_NAME!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
