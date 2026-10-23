"""Register a pipeline candidate only when the gate receipt exists and says pass."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.gate.read_text(encoding="utf-8").strip() != "pass":
        raise SystemExit("evaluation gate receipt is missing or not pass; refusing registration")

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    required = ("run_id", "seed", "val_roc_auc", "test_roc_auc")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise SystemExit("training metrics missing registration fields: " + ", ".join(missing))
    command = [
        sys.executable, str(ROOT / "scripts" / "register_model.py"),
        "--run-id", str(metrics["run_id"]),
        "--training-job-id", str(metrics.get("training_job_id", "azureml-pipeline")),
        "--image-digest", str(metrics.get("image_digest", "unknown")),
        "--seed", str(metrics["seed"]),
        "--metric-val", str(metrics["val_roc_auc"]),
        "--metric-test", str(metrics["test_roc_auc"]),
        "--out", str(args.out),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"registered candidate metadata at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
