"""Validate an Azure ML pipeline data output with the repository's contract tests."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"pipeline data file does not exist: {args.data}")

    os.environ["ITCS355_RAW_DATA_PATH"] = str(args.data.resolve())
    code = pytest.main(["-q", str(ROOT / "tests" / "test_data.py")])
    if code != pytest.ExitCode.OK:
        return int(code)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text("contracts-passed\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
