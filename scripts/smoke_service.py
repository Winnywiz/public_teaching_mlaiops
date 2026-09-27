"""Smoke-test three predictions through the selected adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config

SAMPLES = [
    {"temp_c": 62.0, "vibration_mm_s": 1.8, "pressure_kpa": 334.0,
     "hours_since_service": 900.0, "load_pct": 54.0, "ambient_humidity": 48.0},
    {"temp_c": 79.0, "vibration_mm_s": 3.2, "pressure_kpa": 315.0,
     "hours_since_service": 4200.0, "load_pct": 68.0, "ambient_humidity": 55.0},
    {"temp_c": 102.0, "vibration_mm_s": 6.4, "pressure_kpa": 291.0,
     "hours_since_service": 8100.0, "load_pct": 91.0, "ambient_humidity": 72.0},
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="Container App name or base URL")
    args = parser.parse_args()
    cfg = config.load(strict=cfg_provider_requires_auth(args.endpoint))
    adapter = get_adapter(cfg)
    versions: set[str] = set()
    for index, payload in enumerate(SAMPLES, start=1):
        result = adapter.invoke(args.endpoint, payload)
        probability = float(result["probability"])
        if not 0 <= probability <= 1:
            raise RuntimeError(f"sample {index} returned probability outside [0, 1]")
        if not result.get("model_version") or not result.get("request_id"):
            raise RuntimeError(f"sample {index} did not return model_version and request_id")
        versions.add(str(result["model_version"]))
        print(json.dumps({
            "sample": index,
            "probability": round(probability, 6),
            "model_version": result["model_version"],
            "request_id": result["request_id"],
        }))
    if len(versions) != 1:
        raise RuntimeError(f"smoke requests reached multiple model versions: {sorted(versions)}")
    print(f"PASS: 3 predictions served by model version {next(iter(versions))}")
    return 0


def cfg_provider_requires_auth(endpoint: str) -> bool:
    """A URL can be smoke-tested locally without cloud.env; names need provider config."""
    return not endpoint.startswith(("http://", "https://"))


if __name__ == "__main__":
    raise SystemExit(main())
