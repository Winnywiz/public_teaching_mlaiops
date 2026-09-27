"""Compare N individual HTTP predictions with one batch request."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sample_payload() -> dict[str, float]:
    return {
        "temp_c": 78.4,
        "vibration_mm_s": 3.1,
        "pressure_kpa": 315.2,
        "hours_since_service": 4200.0,
        "load_pct": 68.0,
        "ambient_humidity": 55.0,
    }


def post(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--instance", default="unspecified")
    parser.add_argument("--out", type=Path, default=Path("reports/lab3-batch.md"))
    args = parser.parse_args()
    if not 1 <= args.rows <= 100:
        parser.error("--rows must be between 1 and 100")
    base = args.endpoint.rstrip("/")
    rows = [sample_payload() for _ in range(args.rows)]
    single_times: list[float] = []
    batch_times: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(args.rows, 32)) as pool:
            results = list(pool.map(lambda row: post(f"{base}/predict", row), rows))
        single_times.append((time.perf_counter() - started) * 1000)
        if len(results) != args.rows:
            raise RuntimeError("single request count mismatch")

        started = time.perf_counter()
        batch_result = post(f"{base}/predict/batch", {"rows": rows})
        batch_times.append((time.perf_counter() - started) * 1000)
        if len(batch_result.get("probabilities", [])) != args.rows:
            raise RuntimeError("batch response count mismatch")

    single_median = statistics.median(single_times)
    batch_median = statistics.median(batch_times)
    host = (urlparse(args.endpoint).hostname or "").lower()
    measurement = "local" if host in {"localhost", "127.0.0.1", "::1"} else "managed endpoint"
    report = {
        "rows": args.rows,
        "repeats": args.repeats,
        "endpoint": args.endpoint,
        "single_calls_total_ms_median": round(single_median, 3),
        "one_batch_call_ms_median": round(batch_median, 3),
        "batch_speedup": round(single_median / batch_median, 3) if batch_median else None,
        "single_trial_ms": [round(value, 3) for value in single_times],
        "batch_trial_ms": [round(value, 3) for value in batch_times],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""# Lab 3 — batch-size experiment

{(measurement.title() + ' HTTP measurements.') if measurement == 'managed endpoint' else 'Local HTTP measurements; repeat against the managed endpoint before using for deployment decisions.'}

- Target: `{args.endpoint}`
- Replica size: `{args.instance}`
- Rows per comparison: {args.rows}
- Repetitions: {args.repeats}
- {args.rows} individual calls (concurrent fan-out) median wall time: **{single_median:.3f} ms**
- One `/predict/batch` call median wall time: **{batch_median:.3f} ms**
- Batch speedup: **{report['batch_speedup']:.3f}×**

The single-call baseline fans requests out concurrently with at most 32 client workers; it is not a
sequential-loop comparison. This distinction is intentional so it does not exaggerate batch gains.
The request metrics include network/JSON overhead and therefore differ from model-only inference time.
""", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
