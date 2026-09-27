"""Summarize Locust runs at different request payload sizes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.write_load_report import summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="http://127.0.0.1:8080")
    parser.add_argument("--instance", default="local")
    parser.add_argument("--output-prefix", default="outputs/lab3")
    parser.add_argument("--out", type=Path, default=Path("reports/lab3-payload.md"))
    args = parser.parse_args()
    sizes = (0, 4096, 16384, 60000)
    results = [
        (size, summary(ROOT / f"{args.output_prefix}-payload{size}_stats.csv"))
        for size in sizes
    ]
    host = (urlparse(args.target).hostname or "").lower()
    measurement = "local" if host in {"localhost", "127.0.0.1", "::1"} else "managed endpoint"
    rows = "\n".join(
        f"| {size or 'baseline'} | {int(r['requests'])} | {r['rps']:.3f} | "
        f"{r['p50']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} | {r['error_pct']:.3f}% |"
        for size, r in results
    )
    baseline = results[0][1]
    largest = results[-1][1]
    p95_delta = largest["p95"] - baseline["p95"]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""# Lab 3 — payload-size experiment

{measurement.title()} Locust measurements on `{args.target}`, size `{args.instance}`, 10 users, 90:10
single:batch task ratio. Payload bytes are approximate;
the optional `metadata.padding` field is ignored by the model but validated and parsed by the API.
{"These are not cloud endpoint results." if measurement == "local" else ""}

| Target payload bytes | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |
|---:|---:|---:|---:|---:|---:|---:|
{rows}

{"This is a local-only experiment; repeat it against the managed endpoint before using it for sizing." if measurement == "local" else f"At 60,000 bytes, p95 was {largest['p95']:.1f} ms versus {baseline['p95']:.1f} ms at baseline (change {p95_delta:+.1f} ms). This observed change includes run-to-run and service variation as well as request parsing/serialization."}
""", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
