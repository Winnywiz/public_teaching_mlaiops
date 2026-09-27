"""Turn Locust's aggregated CSVs into committed percentile evidence."""
from __future__ import annotations

import argparse
import csv
import platform
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summary(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Locust output {path}; run `make loadtest` first")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    aggregate = next(
        (row for row in rows if row.get("Name", "").strip().lower() == "aggregated"),
        None,
    )
    if aggregate is None:
        raise ValueError(f"{path} has no Aggregated row; keep Locust's --csv output format")
    values = {key.strip().lower(): value for key, value in aggregate.items()}

    def number(*keys: str) -> float:
        for key in keys:
            value = values.get(key.lower())
            if value not in (None, ""):
                return float(value)
        raise ValueError(f"{path} is missing one of these columns: {keys}")

    count = number("Request Count")
    failures = number("Failure Count")
    return {
        "requests": count,
        "failures": failures,
        "error_pct": failures / count * 100 if count else 0.0,
        "rps": number("Requests/s", "Requests/s"),
        "p50": number("50%", "Median Response Time"),
        "p95": number("95%"),
        "p99": number("99%"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--p95-target-ms", type=float, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--duration", required=True)
    parser.add_argument("--payload-bytes", type=int, default=0)
    parser.add_argument("--output-prefix", default="outputs/lab3")
    parser.add_argument("--out", type=Path, default=Path("reports/lab3-load.md"))
    args = parser.parse_args()

    results = [
        (vus, summary(ROOT / f"{args.output_prefix}-vus{vus}_stats.csv"))
        for vus in (1, 10, 50)
    ]
    breaking = next(
        (vus for vus, result in results
         if result["p95"] > args.p95_target_ms or result["error_pct"] >= 1.0),
        None,
    )
    meeting = next(
        (vus for vus, result in results
         if result["p95"] <= args.p95_target_ms and result["error_pct"] < 1.0),
        None,
    )
    host = (urlparse(args.target).hostname or "").lower()
    is_local = host in {"localhost", "127.0.0.1", "::1"}
    kind = "local process" if is_local else "managed endpoint"
    caveat = (
        "> These are local engineering measurements, not Azure endpoint evidence. Cloud load and\n"
        "> selected replica sizing remain pending until the managed service is deployed and tested.\n"
        if is_local else
        "> These are managed endpoint measurements. Cold-start is recorded separately; actual tagged\n"
        "> Azure billing must still be reconciled in the account.\n"
    )
    rows = "\n".join(
        f"| {vus} | {int(r['requests'])} | {r['rps']:.3f} | {r['p50']:.1f} | "
        f"{r['p95']:.1f} | {r['p99']:.1f} | {r['error_pct']:.3f}% |"
        for vus, r in results
    )
    if breaking is None:
        breaking_text = "Not reached at 50 concurrent users."
    else:
        breaking_text = f"First exceeded at {breaking} concurrent users."
    meeting_text = (
        f"The smallest tested configuration meeting the target is {meeting} concurrent "
        f"{'user' if meeting == 1 else 'users'}."
        if meeting is not None else "No tested concurrency configuration met the target."
    )
    cold_report = ROOT / "reports/lab3-cold-start.md"
    canary_reports = sorted((ROOT / "reports").glob("lab3-rollback-*.md"))
    instance_report = ROOT / "reports/lab3-instance-size.md"
    cold_status = f"`{cold_report.relative_to(ROOT).as_posix()}`" if cold_report.exists() else "Pending"
    canary_status = (
        f"`{canary_reports[-1].relative_to(ROOT).as_posix()}`" if canary_reports else "Pending"
    )
    instance_status = (
        f"`{instance_report.relative_to(ROOT).as_posix()}`" if instance_report.exists() else "Pending"
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""# Lab 3 — load-test results

{caveat}

## Measurement setup

- Target: `{args.target}`
- Load generator: Locust, headless, 1 / 10 / 50 users, `{args.duration}` per run
- Stated p95 target, set before measurement: **{args.p95_target_ms:.0f} ms**
- Service size: `{args.instance}` ({kind})
- Approximate request payload: {args.payload_bytes} bytes
- Host: {platform.platform()} · Python {platform.python_version()}

| Concurrent users | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |
|---:|---:|---:|---:|---:|---:|---:|
{rows}

Breaking point: {breaking_text} A failed request or p95 above the stated target counts as breaking.
{meeting_text}
Locust mixes single and 50-row batch calls (9:1 task weight); its request throughput counts HTTP
requests, not individual predictions. Use `make batch-test` for prediction-throughput comparison.

## Related evidence

| Variable | Current result | Remaining cloud measurement |
|---|---|---|
| Cold start | {cold_status} | {('Measure first request after scale-to-zero' if not cold_report.exists() else 'Captured separately from steady-state percentiles') if not is_local else 'Repeat against the managed endpoint'} |
| Batch size 100 vs. 100 singles | `reports/lab3-batch.md` | {('Repeat against managed endpoint' if is_local else 'Managed comparison recorded')}
| Payload size | `reports/lab3-payload.md` | {('Repeat against managed endpoint' if is_local else 'Managed comparison recorded')}
| Instance-size latency and cost | {instance_status} | {('Measure one size up and calculate its cost delta' if not instance_report.exists() else 'Comparison recorded')}
| Canary detection and rollback | {canary_status} | {('Run the 90/10 metric-gated canary' if not canary_reports else 'Timestamped traffic and rollback evidence recorded')}
| Cost per 1,000 predictions | `reports/lab3-cost.md` | Reconcile retail estimate with actual Azure billing
""", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
