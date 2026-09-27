"""Compute transparent cost/1,000 and batch break-even from measured inputs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def optional_number(name: str, cli_value: str | None) -> float | None:
    value = cli_value if cli_value is not None else os.environ.get(name)
    if value is None or not value.strip():
        return None
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} cannot be negative")
    return number


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utilization", type=float, default=0.25)
    parser.add_argument("--hourly-rate-thb", default=None)
    parser.add_argument("--throughput-rps", default=None, help="measured HTTP requests/second")
    parser.add_argument("--predictions-per-request", type=float, default=1.0)
    parser.add_argument("--request-rate-thb-per-million", default=None)
    parser.add_argument("--warm-hourly-rate-thb", default=None)
    parser.add_argument("--batch-duration-ms", type=float, default=None)
    parser.add_argument("--batch-rows", type=int, default=100)
    parser.add_argument("--batch-hourly-rate-thb", default=None)
    parser.add_argument("--price-region", default="not recorded")
    parser.add_argument("--price-as-of", default="not recorded")
    parser.add_argument("--batch-cost-per-1000-thb", default=None)
    parser.add_argument("--instance", default=os.environ.get("SERVE_INSTANCE", "0.5cpu/1Gi"))
    parser.add_argument("--out", type=Path, default=Path("reports/lab3-cost.md"))
    args = parser.parse_args()
    if not 0 < args.utilization <= 1:
        parser.error("--utilization must be greater than 0 and at most 1")

    rate = optional_number("HOURLY_RATE_THB", args.hourly_rate_thb)
    throughput = optional_number("THROUGHPUT_RPS", args.throughput_rps)
    request_rate = optional_number(
        "REQUEST_RATE_THB_PER_MILLION", args.request_rate_thb_per_million
    )
    warm_rate = optional_number("WARM_HOURLY_RATE_THB", args.warm_hourly_rate_thb)
    batch_unit_cost = optional_number("BATCH_COST_PER_1000_THB", args.batch_cost_per_1000_thb)
    batch_rate = optional_number("BATCH_HOURLY_RATE_THB", args.batch_hourly_rate_thb)
    cost_1k = None
    prediction_throughput = None
    if rate is not None and throughput is not None:
        if throughput <= 0:
            parser.error("--throughput-rps must be positive")
        if args.predictions_per_request <= 0:
            parser.error("--predictions-per-request must be positive")
        prediction_throughput = throughput * args.predictions_per_request
        compute_cost_1k = rate * 1000 / (
            prediction_throughput * args.utilization * 3600
        )
        request_cost_1k = (
            request_rate * (1000 / args.predictions_per_request) / 1_000_000
            if request_rate is not None else 0.0
        )
        cost_1k = compute_cost_1k + request_cost_1k
    if batch_unit_cost is None and args.batch_duration_ms is not None:
        if args.batch_duration_ms <= 0 or args.batch_rows <= 0:
            parser.error("batch duration and rows must be positive")
        effective_batch_rate = batch_rate if batch_rate is not None else rate
        if effective_batch_rate is not None:
            calls_per_1000 = 1000 / args.batch_rows
            per_call = effective_batch_rate * (args.batch_duration_ms / 1000) / 3600
            per_call += (request_rate or 0.0) / 1_000_000
            batch_unit_cost = per_call * calls_per_1000
    break_even_active_day = None
    break_even_warm_day = None
    if rate is not None and batch_unit_cost is not None and batch_unit_cost > 0:
        break_even_active_day = rate * 24 * 1000 / batch_unit_cost
    if warm_rate is not None and batch_unit_cost is not None and batch_unit_cost > 0:
        break_even_warm_day = warm_rate * 24 * 1000 / batch_unit_cost

    def show(value: float | None, suffix: str = "") -> str:
        return f"{value:,.6f}{suffix}" if value is not None else "PENDING — supply the measured cloud value"

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""# Lab 3 — serving cost report

Instance configuration: `{args.instance}` · utilisation assumption: **{args.utilization:.0%}**.
The request mix is represented by **{args.predictions_per_request:.3f} predictions/HTTP request**.
The hourly rate is the active-replica estimate; request charges are added when a retail rate is provided.
Price basis: public Azure retail rates for **{args.price_region}**, retrieved **{args.price_as_of}**.
This is not the account invoice. Azure Consumption has monthly per-subscription grants of 180,000
vCPU-seconds, 360,000 GiB-seconds, and 2 million HTTP requests; whether they remain available is
not visible to this worksheet.

| Input / result | Value |
|---|---:|
| Active replica hourly cost (THB/hour) | {show(rate)} |
| One-replica idle hourly cost if kept warm (THB/hour) | {show(warm_rate)} |
| Measured HTTP throughput (requests/second) | {show(throughput)} |
| Implied prediction throughput (predictions/second) | {show(prediction_throughput)} |
| HTTP request retail price (THB / million requests) | {show(request_rate)} |
| Measured batch request duration (ms) | {show(args.batch_duration_ms)} |
| Predictions per batch request | {args.batch_rows if args.batch_duration_ms is not None else 'not supplied'} |
| Active hourly rate used for batch estimate (THB/hour) | {show(batch_rate if batch_rate is not None else rate if args.batch_duration_ms is not None else None)} |
| Utilisation assumption | {args.utilization:.0%} |
| Cost per 1,000 predictions (THB) | {show(cost_1k)} |
| Batch inference cost per 1,000 predictions (THB) | {show(batch_unit_cost)} |

Method: `hourly_rate × 1000 ÷ (HTTP requests/s × predictions/request × 3600 × utilisation)`,
plus the request-meter cost for `1000 ÷ predictions/request` HTTP calls. This steady-active estimate
excludes scale-to-zero, startup overhead, storage, and egress; compare it with actual tagged Azure billing.
When derived from batch timing, the batch estimate is `calls_per_1000 × (active_hourly_rate ×
median_call_seconds ÷ 3600 + HTTP_request_price_per_call)`.
Sources: [Azure Container Apps pricing](https://azure.microsoft.com/en-us/pricing/details/container-apps/),
[Container Apps billing](https://learn.microsoft.com/en-us/azure/container-apps/billing), and the
[Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices).

""" + (
        f"Batch vs. one continuously idle (warm) replica: approximately **{break_even_warm_day:,.0f} predictions/day**.\n"
        if break_even_warm_day is not None
        else "Batch vs. one continuously idle (warm) replica: pending; supply its idle hourly rate.\n"
    ) + (
        f"Batch vs. one continuously active replica: approximately **{break_even_active_day:,.0f} predictions/day**.\n"
        if break_even_active_day is not None
        else "Batch vs. one continuously active replica: pending; supply its active hourly rate.\n"
    ) + "\n" + (
        "These break-even values compare the measured batch unit cost with one continuously allocated replica; "
        "scale-to-zero can be cheaper when idle, while startup latency and service-level requirements still matter.\n"
        if break_even_active_day is not None or break_even_warm_day is not None
        else "Record the batch duration/rate and idle/active replica cost before selecting a serving pattern.\n"
    ), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
