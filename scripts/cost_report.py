"""Build an evidence-led Lab 5 cost report from user-supplied billing and measurements.

The script deliberately has no bundled Azure rate for a Container Apps configuration:
prices vary by region and meter. Supply the rate and cite the source you checked.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import costs

COMPONENTS = ("training", "storage", "serving", "pipeline", "monitoring")


def money(value: float | None) -> str:
    return "not supplied" if value is None else f"{value:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimate", type=float, required=True, help="pre-run estimate in THB")
    ap.add_argument("--actual", type=float, required=True, help="actual tagged billing total in THB")
    ap.add_argument("--rps", type=float, required=True, help="measured throughput in requests/second")
    ap.add_argument("--instance", required=True, help="exact serving size/configuration")
    ap.add_argument("--hourly-rate-thb", type=float, required=True,
                    help="verified regional provisioned rate in THB/hour")
    ap.add_argument("--price-source", required=True,
                    help="pricing source, region, and date checked")
    ap.add_argument("--billing-source", required=True,
                    help="tag-filtered Cost Management export and date range")
    ap.add_argument("--estimate-evidence", type=Path,
                    help="tracked pre-run estimate record; absence is reported explicitly")
    ap.add_argument("--gap-explanation", default="Not supplied from billing evidence.")
    for component in COMPONENTS:
        ap.add_argument(f"--{component}-thb", type=float,
                        help=f"tagged billing for {component} in THB")
    ap.add_argument("--expected-utilisation", type=float, choices=(0.05, 0.25, 0.80), default=0.25)
    ap.add_argument("--optimization", default=None)
    ap.add_argument("--before-config", default=None)
    ap.add_argument("--after-config", default=None)
    ap.add_argument("--before-rps", type=float)
    ap.add_argument("--after-rps", type=float)
    ap.add_argument("--before-hourly-rate-thb", type=float)
    ap.add_argument("--after-hourly-rate-thb", type=float)
    ap.add_argument("--before-p95-ms", type=float)
    ap.add_argument("--after-p95-ms", type=float)
    ap.add_argument("--out", type=Path, default=Path("reports/lab5-cost.md"))
    args = ap.parse_args()

    numeric = (args.estimate, args.actual, args.rps, args.hourly_rate_thb)
    if any(not math.isfinite(value) or value < 0 for value in numeric):
        ap.error("costs and rates must be finite, non-negative values")
    if args.estimate == 0 or args.rps == 0 or args.hourly_rate_thb == 0:
        ap.error("estimate, rps, and hourly rate must be positive")
    optional_numeric = (
        args.before_rps, args.after_rps, args.before_hourly_rate_thb,
        args.after_hourly_rate_thb, args.before_p95_ms, args.after_p95_ms,
        *(measured for measured in (getattr(args, f"{name}_thb") for name in COMPONENTS)),
    )
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in optional_numeric):
        ap.error("optional measurements and component costs must be finite and non-negative")
    if any(value == 0 for value in (args.before_rps, args.after_rps,
                                    args.before_hourly_rate_thb, args.after_hourly_rate_thb)
           if value is not None):
        ap.error("before/after throughput and hourly rates must be positive")

    gap = args.actual - args.estimate
    gap_pct = gap / args.estimate * 100 if args.estimate else float("nan")
    measured = {name: getattr(args, f"{name}_thb") for name in COMPONENTS}
    breakdown_rows = "\n".join(
        f"| {name.title()} | {money(value)} |"
        for name, value in measured.items()
    )
    supplied = [value for value in measured.values() if value is not None]
    breakdown_check = "not checked; provide all five components" if len(supplied) != len(COMPONENTS) else (
        f"sum={sum(supplied):.2f} THB; difference from total={sum(supplied) - args.actual:+.2f} THB"
    )
    cost_rows = "\n".join(
        f"| {int(utilisation * 100)}% | "
        f"{costs.cost_per_1k_predictions(args.hourly_rate_thb, args.rps, utilisation):.6f} |"
        for utilisation in costs.DEFAULT_UTILISATIONS
    )

    has_optimization = all((args.optimization, args.before_config, args.after_config)) and all(
        value is not None for value in (
            args.before_rps, args.after_rps, args.before_hourly_rate_thb,
            args.after_hourly_rate_thb, args.before_p95_ms, args.after_p95_ms,
        )
    )
    if has_optimization:
        before = costs.cost_per_1k_predictions(
            args.before_hourly_rate_thb, args.before_rps, args.expected_utilisation
        )
        after = costs.cost_per_1k_predictions(
            args.after_hourly_rate_thb, args.after_rps, args.expected_utilisation
        )
        optimisation = (
            f"| Change | {args.before_config} | {args.after_config} |\n"
            f"| Verified rate (THB/hour) | {args.before_hourly_rate_thb:.6f} | {args.after_hourly_rate_thb:.6f} |\n"
            f"| THB/1,000 at {int(args.expected_utilisation * 100)}% | {before:.6f} | {after:.6f} |\n"
            f"| Throughput (req/s) | {args.before_rps:.3f} | {args.after_rps:.3f} |\n"
            f"| Latency p95 (ms) | {args.before_p95_ms:.1f} | {args.after_p95_ms:.1f} |"
        )
    else:
        optimisation = (
            "Not demonstrated. Supply the before/after configurations, verified hourly rates, "
            "measured throughput, and p95 values after applying one change."
        )

    estimate_status = (
        f"Evidence file: `{args.estimate_evidence}`"
        if args.estimate_evidence and args.estimate_evidence.is_file()
        else "NOT EVIDENCED as a pre-run estimate. A value entered after the run does not meet the requirement."
    )
    content = f"""# Lab 5 — Cost report

Status: figures below are supplied by the operator. Verify each against the provider's dated regional price source and billing export.

Provider: Azure · serving configuration: `{args.instance}` · verified provisioned rate: {args.hourly_rate_thb:.6f} THB/hour
Rate source: {args.price_source}

## 1. Estimate before the run

{args.estimate:.2f} THB
{estimate_status}

## 2. Actual tagged billing

{args.actual:.2f} THB
Billing source: {args.billing_source}

## 3. Estimate gap

{gap:+.2f} THB ({gap_pct:+.1f}%).

{args.gap_explanation}

## 4. Breakdown by component

| Component | THB |
|---|---:|
{breakdown_rows}

Breakdown check: {breakdown_check}.

## 5. Cost per 1,000 predictions

Measured throughput: {args.rps:.3f} requests/second. The following provisioned-capacity model uses the verified hourly rate and assumes the listed utilisation is sustained; it is not an invoice forecast for scale-to-zero service.

| Utilisation | THB per 1,000 |
|---:|---:|
{cost_rows}

Selected planning assumption: {int(args.expected_utilisation * 100)}% utilisation. State why that matches the measured traffic pattern before submitting.

## 6. Applied optimisation

{optimisation}
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"estimate gap: {gap:+.2f} THB ({gap_pct:+.1f}%)")
    if abs(gap_pct) > 20:
        print("Warning: estimate gap is over 20%; reconcile the tagged billing export.")
    if len(supplied) == len(COMPONENTS) and args.actual > 0:
        if abs(sum(supplied) - args.actual) / args.actual > 0.20:
            print("Warning: component breakdown differs from the billing total by more than 20%.")
    if not has_optimization:
        print("Warning: the required before/after optimisation evidence is not supplied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
