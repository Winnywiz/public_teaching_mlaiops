"""Lab 4 — deliberately shift a feature's distribution.

    python scripts/inject_drift.py --feature temp_c --mode shift --magnitude 6

Three modes, and they are not equivalent:
  shift   moves the mean         — KS reacts strongly, PSI moderately
  scale   changes the spread     — PSI reacts, the mean barely moves
  mix     reweights the machines — the realistic one, and the hardest to spot

Run your detector against each. Which statistic catches which is exactly what Quiz 4 asks.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/raw/sensors.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/current.csv"))
    ap.add_argument("--feature", default="temp_c")
    ap.add_argument("--mode", choices=["shift", "scale", "mix"], default="shift")
    ap.add_argument("--magnitude", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--manifest", type=Path, default=Path("reports/lab4-injection.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.source)
    if args.feature not in df.columns:
        raise SystemExit(f"feature {args.feature!r} is not present in {args.source}")
    injected_at = datetime.now(timezone.utc)
    before = df[args.feature].mean()

    if args.mode == "shift":
        df[args.feature] = df[args.feature] + args.magnitude
    elif args.mode == "scale":
        mean = df[args.feature].mean()
        df[args.feature] = mean + (df[args.feature] - mean) * args.magnitude
    else:  # mix — over-sample a subset of machines, as a fleet change would
        machines = df["machine_id"].unique()
        favoured = rng.choice(machines, size=max(1, len(machines) // 5), replace=False)
        weights = np.where(df["machine_id"].isin(favoured), 5.0, 1.0)
        df = df.sample(n=len(df), replace=True, weights=weights, random_state=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "injected_at_utc": injected_at.isoformat(timespec="milliseconds"),
                "source": str(args.source),
                "output": str(args.out),
                "feature": args.feature,
                "mode": args.mode,
                "magnitude": args.magnitude,
                "seed": args.seed,
                "rows": len(df),
                "mean_before": float(before),
                "mean_after": float(df[args.feature].mean()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}  mode={args.mode}  feature={args.feature}")
    print(f"  mean {before:.4f} -> {df[args.feature].mean():.4f}")
    print(f"  injected_at_utc={injected_at.isoformat(timespec='milliseconds')}")
    print(f"\nNow run: python -m monitoring.drift --current {args.out}")
    print("Record the time from injection to alert. That is your detection latency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
