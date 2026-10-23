"""Lab 5 — the registration gate.

Compares a candidate against the currently registered production model. Exits non-zero
when the candidate should NOT be registered, which is what makes the pipeline condition
real rather than decorative.

    python scripts/evaluation_gate.py --metrics reports/metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Historical Lab 2 selected model scored 0.8533 test ROC-AUC. This 0.82 floor leaves
# 0.033 headroom for seed/data variation while blocking a material quality regression.
MIN_ABSOLUTE = 0.82
MIN_IMPROVEMENT = 0.002


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--metric", default="test_roc_auc")
    ap.add_argument("--incumbent", type=float, default=None,
                    help="optional current production score for an additional margin check")
    ap.add_argument("--decision-out", type=Path,
                    help="write a pass receipt for downstream pipeline steps")
    args = ap.parse_args()

    payload = json.loads(args.metrics.read_text())
    candidate = float(payload[args.metric])
    if not math.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
        raise SystemExit(f"invalid {args.metric} score: {candidate}")
    print(f"candidate {args.metric} = {candidate:.4f}")

    if candidate < MIN_ABSOLUTE:
        print(f"GATE FAIL  below absolute floor {MIN_ABSOLUTE}")
        return 1

    incumbent = args.incumbent
    if incumbent is not None:
        if not math.isfinite(incumbent) or not 0.0 <= incumbent <= 1.0:
            raise SystemExit(f"invalid incumbent score: {incumbent}")
        delta = candidate - incumbent
        print(f"incumbent {incumbent:.4f}  delta {delta:+.4f}  required {MIN_IMPROVEMENT:+.4f}")
        if delta < MIN_IMPROVEMENT:
            print("GATE FAIL  improvement within noise; not registering")
            return 1
    else:
        print(f"bootstrap threshold  {MIN_ABSOLUTE:.4f}; this is a real score gate")

    print("GATE PASS")
    if args.decision_out:
        args.decision_out.parent.mkdir(parents=True, exist_ok=True)
        args.decision_out.write_text("pass\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
