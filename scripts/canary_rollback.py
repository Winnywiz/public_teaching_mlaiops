"""Run a 90/10, metric-gated canary and always restore stable traffic afterward."""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import roc_auc_score

from cloudlayer.factory import get_adapter
from src import config, data, seeds


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def weighted(
    active: list[dict[str, Any]], first: str, first_weight: int, second: str | None = None
) -> dict[str, int]:
    result = {item["name"]: 0 for item in active if item.get("name")}
    if first not in result:
        raise RuntimeError(f"revision {first!r} is not active")
    result[first] = first_weight
    if first_weight < 100:
        if second is None or second not in result or second == first:
            raise RuntimeError("need two active revisions for a canary traffic split")
        result[second] = 100 - first_weight
    return result


def snapshot(revisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: item.get(key) for key in ("name", "active", "traffic_weight", "running_state")}
        for item in revisions
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="itcs355-lab3")
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--instance", default="0.5cpu/1Gi")
    parser.add_argument("--max-requests", type=int, default=1200)
    parser.add_argument("--min-per-arm", type=int, default=30)
    parser.add_argument("--check-every", type=int, default=25)
    parser.add_argument("--auc-drop", type=float, default=0.015)
    parser.add_argument("--p95-target-ms", type=float, default=200)
    parser.add_argument("--consecutive-windows", type=int, default=2)
    parser.add_argument("--ready-timeout", type=int, default=300)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.max_requests < args.min_per_arm * 2:
        parser.error("--max-requests must allow at least --min-per-arm samples from both variants")

    cfg = config.load(strict=True)
    if cfg.provider != "azure":
        raise SystemExit("This canary implementation uses Azure Container Apps revision traffic.")
    if not cfg.raw_path.exists():
        raise SystemExit(f"{cfg.raw_path} is missing; run `make data` or `dvc pull` first")
    adapter = get_adapter(cfg)
    revisions = adapter.list_revisions(args.endpoint)
    active = [item for item in revisions if item.get("active") and item.get("name")]
    if not active:
        raise SystemExit("No active revisions found; deploy and smoke-test the stable version first.")
    stable = max(active, key=lambda item: int(item.get("traffic_weight", 0)))
    stable_revision = str(stable["name"])
    stable_version = str(stable.get("model_version", ""))
    if not stable_version or stable_version == args.candidate_version:
        raise SystemExit("Could not distinguish the requested candidate from the stable revision.")

    report_path = Path(args.out) if args.out else Path("reports") / f"lab3-rollback-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md"
    evidence: dict[str, Any] = {
        "started_at_utc": utc_now(),
        "stable_revision": stable_revision,
        "before": snapshot(revisions),
        "traffic_90_10_at_utc": None,
        "rollback_at_utc": None,
        "after": [],
        "detection": None,
        "errors": [],
    }
    candidate_revision: str | None = None
    rollback_error: str | None = None
    started_monitoring: float | None = None

    try:
        # Pin by explicit revision name before creating a new revision. This prevents
        # Container Apps' implicit "latest revision" route from receiving all traffic.
        adapter.set_traffic_weights(args.endpoint, weighted(active, stable_revision, 100))
        app_url = adapter.deploy_canary_revision(
            f"models:/{cfg.model_registry_name}/{args.candidate_version}",
            args.endpoint,
            args.instance,
        )
        deadline = time.monotonic() + args.ready_timeout
        while time.monotonic() < deadline:
            current = adapter.list_revisions(args.endpoint)
            candidate = next((item for item in current
                              if str(item.get("model_version")) == args.candidate_version
                              and item.get("name") != stable_revision), None)
            if candidate and candidate.get("active") and "running" in str(candidate.get("running_state", "")).lower():
                candidate_revision = str(candidate["name"])
                break
            time.sleep(10)
        if not candidate_revision:
            raise TimeoutError(f"Candidate revision did not become ready within {args.ready_timeout}s")

        current = adapter.list_revisions(args.endpoint)
        active = [item for item in current if item.get("active") and item.get("name")]
        adapter.set_traffic_weights(
            args.endpoint, weighted(active, stable_revision, 90, candidate_revision)
        )
        evidence["traffic_90_10_at_utc"] = utc_now()
        evidence["traffic_90_10"] = snapshot(adapter.list_revisions(args.endpoint))
        evidence["endpoint"] = app_url
        evidence["traffic"] = "90% stable / 10% candidate"

        frame = data.load_raw(cfg.raw_path)
        _, _, test_df = data.split(frame, seed=seeds.DEFAULT_SEED)
        indices = list(range(len(test_df)))
        random.Random(seeds.DEFAULT_SEED + 31).shuffle(indices)
        labels: dict[str, list[int]] = {"control": [], "canary": []}
        scores: dict[str, list[float]] = {"control": [], "canary": []}
        arm_for_version = {stable_version: "control", args.candidate_version: "canary"}
        consecutive = 0
        detection: dict[str, Any] | None = None
        started_monitoring = time.monotonic()

        for request_number in range(1, args.max_requests + 1):
            row = test_df.iloc[indices[(request_number - 1) % len(indices)]]
            payload = {feature: float(row[feature]) for feature in data.FEATURES}
            # Reuse the resolved URL from deployment; passing a resource name here
            # would execute `az containerapp show` for every single monitor request.
            response = adapter.invoke(app_url, payload)
            arm = arm_for_version.get(str(response.get("model_version")))
            if arm is None:
                raise RuntimeError("response returned an unrecognized model version; stopped safely")
            labels[arm].append(int(row[data.TARGET]))
            scores[arm].append(float(response["probability"]))

            if request_number % args.check_every != 0:
                continue
            if min(len(scores["control"]), len(scores["canary"])) < args.min_per_arm:
                continue
            if len(set(labels["control"])) < 2 or len(set(labels["canary"])) < 2:
                continue
            control_auc = float(roc_auc_score(labels["control"], scores["control"]))
            canary_auc = float(roc_auc_score(labels["canary"], scores["canary"]))
            degradation = control_auc - canary_auc
            evidence["last_metric_window"] = {
                "request": request_number,
                "control_samples": len(scores["control"]),
                "canary_samples": len(scores["canary"]),
                "control_roc_auc": control_auc,
                "canary_roc_auc": canary_auc,
                "auc_drop": degradation,
            }
            consecutive = consecutive + 1 if degradation >= args.auc_drop else 0
            if consecutive >= args.consecutive_windows:
                elapsed = time.monotonic() - started_monitoring
                detection = {
                    **evidence["last_metric_window"],
                    "detected_at_utc": utc_now(),
                    "detection_seconds": round(elapsed, 3),
                    "rule": f"candidate ROC-AUC lower by at least {args.auc_drop:.4f} in "
                            f"{args.consecutive_windows} consecutive windows",
                }
                break
        evidence["detection"] = detection
        if detection is None:
            evidence["errors"].append(
                f"No degradation crossed the metric threshold after {args.max_requests} requests."
            )
    except Exception as exc:
        evidence["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        evidence["rollback_at_utc"] = utc_now()
        try:
            current = adapter.list_revisions(args.endpoint)
            active = [item for item in current if item.get("active") and item.get("name")]
            adapter.set_traffic_weights(args.endpoint, weighted(active, stable_revision, 100))
            evidence["after"] = snapshot(adapter.list_revisions(args.endpoint))
            stable_weight = next(
                (int(item.get("traffic_weight", 0)) for item in evidence["after"]
                 if item.get("name") == stable_revision), 0,
            )
            candidate_weight = next(
                (int(item.get("traffic_weight", 0)) for item in evidence["after"]
                 if item.get("name") == candidate_revision), 0,
            ) if candidate_revision else 0
            evidence["rollback_verified"] = stable_weight == 100 and candidate_weight == 0
            if not evidence["rollback_verified"]:
                rollback_error = "Traffic weights did not verify as stable=100%, candidate=0%."
        except Exception as exc:
            rollback_error = f"{type(exc).__name__}: {exc}"
            evidence["rollback_verified"] = False
        if rollback_error:
            evidence["errors"].append("ROLLBACK VERIFICATION FAILED: " + rollback_error)

        report_path.parent.mkdir(parents=True, exist_ok=True)
        detection = evidence.get("detection")
        before_table = "\n".join(
            f"| `{item.get('name')}` | {item.get('traffic_weight', 0)}% | {item.get('running_state')} |"
            for item in evidence.get("before", [])
        ) or "| — | — | no prior revision data |"
        canary_table = "\n".join(
            f"| `{item.get('name')}` | {item.get('traffic_weight', 0)}% | {item.get('running_state')} |"
            for item in evidence.get("traffic_90_10", [])
        ) or "| — | — | 90/10 stage was not reached |"
        after_table = "\n".join(
            f"| `{item.get('name')}` | {item.get('traffic_weight', 0)}% | {item.get('running_state')} |"
            for item in evidence.get("after", [])
        ) or "| — | — | rollback not verified |"
        metric_text = (
            f"ROC-AUC: control {detection['control_roc_auc']:.4f} vs opaque canary arm "
            f"{detection['canary_roc_auc']:.4f} (drop {detection['auc_drop']:.4f}); "
            f"{detection['canary_samples']} canary observations."
            if detection else f"Not detected: {evidence.get('errors', ['threshold not crossed'])[0]}"
        )
        detection_time = f"{detection['detection_seconds']:.3f} s" if detection else "Not detected"
        rollback_text = "PASS" if evidence.get("rollback_verified") else "FAILED — inspect the app's revision traffic immediately"
        errors = "\n".join(f"- {item}" for item in evidence.get("errors", [])) or "- None"
        report_path.write_text(f"""# Lab 3 — canary and rollback evidence

- Start (UTC): {evidence['started_at_utc']}
- 90/10 traffic applied (UTC): {evidence.get('traffic_90_10_at_utc') or 'not reached'}
- Rollback applied (UTC): {evidence.get('rollback_at_utc')}
- Rollback verification: **{rollback_text}**
- Detection rule: ROC-AUC on labeled holdout requests, grouped under opaque `control` / `canary` labels; model version strings are not used to decide degradation.
- Metric result: {metric_text}
- Detection time after 90/10: {detection_time}
- Previously stated service p95 target: {args.p95_target_ms:.0f} ms (the canary detector uses ROC-AUC instead)

## Timestamped traffic evidence

### Before
| Revision | Traffic | State |
|---|---:|---|
{before_table}

### Canary at 90/10
| Revision | Traffic | State |
|---|---:|---|
{canary_table}

### After rollback
| Revision | Traffic | State |
|---|---:|---|
{after_table}

## Five-line write-up

1. Metric: {metric_text}
2. Detection took {detection_time}; the monitoring loop checked every {args.check_every} requests and required {args.consecutive_windows} consecutive windows.
3. A faster label/feedback path and smaller predeclared minimum sample count would detect earlier; both increase false-alarm risk.
4. At 50/50, the canary would collect about five times as many samples per total request; this is a traffic-volume inference, not a separate 50/50 run.
5. Rollback evidence: traffic was pinned back to the original stable revision at {evidence.get('rollback_at_utc')}; verification status is **{rollback_text}**.

Errors and caveats:
{errors}

This experiment uses the course's labeled held-out dataset as immediate feedback. Real production labels
may arrive later; a production alert must use timely service/business metrics and an explicit delay-aware
evaluation policy. A failed rollback verification requires immediate manual traffic correction.
""", encoding="utf-8")
        print(f"evidence: {report_path}")
        print(f"rollback_verified={evidence.get('rollback_verified', False)}")

    if not evidence.get("rollback_verified"):
        raise SystemExit(2)
    if evidence.get("detection") is None:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
