"""Lab 4 — drift detection.

PSI and Kolmogorov–Smirnov, implemented directly so you can see what the numbers mean.
Evidently is allowed instead, but you must still be able to explain what your chosen
statistic measures and why your threshold is what it is.

    python -m monitoring.drift --reference data/raw/sensors.csv --current data/current.csv

A threshold copied from a tutorial is not a justified threshold, and Lab 4 grades the
justification, not the code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# A 0.20 threshold is intentionally below the common 0.25 "moderate" heuristic. With
# 6,000 reference readings and quantile bins, it catches a material shift before a full
# moderate shift while filtering small sampling noise. Validate it against this domain.
PSI_NO_CHANGE = 0.10
PSI_ALERT_THRESHOLD = 0.20


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ref_mean: float
    cur_mean: float
    verdict: str


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index.

    Bins the reference into quantiles and compares the proportion of mass falling in each.
    Sensitive to changes in shape, not only in mean — which is why a feature can drift
    badly while its average looks untouched.
    """
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Laplace smoothing: an empty bin would otherwise make the log term infinite.
    ref_prop = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    cur_prop = (cur_counts + 1) / (cur_counts.sum() + len(cur_counts))

    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic: the largest gap between the two CDFs.

    Complements PSI. KS is more sensitive to a shift in location; PSI to a change in shape.
    Reporting both, and noticing when they disagree, is worth more than either alone.
    """
    ref = np.sort(reference)
    cur = np.sort(current)
    pooled = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, pooled, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, pooled, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def verdict_for(score: float) -> str:
    if score < PSI_NO_CHANGE:
        return "stable"
    if score < PSI_ALERT_THRESHOLD:
        return "moderate"
    return "significant"


def compare(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> list[FeatureDrift]:
    for label, frame in (("reference", reference), ("current", current)):
        missing = set(features) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} inputs are missing required features: {sorted(missing)}")
        for feature in features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            if values.empty or values.isna().any() or not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"{label} feature {feature} is empty, null, or non-finite")

    results = []
    for feature in features:
        ref = reference[feature].to_numpy(dtype=float)
        cur = current[feature].to_numpy(dtype=float)
        score = psi(ref, cur)
        results.append(FeatureDrift(
            feature=feature,
            psi=round(score, 5),
            ks_statistic=round(ks_statistic(ref, cur), 5),
            ref_mean=round(float(ref.mean()), 4),
            cur_mean=round(float(cur.mean()), 4),
            verdict=verdict_for(score),
        ))
    return sorted(results, key=lambda r: r.psi, reverse=True)


def _metric_payload(results: list[FeatureDrift], threshold: float) -> bytes:
    lines = [
        "# TYPE drift_psi gauge",
        "# TYPE drift_ks_statistic gauge",
        "# TYPE drift_alert_threshold gauge",
        "# TYPE drift_alert gauge",
    ]
    for result in results:
        feature = result.feature.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        labels = f'{{feature="{feature}"}}'
        lines.extend(
            (
                f"drift_psi{labels} {result.psi}",
                f"drift_ks_statistic{labels} {result.ks_statistic}",
                f"drift_alert_threshold{labels} {threshold}",
                f"drift_alert{labels} {int(result.psi >= threshold)}",
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def push_metrics(pushgateway_url: str, results: list[FeatureDrift], threshold: float) -> None:
    target = pushgateway_url.rstrip("/") + "/metrics/job/itcs355-drift"
    headers = {"Content-Type": "text/plain; version=0.0.4; charset=utf-8"}
    token = os.environ.get("PROMETHEUS_PUSHGATEWAY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        target,
        data=_metric_payload(results, threshold),
        headers=headers,
        method="PUT",
    )
    with urlopen(request, timeout=15) as response:
        if response.status not in (200, 202):
            raise RuntimeError(f"metrics sink returned HTTP {response.status}")


def send_slack_alert(webhook_url: str, results: list[FeatureDrift], threshold: float) -> None:
    breached = [result for result in results if result.psi >= threshold]
    details = "\n".join(
        f"• {result.feature}: PSI {result.psi:.3f}, KS {result.ks_statistic:.3f}"
        for result in breached
    )
    payload = json.dumps(
        {
            "text": (
                f"ITCS355 drift alert at {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
                f"Threshold: PSI >= {threshold:.2f}\n{details}\n"
                "Check schema and source quality before deciding to retrain."
            )
        }
    ).encode("utf-8")
    request = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        if response.status not in (200, 201, 202, 204):
            raise RuntimeError(f"alert channel returned HTTP {response.status}")


def _resolve_secret(env_name: str, secret_name_env: str) -> str | None:
    direct_value = os.environ.get(env_name)
    if direct_value:
        return direct_value
    secret_name = os.environ.get(secret_name_env)
    if not secret_name:
        return None

    from cloudlayer.factory import get_adapter
    from src import config

    adapter = get_adapter(config.load(strict=False))
    return adapter.get_secret(secret_name)


def _injection_timestamp(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("injected_at_utc") or "") or None
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, data

    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=Path("data/raw/sensors.csv"))
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=PSI_ALERT_THRESHOLD,
                    help="alert at or above this PSI (default justified in the README)")
    ap.add_argument("--out", type=Path, default=Path("reports/drift.json"))
    ap.add_argument("--emit", action="store_true", help="send scores through the cloud adapter")
    ap.add_argument(
        "--pushgateway-url",
        default=os.environ.get("PROMETHEUS_PUSHGATEWAY_URL"),
        help="optional Prometheus Pushgateway URL; can also be set in the environment",
    )
    ap.add_argument(
        "--alert-webhook-url",
        default=os.environ.get("DRIFT_ALERT_WEBHOOK_URL"),
        help="Slack incoming webhook; preferably resolve it from Azure Key Vault",
    )
    ap.add_argument("--require-metric-delivery", action="store_true")
    ap.add_argument("--require-alert-delivery", action="store_true")
    ap.add_argument(
        "--injection-manifest",
        type=Path,
        default=Path("reports/lab4-injection.json"),
    )
    args = ap.parse_args()
    if args.threshold <= 0:
        ap.error("--threshold must be positive")

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    results = compare(reference, current, data.FEATURES)
    detected_at = datetime.now(timezone.utc)
    breached = [result for result in results if result.psi >= args.threshold]
    injected_at = _injection_timestamp(args.injection_manifest)
    detection_seconds = None
    injection_time = None
    if injected_at and breached:
        try:
            injection_time = datetime.fromisoformat(injected_at.replace("Z", "+00:00"))
            detection_seconds = round((detected_at - injection_time).total_seconds(), 3)
        except ValueError:
            injected_at = None

    print(f"{'feature':<22}{'psi':>10}{'ks':>10}  verdict")
    for r in results:
        print(f"{r.feature:<22}{r.psi:>10.5f}{r.ks_statistic:>10.5f}  {r.verdict}")

    metric_error = None
    if args.pushgateway_url:
        try:
            if not os.environ.get("PROMETHEUS_PUSHGATEWAY_TOKEN") and os.environ.get(
                "PROMETHEUS_PUSHGATEWAY_TOKEN_SECRET_NAME"
            ):
                os.environ["PROMETHEUS_PUSHGATEWAY_TOKEN"] = _resolve_secret(
                    "PROMETHEUS_PUSHGATEWAY_TOKEN", "PROMETHEUS_PUSHGATEWAY_TOKEN_SECRET_NAME"
                ) or ""
            push_metrics(args.pushgateway_url, results, args.threshold)
            print("Metrics written to the configured Prometheus Pushgateway.")
        except Exception as exc:
            metric_error = f"{type(exc).__name__}: metrics sink unavailable"
            print(metric_error, file=sys.stderr)
    elif args.require_metric_delivery and not args.emit:
        metric_error = "PROMETHEUS_PUSHGATEWAY_URL is required"
        print(metric_error, file=sys.stderr)

    adapter_metric_delivered = False
    if args.emit:
        try:
            from cloudlayer.factory import get_adapter
            adapter = get_adapter(config.load(strict=False))
            for r in results:
                adapter.emit_metric(f"drift_psi_{r.feature}", r.psi, unit="1")
            adapter_metric_delivered = True
        except Exception as exc:
            metric_error = f"{type(exc).__name__}: cloud metric delivery failed"
            print(metric_error, file=sys.stderr)

    alert_error = None
    alert_delivered_at = None
    if breached:
        print(f"\nALERT  {len(breached)} feature(s) above threshold {args.threshold}: "
              + ", ".join(r.feature for r in breached))
        print("Before you retrain: is this drift, or is it a broken upstream pipeline? "
              "Retraining on corrupted data destroys a working model faster than any "
              "schedule would.")
        try:
            alert_webhook_url = args.alert_webhook_url or _resolve_secret(
                "DRIFT_ALERT_WEBHOOK_URL", "DRIFT_ALERT_SECRET_NAME"
            )
        except Exception as exc:
            alert_webhook_url = None
            alert_error = f"{type(exc).__name__}: alert secret resolution failed"
            print(alert_error, file=sys.stderr)
        if alert_webhook_url:
            try:
                send_slack_alert(alert_webhook_url, results, args.threshold)
                alert_delivered_at = datetime.now(timezone.utc)
                print("Alert delivered to the configured Slack channel.")
            except (HTTPError, URLError, OSError, RuntimeError) as exc:
                alert_error = f"{type(exc).__name__}: alert delivery failed"
                print(alert_error, file=sys.stderr)
        elif args.require_alert_delivery:
            alert_error = "DRIFT_ALERT_WEBHOOK_URL is required"
            print(alert_error, file=sys.stderr)
    else:
        print(f"\nOK  no feature above threshold {args.threshold}")

    evidence = {
        "detected_at_utc": detected_at.isoformat(timespec="milliseconds"),
        "alert_delivered_at_utc": (
            alert_delivered_at.isoformat(timespec="milliseconds") if alert_delivered_at else None
        ),
        "injection_to_alert_seconds": (
            round((alert_delivered_at - injection_time).total_seconds(), 3)
            if alert_delivered_at and injection_time else None
        ),
        "threshold_psi": args.threshold,
        "breached_features": [result.feature for result in breached],
        "injected_at_utc": injected_at,
        "detection_seconds": detection_seconds,
        "alert_delivered": bool(breached and alert_delivered_at and not alert_error),
        "metric_delivered": bool((args.pushgateway_url or adapter_metric_delivered) and not metric_error),
        "results": [asdict(result) for result in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    if metric_error or alert_error:
        return 3
    return 2 if breached else 0


if __name__ == "__main__":
    raise SystemExit(main())
