"""Small, dependency-free Prometheus exporter for the inference service.

Request labels stay deliberately low-cardinality. Feature values are not retained; only
a bounded rolling window of aggregates is exposed.
"""
from __future__ import annotations

import math
import statistics
import threading
from collections import defaultdict, deque
from collections.abc import Iterable


FEATURES = (
    "temp_c",
    "vibration_mm_s",
    "pressure_kpa",
    "hours_since_service",
    "load_pct",
    "ambient_humidity",
)
WINDOW_SIZE = 500
LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 200, 500, 1000, 2500, 5000)
_KNOWN_ROUTES = {"/health", "/ready", "/metrics", "/predict", "/predict/batch"}
_LOCK = threading.RLock()
_REQUESTS: defaultdict[tuple[str, str, str], int] = defaultdict(int)
_LATENCY_COUNTS: defaultdict[tuple[str, str], list[int]] = defaultdict(
    lambda: [0] * (len(LATENCY_BUCKETS_MS) + 1)
)
_LATENCY_SUMS: defaultdict[tuple[str, str], float] = defaultdict(float)
_LATENCY_TOTALS: defaultdict[tuple[str, str], int] = defaultdict(int)
_FEATURE_WINDOWS = {feature: deque(maxlen=WINDOW_SIZE) for feature in FEATURES}


def _route(path: str) -> str:
    return path if path in _KNOWN_ROUTES else "/other"


def observe_request(method: str, path: str, status_code: int, latency_ms: float) -> None:
    route = _route(path)
    status_class = f"{status_code // 100}xx"
    safe_latency = max(0.0, latency_ms) if math.isfinite(latency_ms) else 0.0
    labels = (method.upper(), route, status_class)
    histogram_labels = (method.upper(), route)
    with _LOCK:
        _REQUESTS[labels] += 1
        for index, bound in enumerate(LATENCY_BUCKETS_MS):
            if safe_latency <= bound:
                _LATENCY_COUNTS[histogram_labels][index] += 1
        _LATENCY_COUNTS[histogram_labels][-1] += 1
        _LATENCY_SUMS[histogram_labels] += safe_latency
        _LATENCY_TOTALS[histogram_labels] += 1


def observe_features(rows: Iterable[dict[str, object]]) -> None:
    with _LOCK:
        for row in rows:
            for feature in FEATURES:
                value = row.get(feature)
                try:
                    number = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    _FEATURE_WINDOWS[feature].append(number)


def _escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(**values: object) -> str:
    fields = ",".join(f'{key}="{_escape(value)}"' for key, value in sorted(values.items()))
    return "{" + fields + "}"


def render_prometheus(model_version: str) -> str:
    lines = [
        "# HELP http_requests_total HTTP requests served by method, route, and status class.",
        "# TYPE http_requests_total counter",
    ]
    with _LOCK:
        for (method, route, status_class), count in sorted(_REQUESTS.items()):
            lines.append(
                f"http_requests_total{_labels(method=method, route=route, status_class=status_class)} {count}"
            )

        lines.extend(
            (
                "# HELP request_latency_ms Request latency in milliseconds.",
                "# TYPE request_latency_ms histogram",
            )
        )
        for (method, route), counts in sorted(_LATENCY_COUNTS.items()):
            for index, bound in enumerate(LATENCY_BUCKETS_MS):
                lines.append(
                    f'request_latency_ms_bucket{_labels(method=method, route=route, le=bound)} '
                    f"{counts[index]}"
                )
            lines.append(
                f'request_latency_ms_bucket{_labels(method=method, route=route, le="+Inf")} '
                f"{counts[-1]}"
            )
            labels = _labels(method=method, route=route)
            lines.append(f"request_latency_ms_sum{labels} {_LATENCY_SUMS[(method, route)]:.6f}")
            lines.append(f"request_latency_ms_count{labels} {_LATENCY_TOTALS[(method, route)]}")

        lines.extend(
            (
                "# HELP feature_rolling_mean Rolling mean for recent prediction inputs.",
                "# TYPE feature_rolling_mean gauge",
                "# HELP feature_rolling_stddev Rolling population standard deviation for recent prediction inputs.",
                "# TYPE feature_rolling_stddev gauge",
                "# HELP feature_rolling_samples Number of recent values contributing to feature aggregates.",
                "# TYPE feature_rolling_samples gauge",
            )
        )
        for feature, window in _FEATURE_WINDOWS.items():
            if not window:
                continue
            values = tuple(window)
            labels = _labels(feature=feature)
            lines.append(f"feature_rolling_mean{labels} {statistics.fmean(values):.6f}")
            lines.append(f"feature_rolling_stddev{labels} {statistics.pstdev(values):.6f}")
            lines.append(f"feature_rolling_samples{labels} {len(values)}")

    lines.extend(
        (
            "# HELP model_version_info The model version currently serving requests.",
            "# TYPE model_version_info gauge",
            f"model_version_info{_labels(model_version=model_version or 'unknown')} 1",
        )
    )
    return "\n".join(lines) + "\n"


def reset_metrics_for_test() -> None:
    """Clear process-local metrics between isolated unit tests."""
    with _LOCK:
        _REQUESTS.clear()
        _LATENCY_COUNTS.clear()
        _LATENCY_SUMS.clear()
        _LATENCY_TOTALS.clear()
        for window in _FEATURE_WINDOWS.values():
            window.clear()
