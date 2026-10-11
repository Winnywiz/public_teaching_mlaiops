"""Lab 4 monitoring and drift regression tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoring import drift, metrics
from monitoring.drift import FeatureDrift, _metric_payload, compare, ks_statistic, psi, verdict_for


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.reset_metrics_for_test()
    yield
    metrics.reset_metrics_for_test()


def test_metrics_count_requests_by_status_and_bound_route_labels():
    metrics.observe_request("POST", "/predict", 200, 17.0)
    metrics.observe_request("POST", "/predict", 422, 3.0)
    metrics.observe_request("POST", "/predict", 500, 701.0)
    metrics.observe_request("GET", "/predict/secret-id", 404, 2.0)

    exposition = metrics.render_prometheus("sha-123")
    assert 'http_requests_total{method="POST",route="/predict",status_class="2xx"} 1' in exposition
    assert 'http_requests_total{method="POST",route="/predict",status_class="4xx"} 1' in exposition
    assert 'http_requests_total{method="POST",route="/predict",status_class="5xx"} 1' in exposition
    assert 'route="/other"' in exposition
    assert "secret-id" not in exposition


def test_latency_histogram_has_cumulative_buckets_and_model_version():
    metrics.observe_request("POST", "/predict", 200, 17.0)
    metrics.observe_request("POST", "/predict", 200, 701.0)

    exposition = metrics.render_prometheus("release-abc")
    assert 'request_latency_ms_bucket{le="25",method="POST",route="/predict"} 1' in exposition
    assert 'request_latency_ms_bucket{le="1000",method="POST",route="/predict"} 2' in exposition
    assert 'model_version_info{model_version="release-abc"} 1' in exposition


def test_feature_distribution_is_rolling_and_bounded():
    metrics.observe_features([{"temp_c": float(value)} for value in range(501)])

    exposition = metrics.render_prometheus("local")
    assert 'feature_rolling_samples{feature="temp_c"} 500' in exposition
    assert 'feature_rolling_mean{feature="temp_c"} 250.500000' in exposition
    assert 'feature_rolling_stddev{feature="temp_c"}' in exposition


def test_feature_metrics_ignore_non_finite_values():
    metrics.observe_features([{"temp_c": 2.0}, {"temp_c": float("nan")}, {"temp_c": "bad"}])

    exposition = metrics.render_prometheus("local")
    assert 'feature_rolling_samples{feature="temp_c"} 1' in exposition


def test_psi_and_ks_detect_a_material_mean_shift():
    reference = np.linspace(0.0, 1.0, 6000)
    current = reference + 0.4

    assert psi(reference, current) >= 0.20
    assert ks_statistic(reference, current) > 0.30
    assert verdict_for(0.05) == "stable"
    assert verdict_for(0.15) == "moderate"
    assert verdict_for(0.20) == "significant"


def test_psi_is_stable_for_same_distribution():
    rng = np.random.default_rng(17)
    reference = rng.normal(0.0, 1.0, size=6000)
    current = rng.normal(0.0, 1.0, size=6000)

    assert psi(reference, current) < 0.05


def test_compare_rejects_missing_and_non_finite_contract_data():
    reference = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing required features"):
        compare(reference, pd.DataFrame({"x": [1.0]}), ["x", "y"])

    invalid = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="empty, null, or non-finite"):
        compare(invalid, reference, ["x"])


def test_metric_payload_contains_psi_threshold_and_alert_signal():
    results = [FeatureDrift("temp_c", 0.42, 0.51, 60.0, 66.0, "significant")]

    payload = _metric_payload(results, 0.20).decode("utf-8")
    assert 'drift_psi{feature="temp_c"} 0.42' in payload
    assert 'drift_alert_threshold{feature="temp_c"} 0.2' in payload
    assert 'drift_alert{feature="temp_c"} 1' in payload


def test_pushgateway_delivery_uses_put_and_keeps_bearer_token_out_of_payload(monkeypatch):
    captured = {}

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        captured.update(request=request, timeout=timeout)
        return Response()

    monkeypatch.setattr(drift, "urlopen", fake_urlopen)
    monkeypatch.setenv("PROMETHEUS_PUSHGATEWAY_TOKEN", "local-test-token")
    score = FeatureDrift("temp_c", 0.42, 0.51, 60.0, 66.0, "significant")

    drift.push_metrics("https://push.example.test", [score], 0.20)

    request = captured["request"]
    assert request.get_method() == "PUT"
    assert request.full_url.endswith("/metrics/job/itcs355-drift")
    assert request.get_header("Authorization") == "Bearer local-test-token"
    assert b"drift_psi{feature=\"temp_c\"} 0.42" in request.data
    assert b"local-test-token" not in request.data


def test_slack_alert_only_contains_scores_for_breached_features(monkeypatch):
    captured = {}

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        captured.update(request=request, timeout=timeout)
        return Response()

    monkeypatch.setattr(drift, "urlopen", fake_urlopen)
    scores = [
        FeatureDrift("temp_c", 0.42, 0.51, 60.0, 66.0, "significant"),
        FeatureDrift("load_pct", 0.03, 0.04, 60.0, 60.1, "stable"),
    ]

    drift.send_slack_alert("https://hooks.example.test/secret", scores, 0.20)

    payload = json.loads(captured["request"].data)
    assert "temp_c" in payload["text"]
    assert "load_pct" not in payload["text"]
    assert "secret" not in payload["text"]


def test_dashboard_covers_the_five_required_monitoring_signals():
    dashboard_path = Path(__file__).resolve().parents[1] / "monitoring" / "dashboard.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    panels = {panel["title"]: panel for panel in dashboard["panels"]}

    assert "Request rate" in panels
    assert "4xx and 5xx error rate" in panels
    assert "Prediction latency p50, p95, p99" in panels
    assert "Recent input distribution: temperature" in panels
    assert "Model version in production" in panels
    assert "drift_psi" in panels["Scheduled drift score (PSI)"]["targets"][0]["expr"]  # Keep the scheduled PSI query covered.
