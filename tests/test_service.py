"""Lab 3/4 — service contract tests."""
from __future__ import annotations

import os
import json
import logging

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src import config, data, seeds  # noqa: E402
from service import app as service_app  # noqa: E402

RAW = config.REPO_ROOT / "data" / "raw" / "sensors.csv"
VALID = {
    "temp_c": 78.4, "vibration_mm_s": 3.1, "pressure_kpa": 315.2,
    "hours_since_service": 4200.0, "load_pct": 68.0, "ambient_humidity": 55.0,
}


def test_registry_loader_only_trusts_reviewed_skops_types(tmp_path, monkeypatch):
    import mlflow
    import skops.io
    from mlflow.models import Model
    from types import SimpleNamespace

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_file = model_dir / "model.skops"
    model_file.touch()
    metadata = SimpleNamespace(flavors={
        "sklearn": {"pickled_model": "model.skops", "serialization_format": "skops"}
    })
    requested_uris = []
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts",
        lambda artifact_uri: requested_uris.append(artifact_uri) or str(model_dir),
    )
    monkeypatch.setattr(Model, "load", lambda path: metadata)

    loaded = object()
    observed = {}

    def fake_load(path, trusted):
        observed.update(path=path, trusted=trusted)
        return loaded

    monkeypatch.setattr(
        skops.io, "get_untrusted_types", lambda file: ["sklearn.tree._tree.Tree"]
    )
    monkeypatch.setattr(skops.io, "load", fake_load)
    assert service_app._load_registered_sklearn_model("registry", "1") is loaded
    assert requested_uris == ["models:/registry/1"]
    assert observed == {
        "path": str(model_file),
        "trusted": ["sklearn.tree._tree.Tree"],
    }

    monkeypatch.setattr(skops.io, "get_untrusted_types", lambda file: ["unexpected.Payload"])
    with pytest.raises(RuntimeError, match="unreviewed skops types"):
        service_app._load_registered_sklearn_model("registry", "1")


@pytest.fixture(scope="module")
def client():
    if not RAW.exists():
        pytest.skip("run `make data` first")
    model_path = config.REPO_ROOT / "reports" / "model.joblib"
    if not model_path.exists():
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        seed = seeds.set_all()
        df = data.load_raw(RAW)
        train_df, _, _ = data.split(df, seed=seed)
        model = RandomForestClassifier(n_estimators=60, max_depth=8, random_state=seed)
        model.fit(train_df[data.FEATURES], train_df[data.TARGET])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["MODEL_VERSION"] = "test-1"
    os.environ.pop("MODEL_REGISTRY_NAME", None)

    from service.app import app
    with TestClient(app) as c:
        yield c


def test_health_and_ready_are_different(client):
    """Liveness passes whenever the process is up. Readiness only when it can score."""
    assert client.get("/health").json()["status"] == "alive"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    for response in (client.get("/health"), client.get("/ready")):
        assert response.json()["model_version"] == "test-1"
        assert response.json()["request_id"]


def test_predict_returns_probability_and_version(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "test-1"
    assert r.headers["x-model-version"] == "test-1"
    assert r.headers["x-request-id"]
    assert body["request_id"] == r.headers["x-request-id"]


def test_out_of_range_input_is_rejected(client):
    bad = {**VALID, "load_pct": 250.0}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
    assert response.json()["model_version"] == "test-1"
    assert response.json()["request_id"]
    assert any("load_pct" in item["field"] for item in response.json()["detail"])


def test_unknown_field_is_rejected(client):
    """extra='forbid'. A silently ignored field is how a caller ends up sending a feature
    you never read while believing it matters."""
    assert client.post("/predict", json={**VALID, "surprise": 1}).status_code == 422


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in VALID.items() if k != "temp_c"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_batch_matches_singles(client):
    rows = [VALID, {**VALID, "temp_c": 92.0}]
    batch = client.post("/predict/batch", json={"rows": rows}).json()["probabilities"]
    singles = [client.post("/predict", json=r).json()["probability"] for r in rows]
    assert batch == pytest.approx(singles, abs=1e-9)


def test_batch_size_limit_enforced(client):
    r = client.post("/predict/batch", json={"rows": [VALID] * 101})
    assert r.status_code == 422
    assert r.json()["model_version"] == "test-1"


def test_batch_response_carries_request_context(client):
    response = client.post(
        "/predict/batch", json={"rows": [VALID]}, headers={"x-request-id": "lab3-test-42"}
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == "lab3-test-42"
    assert response.json()["model_version"] == "test-1"


def test_oversized_metadata_is_rejected(client):
    response = client.post("/predict", json={**VALID, "metadata": {"padding": "x" * 65_600}})
    assert response.status_code == 422
    assert response.json()["request_id"]
    assert response.json()["model_version"] == "test-1"


def test_nan_is_rejected(client):
    payload = json.dumps({**VALID, "temp_c": "NaN"})
    response = client.post("/predict", content=payload, headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_request_log_is_structured_json_with_latency_and_version(client, caplog):
    with caplog.at_level(logging.INFO, logger="service"):
        client.post("/predict", json=VALID, headers={"x-request-id": "log-check"})
    events = [json.loads(record.getMessage()) for record in caplog.records if record.name == "service"]
    request_event = next(event for event in events if event.get("event") == "http_request")
    assert request_event["request_id"] == "log-check"
    assert request_event["model_version"] == "test-1"
    assert request_event["latency_ms"] >= 0


def test_ready_and_prediction_fail_closed_if_model_disappears(client):
    from service.app import STATE

    model = STATE["model"]
    try:
        STATE["model"] = None
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["model_version"] == "test-1"
        assert client.post("/predict", json=VALID).status_code == 503
    finally:
        STATE["model"] = model
