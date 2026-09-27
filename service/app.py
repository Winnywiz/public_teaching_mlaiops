"""Lab 3 — inference service.

Provider-neutral by construction: the model arrives through the adapter, and the same
container image deploys to SageMaker, Azure ML, or Vertex AI. Route paths differ per
platform; that difference belongs in cloudlayer/, never here.

Run locally:  uvicorn service.app:app --port 8080
"""
from __future__ import annotations

import logging
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from service.schemas import BatchRequest, BatchResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger("service")

STATE: dict[str, Any] = {"model": None, "version": os.environ.get("MODEL_VERSION", "unknown")}
_TRUSTED_SKOPS_TYPES = frozenset({"sklearn.tree._tree.Tree"})


def _log_event(level: int, event: str, **fields: Any) -> None:
    log.log(level, json.dumps({"event": event, **fields}, separators=(",", ":"), default=str))


def _safe_request_id(value: str | None) -> str:
    if value and len(value) <= 128 and all(char.isprintable() and char not in '"\\' for char in value):
        return value
    return str(uuid.uuid4())


def _load_registered_sklearn_model(name: str, version: str):
    """Load one pinned registry artifact, allowing only the reviewed Random Forest type.

    Newer skops releases classify sklearn's tree-storage type as untrusted on some Python
    versions. The Lab 2 Random Forest artifact is produced by our own training pipeline, so
    explicitly allow that one type; reject every other unreviewed type before deserializing.
    """
    import mlflow
    import skops.io
    from mlflow.models import Model

    model_uri = f"models:/{name}/{version}"
    artifact_path = Path(mlflow.artifacts.download_artifacts(model_uri))
    sklearn_flavor = (Model.load(str(artifact_path)).flavors or {}).get("sklearn")
    if not sklearn_flavor:
        raise RuntimeError(f"Registry model {name}/{version} has no MLflow sklearn flavor")
    if sklearn_flavor.get("serialization_format") != "skops":
        raise RuntimeError(
            f"Registry model {name}/{version} is not in the reviewed skops serialization format"
        )

    model_file = artifact_path / sklearn_flavor["pickled_model"]
    if not model_file.is_file():
        raise RuntimeError(f"Registry model {name}/{version} is missing its sklearn artifact")
    untrusted_types = set(skops.io.get_untrusted_types(file=str(model_file)))
    unsupported_types = untrusted_types - _TRUSTED_SKOPS_TYPES
    if unsupported_types:
        raise RuntimeError(
            f"Registry model {name}/{version} contains unreviewed skops types: "
            f"{sorted(unsupported_types)}"
        )
    return skops.io.load(str(model_file), trusted=sorted(untrusted_types))


def _load_model():
    """Load once, at startup. Never per request.

    Loading per request is the commonest cause of a p99 that looks nothing like p50, and
    it is the first thing to check when your latency distribution has a long tail.
    """
    name = os.environ.get("MODEL_REGISTRY_NAME")
    version = os.environ.get("MODEL_VERSION")
    if name:
        if not version:
            raise RuntimeError("MODEL_REGISTRY_NAME requires MODEL_VERSION")
        import mlflow  # imported lazily so tests can run without a registry

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise RuntimeError("MLFLOW_TRACKING_URI is required for registry-backed serving")
        mlflow.set_tracking_uri(tracking_uri)
        return _load_registered_sklearn_model(name, version)

    # Fallback for local development and tests only. Submitting this is not acceptable:
    # your deployed service must load a registered version.
    import joblib

    path = Path(os.environ.get("MODEL_PATH", "reports/model.joblib"))
    if not path.exists():
        raise RuntimeError(
            "No model available. Set MODEL_REGISTRY_NAME and MODEL_VERSION, or MODEL_PATH."
        )
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["version"] = os.environ.get("MODEL_VERSION", "unknown")
    try:
        STATE["model"] = _load_model()
        _log_event(logging.INFO, "model_loaded", model_version=STATE["version"])
    except Exception as exc:  # readiness stays false; liveness still passes
        STATE["model"] = None
        _log_event(
            logging.ERROR,
            "model_load_failed",
            model_version=STATE["version"],
            error_type=type(exc).__name__,
        )
    yield
    STATE["model"] = None


app = FastAPI(title="ITCS355 inference", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = _safe_request_id(request.headers.get("x-request-id"))
    request.state.request_id = request_id
    started = time.perf_counter()
    request.state.started_at = started
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-model-version"] = str(STATE["version"])
    if response.headers.get("content-type", "").startswith("application/json"):
        body = b"".join([chunk async for chunk in response.body_iterator])
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict):
            payload.setdefault("request_id", request_id)
            payload.setdefault("model_version", str(STATE["version"]))
            headers = {
                key: value for key, value in response.headers.items()
                if key.lower() not in {"content-length", "content-type"}
            }
            response = JSONResponse(
                payload,
                status_code=response.status_code,
                headers=headers,
                background=response.background,
            )
    _log_event(
        logging.INFO,
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency_ms, 3),
        model_version=str(STATE["version"]),
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = [
        {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    started = getattr(request.state, "started_at", None)
    latency_ms = round((time.perf_counter() - started) * 1000, 3) if started else None
    _log_event(
        logging.ERROR,
        "request_failed",
        request_id=request_id,
        path=request.url.path,
        latency_ms=latency_ms,
        model_version=str(STATE["version"]),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "internal server error",
            "request_id": request_id,
            "model_version": str(STATE["version"]),
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness. The process is up. Says nothing about whether it can serve."""
    return {"status": "alive", "model_version": str(STATE["version"])}


@app.get("/ready")
def ready():
    """Readiness. The model is loaded and can score.

    These two are genuinely different, and confusing them causes a specific production
    failure: traffic routed to a container whose model has not finished loading. All three
    providers distinguish them, and Quiz 3 asks about it.
    """
    if STATE["model"] is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "model not loaded", "model_version": str(STATE["version"])},
        )
    return {"status": "ready", "model_version": str(STATE["version"])}


def _score(rows: list[dict]) -> list[float]:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    import pandas as pd

    from src.data import FEATURES

    frame = pd.DataFrame(rows)[FEATURES]
    return [float(p) for p in STATE["model"].predict_proba(frame)[:, 1]]


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    score = _score([payload.model_dump()])[0]
    return PredictResponse(probability=score, model_version=str(STATE["version"]))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest) -> BatchResponse:
    scores = _score([row.model_dump() for row in payload.rows])
    return BatchResponse(probabilities=scores, model_version=str(STATE["version"]))
