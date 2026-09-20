"""MLflow helpers for tracking backends with different model APIs."""
from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn


def log_sklearn_model(model, artifact_path: str = "model") -> None:
    """Log a scikit-learn model while keeping Azure ML tracking compatible.

    MLflow 3's ``log_model`` first calls the newer logged-models API. The Azure ML
    tracking endpoint used by this lab does not expose that endpoint yet, but it
    does support the established run-artifact API. Saving the MLflow model locally
    and uploading its files preserves the normal ``runs:/<run>/model`` URI.
    """
    if not str(mlflow.get_tracking_uri()).startswith("azureml://"):
        mlflow.sklearn.log_model(model, name=artifact_path)
        return

    with tempfile.TemporaryDirectory(prefix="itcs355-mlflow-model-") as directory:
        mlflow.sklearn.save_model(model, path=Path(directory))
        mlflow.log_artifacts(directory, artifact_path=artifact_path)
