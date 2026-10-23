"""Compile the provider-neutral DAG into an Azure ML CLI v2 pipeline job.

Compilation is local and does not submit work or create cloud resources. Azure ML's
pipeline runner stops downstream work when a command dependency fails, so the evaluation
step's non-zero exit plus ``continue_on_step_failure: false`` is the registration gate.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

# Load the ignored cloud.env file using the same configuration boundary as the app.
from src import config  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "pipeline" / "pipeline.yaml"
GENERATED = ROOT / "pipeline" / "generated"
_ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_DIGEST_PATTERN = re.compile(r"@sha256:[0-9a-f]{64}$")


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(0)), value)
    return value


def load_spec(path: Path = SPEC) -> dict[str, Any]:
    """Read the neutral DAG and expand only ordinary ${ENV_VAR} references."""
    spec = _expand(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not isinstance(spec, dict) or not isinstance(spec.get("steps"), list):
        raise ValueError("pipeline spec must be a mapping with a steps list")
    return spec


def _validate_dag(spec: dict[str, Any]) -> list[dict[str, Any]]:
    steps = spec.get("steps", [])
    names = [step.get("name") for step in steps]
    if not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
        raise ValueError("pipeline step names must be non-empty and unique")
    known = set(names)
    for step in steps:
        dependencies = step.get("depends_on", [])
        missing = sorted(set(dependencies) - known)
        if missing:
            raise ValueError(f"{step['name']} depends on unknown steps: {missing}")
        if not isinstance(step.get("command"), str) or not step["command"].strip():
            raise ValueError(f"{step['name']} must have a non-empty command")
    for required in ("ingest", "validate", "train", "evaluate", "register", "deploy_staging"):
        if required not in known:
            raise ValueError(f"pipeline is missing required step {required!r}")
    by_name = {step["name"]: step for step in steps}
    if "evaluate" not in by_name["register"].get("depends_on", []):
        raise ValueError("register must depend on the evaluation gate")
    if "register" not in by_name["deploy_staging"].get("depends_on", []):
        raise ValueError("deploy_staging must depend on successful registration")
    return steps


def compile_azure(spec: dict[str, Any], out_dir: Path = GENERATED) -> Path:
    """Write Azure ML CLI v2 pipeline and schedule YAML; never submit them."""
    steps = _validate_dag(spec)
    image = str(spec.get("image", "")).strip()
    compute = str(spec.get("compute", "")).strip()
    spot_compute = str(spec.get("spot_compute", "")).strip()
    required_env = {
        "CLOUD_PROVIDER", "PROJECT_ID", "REGION", "BLOB_URI", "CONTAINER_REGISTRY",
        "MLFLOW_TRACKING_URI", "MODEL_REGISTRY_NAME", "IDENTITY_REF",
        "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ML_WORKSPACE",
        "AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_CONTAINER", "AZURE_ACR_NAME",
        "AZURE_ML_COMPUTE", "AZURE_ML_SPOT_COMPUTE", "SERVING_IMAGE_URI",
        "AZURE_LAB5_CONTAINERAPP_ENVIRONMENT", "AZURE_LAB5_CONTAINERAPP_NAME",
        "AZURE_LAB5_CONTAINERAPP_IDENTITY", "AZURE_LAB5_TRAINING_IDENTITY_CLIENT_ID",
        "AZURE_LAB5_JOB_IDENTITY_CLIENT_ID",
    }
    base_environment = spec.get("environment_variables", {})
    if base_environment:
        unresolved = [
            key for key, value in base_environment.items()
            if isinstance(value, str) and ("${" in value or ("<" in value and ">" in value))
        ]
        missing = [
            key for key in sorted(required_env)
            if not str(base_environment.get(key, "")).strip()
            or ("<" in str(base_environment.get(key, ""))
                and ">" in str(base_environment.get(key, "")))
        ]
        if not image or "${" in image or ("<" in image and ">" in image):
            missing.append("PIPELINE_IMAGE_URI")
        if compute.lower() == "serverless":
            missing.append("AZURE_ML_COMPUTE must be an existing Standard_B2s target, not serverless")
        missing_or_unresolved = sorted(set(unresolved + missing))
        if missing_or_unresolved:
            raise ValueError(
                "missing, placeholder, or unresolved Lab 5 configuration; set these in "
                "the ignored cloud.env and provision/authorize the referenced resources: "
                + ", ".join(missing_or_unresolved)
            )
    if "${" in image or not _DIGEST_PATTERN.search(image):
        raise ValueError(
            "PIPELINE_IMAGE_URI must be a digest-pinned training image "
            "(repository@sha256:<64 hex characters>)"
        )
    if not compute or "${" in compute or compute.lower() == "serverless":
        raise ValueError(
            "AZURE_ML_COMPUTE must name an existing Standard_B2s general-purpose target"
        )
    if not spot_compute or "${" in spot_compute:
        raise ValueError(
            "AZURE_ML_SPOT_COMPUTE must name an existing low-priority/spot compute target; "
            "the pipeline will not silently claim spot savings on regular compute"
        )

    compute = compute.removeprefix("azureml:")
    spot_compute = spot_compute.removeprefix("azureml:")
    jobs: dict[str, dict[str, Any]] = {}
    for step in steps:
        name = step["name"]
        job: dict[str, Any] = {
            "type": "command",
            "code": ".",
            "environment": {"image": image},
            "command": step["command"],
            "compute": f"azureml:{spot_compute if step.get('spot') else compute}",
            "tags": {**spec.get("tags", {}), "component": name},
        }
        if step.get("inputs"):
            job["inputs"] = step["inputs"]
        if step.get("outputs"):
            job["outputs"] = {
                key: {"type": value, "mode": "rw_mount"}
                for key, value in step["outputs"].items()
            }
        environment_variables = {
            **spec.get("environment_variables", {}),
            **step.get("environment_variables", {}),
        }
        if environment_variables:
            unresolved = [key for key, value in environment_variables.items()
                          if isinstance(value, str) and ("${" in value or ("<" in value and ">" in value))]
            if unresolved:
                raise ValueError(
                    "unresolved per-step pipeline configuration values for: "
                    + ", ".join(sorted(set(unresolved)))
                )
            job["environment_variables"] = environment_variables
        if step.get("identity"):
            identity = step["identity"]
            job["identity"] = {"type": identity} if isinstance(identity, str) else identity
        jobs[name] = job

    pipeline = {
        "$schema": "https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json",
        "type": "pipeline",
        "display_name": spec.get("name", "itcs355-lab5-pipeline"),
        "experiment_name": "itcs355-lab5",
        "settings": {
            "default_compute": f"azureml:{compute}",
            "continue_on_step_failure": False,
            "force_rerun": True,
        },
        "jobs": jobs,
        "tags": spec.get("tags", {}),
    }
    schedule = {
        "$schema": "https://azuremlschemas.azureedge.net/latest/schedule.schema.json",
        "name": "itcs355-lab5-weekly-retrain",
        "display_name": "ITCS355 Lab 5 weekly retraining",
        "description": (
            "Weekly sensor-model refresh; contract validation and the metric gate must pass "
            "before registration or staging deployment."
        ),
        "trigger": {
            "type": "cron",
            "expression": str(spec.get("schedule", "0 2 * * 0")),
            "timezone": "UTC",
        },
        "create_job": {
            "type": "pipeline",
            "job": "./azureml-pipeline.yml",
            "settings": {
                "default_compute": f"azureml:{compute}",
                "continue_on_step_failure": False,
                "force_rerun": True,
            },
            "tags": spec.get("tags", {}),
        },
        "tags": spec.get("tags", {}),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = out_dir / "azureml-pipeline.yml"
    schedule_path = out_dir / "azureml-schedule.yml"
    pipeline_path.write_text(yaml.safe_dump(pipeline, sort_keys=False), encoding="utf-8")
    schedule_path.write_text(yaml.safe_dump(schedule, sort_keys=False), encoding="utf-8")
    return pipeline_path


def compile_aws(spec: dict[str, Any]):
    raise NotImplementedError(
        "This Lab 5 checkout is configured for Azure; AWS pipeline compilation is outside "
        "the three-method portability swap."
    )


def compile_gcp(spec: dict[str, Any]):
    raise NotImplementedError(
        "This Lab 5 checkout is configured for Azure; Vertex pipeline compilation is outside "
        "the three-method portability swap."
    )


COMPILERS = {"aws": compile_aws, "azure": compile_azure, "gcp": compile_gcp}


def compile_for(provider: str, spec: dict[str, Any] | None = None):
    spec = spec or load_spec()
    try:
        return COMPILERS[provider.lower()](spec)
    except KeyError:
        raise ValueError(f"No pipeline compiler for {provider!r}") from None
