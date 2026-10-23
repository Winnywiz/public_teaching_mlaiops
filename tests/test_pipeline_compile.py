"""Offline tests for the Azure ML pipeline compiler; no cloud jobs are submitted."""
from __future__ import annotations

import pytest
import yaml

from cloudlayer.pipelines import compile_azure


def example_spec() -> dict:
    return {
        "name": "lab5-test",
        "schedule": "0 2 * * 0",
        "image": "registry.example/itcs355@sha256:" + "a" * 64,
        "compute": "student-b2s",
        "spot_compute": "low-pri-cluster",
        "tags": {"course": "itcs355", "lab": "5"},
        "steps": [
            {"name": "ingest", "command": "python ingest.py", "outputs": {"raw_data": "uri_folder"}},
            {"name": "validate", "command": "python validate.py", "depends_on": ["ingest"],
             "inputs": {"raw_data": "${{parent.jobs.ingest.outputs.raw_data}}"},
             "outputs": {"validated": "uri_file"}},
            {"name": "train", "command": "python train.py", "depends_on": ["validate"], "spot": True,
             "inputs": {"validated": "${{parent.jobs.validate.outputs.validated}}"},
             "outputs": {"metrics": "uri_folder"}},
            {"name": "evaluate", "command": "python gate.py", "depends_on": ["train"],
             "inputs": {"metrics": "${{parent.jobs.train.outputs.metrics}}"},
             "outputs": {"gate": "uri_file"}},
            {"name": "register", "command": "python register.py", "depends_on": ["evaluate"],
             "inputs": {"gate": "${{parent.jobs.evaluate.outputs.gate}}"}},
            {"name": "deploy_staging", "command": "python deploy.py", "depends_on": ["register"],
             "inputs": {"registration": "${{parent.jobs.register.outputs.registration}}"}},
        ],
    }


def test_compiler_creates_fail_closed_pipeline_and_weekly_schedule(tmp_path):
    pipeline_path = compile_azure(example_spec(), tmp_path)
    pipeline = yaml.safe_load(pipeline_path.read_text())
    schedule = yaml.safe_load((tmp_path / "azureml-schedule.yml").read_text())

    assert pipeline["settings"]["continue_on_step_failure"] is False
    assert pipeline["jobs"]["train"]["compute"] == "azureml:low-pri-cluster"
    assert pipeline["jobs"]["register"]["inputs"]["gate"].endswith("evaluate.outputs.gate}}")
    assert pipeline["jobs"]["deploy_staging"]["inputs"]["registration"].endswith(
        "register.outputs.registration}}"
    )
    assert schedule["trigger"]["expression"] == "0 2 * * 0"
    assert schedule["trigger"]["timezone"] == "UTC"
    assert schedule["create_job"]["settings"]["continue_on_step_failure"] is False


def test_compiler_keeps_the_training_identity_on_the_spot_training_step(tmp_path):
    spec = example_spec()
    spec["steps"][2]["environment_variables"] = {"AZURE_CLIENT_ID": "training-client-id"}

    pipeline_path = compile_azure(spec, tmp_path)
    pipeline = yaml.safe_load(pipeline_path.read_text())

    assert pipeline["jobs"]["train"]["environment_variables"] == {
        "AZURE_CLIENT_ID": "training-client-id"
    }


def test_compiler_rejects_registration_not_gated_by_evaluation(tmp_path):
    spec = example_spec()
    spec["steps"][4]["depends_on"] = ["train"]
    with pytest.raises(ValueError, match="register must depend on the evaluation gate"):
        compile_azure(spec, tmp_path)


def test_compiler_requires_digest_pinned_image(tmp_path):
    spec = example_spec()
    spec["image"] = "registry.example/itcs355:latest"
    with pytest.raises(ValueError, match="digest-pinned"):
        compile_azure(spec, tmp_path)


def test_compiler_reports_unconfigured_lab5_account_inputs_together(tmp_path):
    spec = example_spec()
    spec["environment_variables"] = {"CLOUD_PROVIDER": "azure"}
    with pytest.raises(ValueError, match="AZURE_LAB5_CONTAINERAPP_IDENTITY"):
        compile_azure(spec, tmp_path)


def test_compiler_rejects_serverless_for_the_student_subscription_compute_limit(tmp_path):
    spec = example_spec()
    spec["compute"] = "serverless"
    with pytest.raises(ValueError, match="Standard_B2s"):
        compile_azure(spec, tmp_path)
