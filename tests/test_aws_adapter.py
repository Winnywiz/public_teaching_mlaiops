"""No-cloud contract tests for the AWS half of the Lab 5 portability seam."""
from __future__ import annotations

from io import BytesIO
import json
from types import SimpleNamespace

import pytest

from cloudlayer.aws import AwsAdapter


def adapter() -> AwsAdapter:
    return AwsAdapter(SimpleNamespace(region="ap-northeast-1"))


def test_sagemaker_invoke_sends_json_and_decodes_response(monkeypatch):
    instance = adapter()
    captured = {}

    class Runtime:
        def invoke_endpoint(self, **kwargs):
            captured.update(kwargs)
            return {"Body": BytesIO(b'{"predictions": [0.12]}')}

    monkeypatch.setattr(instance, "_sagemaker_runtime_client", lambda: Runtime())
    result = instance.invoke("triage-staging", {"rows": [{"temp_c": 70.0}]})

    assert captured["EndpointName"] == "triage-staging"
    assert captured["ContentType"] == "application/json"
    assert json.loads(captured["Body"]) == {"rows": [{"temp_c": 70.0}]}
    assert result == {"predictions": [0.12]}


def test_sagemaker_invoke_rejects_invalid_response(monkeypatch):
    instance = adapter()

    class Runtime:
        def invoke_endpoint(self, **kwargs):
            return {"Body": BytesIO(b"not-json")}

    monkeypatch.setattr(instance, "_sagemaker_runtime_client", lambda: Runtime())
    with pytest.raises(RuntimeError, match="invalid JSON"):
        instance.invoke("triage-staging", {"rows": []})


def test_sagemaker_invoke_rejects_blank_endpoint():
    with pytest.raises(ValueError, match="must not be empty"):
        adapter().invoke(" ", {"rows": []})


def test_portability_probe_cleanup_deletes_only_the_uploaded_object(monkeypatch):
    instance = adapter()
    captured = {}

    class S3:
        def delete_object(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(instance, "_s3_client", lambda: S3())
    instance.delete_uploaded_object("s3://lab5-bucket/lab5/probe.txt")
    assert captured == {"Bucket": "lab5-bucket", "Key": "lab5/probe.txt"}
