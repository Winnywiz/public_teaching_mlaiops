"""Safety tests for exact-tag Lab 5 teardown. No Azure calls are made."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cloudlayer.azure import AzureAdapter


def adapter() -> AzureAdapter:
    return AzureAdapter(SimpleNamespace(
        region="japaneast",
        project_id="student-test",
        tags=lambda lab: {"course": "itcs355", "student": "student-test", "lab": str(lab)},
    ))


def test_lab5_teardown_only_deletes_exact_tagged_resources(monkeypatch):
    instance = adapter()
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "test-subscription")
    monkeypatch.setenv("AZURE_RESOURCE_GROUP", "student-rg")
    monkeypatch.setenv("AZURE_ML_WORKSPACE", "student-workspace")
    tags = instance.cfg.tags(5)
    other = instance.cfg.tags(4)
    calls = []
    lists = {
        ("ml", "schedule", "list"): [
            {"name": "lab5-weekly", "tags": tags, "is_enabled": True},
            {"name": "other-schedule", "tags": other},
        ],
        ("ml", "job", "list"): [
            {"name": "lab5-run", "status": "Completed", "tags": tags},
            {"name": "other-run", "status": "Running", "tags": other},
        ],
        ("ml", "compute", "list"): [
            {"name": "lab5-spot", "tags": tags},
            {"name": "shared", "tags": other},
        ],
        ("containerapp", "list"): [
            {"name": "lab5-staging", "tags": tags},
            {"name": "lab3-serving", "tags": instance.cfg.tags(3)},
        ],
        ("containerapp", "env", "list"): [
            {"name": "lab5-env", "tags": tags},
            {"name": "shared-env", "tags": other},
        ],
    }

    def run_az(*args, **kwargs):
        calls.append(args)
        for prefix, value in lists.items():
            if args[:len(prefix)] == prefix:
                return value
        return None

    monkeypatch.setattr(instance, "_run_az", run_az)
    changed = instance.teardown(tags)

    assert any("Lab 5 schedule lab5-weekly" in item for item in changed)
    assert any("Lab 5 Container App lab5-staging" in item for item in changed)
    destructive = [call for call in calls if call[:2] in {
        ("ml", "schedule"), ("ml", "compute"), ("containerapp", "delete"),
        ("containerapp", "env"),
    }]
    flattened = " ".join(" ".join(call) for call in destructive)
    assert "lab5-weekly" in flattened and "lab5-spot" in flattened
    assert "lab5-staging" in flattened and "lab5-env" in flattened
    assert "other-schedule" not in flattened and "shared-env" not in flattened
    assert "lab3-serving" not in flattened and "other-run" not in flattened


def test_lab5_teardown_refuses_nonexact_tags():
    instance = adapter()
    with pytest.raises(ValueError, match="exact configured"):
        instance.teardown({"course": "itcs355", "lab": "5"})


def test_lab5_teardown_verification_lists_only_active_matching_resources(monkeypatch):
    instance = adapter()
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "test-subscription")
    monkeypatch.setenv("AZURE_RESOURCE_GROUP", "student-rg")
    monkeypatch.setenv("AZURE_ML_WORKSPACE", "student-workspace")
    tags = instance.cfg.tags(5)
    lists = [
        [{"name": "weekly", "tags": tags}],
        [{"name": "finished", "status": "Completed", "tags": tags},
         {"name": "running", "status": "Running", "tags": tags}],
        [],
        [{"name": "app", "tags": tags}],
        [],
    ]
    monkeypatch.setattr(instance, "_run_az", lambda *args, **kwargs: lists.pop(0))
    remaining = instance.verify_lab5_teardown(tags)
    assert remaining == ["schedules: weekly", "jobs: running", "apps: app"]
