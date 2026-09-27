"""No-cloud tests for the Lab 3 Azure Container Apps adapter."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloudlayer.azure import AzureAdapter
from scripts.canary_rollback import weighted


def adapter() -> AzureAdapter:
    cfg = SimpleNamespace(
        container_registry="example.azurecr.io/itcs355",
        serving_image_uri="example.azurecr.io/itcs355@sha256:" + "a" * 64,
        model_registry_name="itcs355-test",
        mlflow_tracking_uri="azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/test",
        region="japaneast",
        project_id="student-test",
        tags=lambda lab: {"course": "itcs355", "student": "student-test", "lab": str(lab)},
    )
    return AzureAdapter(cfg)


def test_model_reference_requires_exact_configured_registry_name():
    assert AzureAdapter._model_name_version("models:/itcs355-test/7", "itcs355-test") == (
        "itcs355-test", "7"
    )
    with pytest.raises(ValueError, match="specific registry version"):
        AzureAdapter._model_name_version("latest", "itcs355-test")
    with pytest.raises(ValueError, match="configured MODEL_REGISTRY_NAME"):
        AzureAdapter._model_name_version("other:1", "itcs355-test")


@pytest.mark.parametrize("instance", ["0.25cpu/0.5Gi", "0.5cpu/1Gi", "1cpu/2Gi", "2cpu/4Gi"])
def test_supported_aca_resource_pairs(instance):
    cpu, memory = AzureAdapter._container_resources(instance)
    assert cpu > 0
    assert memory.endswith("Gi")


def test_unsupported_aca_resource_pair_is_rejected():
    with pytest.raises(ValueError, match="supported pair"):
        AzureAdapter._container_resources("1cpu/4Gi")


def test_new_app_template_uses_digest_image_registry_model_and_probes(monkeypatch):
    instance = adapter()
    monkeypatch.setattr(instance, "_containerapp_identity", lambda: ({"type": "SystemAssigned"}, "system", []))
    monkeypatch.setattr(
        instance,
        "_env",
        lambda name, default=None: "itcs355-lab3-env" if name == "AZURE_CONTAINERAPP_ENVIRONMENT" else default,
    )
    monkeypatch.setattr(instance, "_run_az", lambda *args, **kwargs: {"id": "/subscriptions/test/envs/lab3"})
    document = instance._app_yaml("itcs355-lab3", "itcs355-test", "7", "0.5cpu/1Gi", "v7-123")
    properties = document["properties"]
    assert document["tags"]["lab"] == "3"
    assert properties["configuration"]["activeRevisionsMode"] == "multiple"
    assert properties["configuration"]["ingress"]["targetPort"] == 8080
    assert properties["configuration"]["registries"][0]["identity"] == "system"
    container = properties["template"]["containers"][0]
    assert container["image"].endswith("@sha256:" + "a" * 64)
    assert {item["name"]: item["value"] for item in container["env"]}["MODEL_VERSION"] == "7"
    assert {probe["type"] for probe in container["probes"]} == {"Liveness", "Readiness", "Startup"}
    assert properties["template"]["scale"]["minReplicas"] == 0


def test_pre_authorized_user_identity_is_used_for_acr_and_mlflow(monkeypatch):
    instance = adapter()
    identity_id = "/subscriptions/test/resourceGroups/rg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/lab3"
    monkeypatch.setattr(
        instance,
        "_env",
        lambda name, default=None: identity_id if name == "AZURE_CONTAINERAPP_IDENTITY" else default,
    )
    monkeypatch.setattr(instance, "_run_az", lambda *args, **kwargs: "client-id-123")
    identity, registry_identity, env = instance._containerapp_identity()
    assert identity["type"] == "UserAssigned"
    assert identity["userAssignedIdentities"] == {identity_id: {}}
    assert registry_identity == identity_id
    assert env == [{"name": "AZURE_CLIENT_ID", "value": "client-id-123"}]


def test_traffic_weights_require_known_revisions_and_total_100(monkeypatch):
    instance = adapter()
    monkeypatch.setattr(
        instance,
        "list_revisions",
        lambda endpoint: [{"name": "app--stable"}, {"name": "app--candidate"}],
    )
    commands = []
    monkeypatch.setattr(instance, "_run_az", lambda *args, **kwargs: commands.append(args))
    instance.set_traffic_weights("app", {"app--stable": 90, "app--candidate": 10})
    assert "app--stable=90" in commands[0]
    assert "app--candidate=10" in commands[0]
    with pytest.raises(ValueError, match="sum to 100"):
        instance.set_traffic_weights("app", {"app--stable": 90, "app--candidate": 5})
    with pytest.raises(ValueError, match="unknown"):
        instance.set_traffic_weights("app", {"app--unknown": 100})


def test_teardown_refuses_unscoped_or_non_course_filters():
    instance = adapter()
    with pytest.raises(ValueError, match="exact course"):
        instance.teardown({})
    with pytest.raises(ValueError, match="exact course"):
        instance.teardown({"lab": "3"})


def test_lab3_teardown_deletes_only_tagged_app_identity_and_unused_environment(monkeypatch):
    instance = adapter()
    tags = {"course": "itcs355", "student": "student-test", "lab": "3"}
    app = {"name": "itcs355-lab3", "tags": tags,
           "properties": {"environmentId": "/subscriptions/test/env"},
           "identity": {"userAssignedIdentities": {}}}
    identity = {"name": "itcs355-lab3-id", "id": "/subscriptions/test/identity", "tags": tags}
    environment = {"name": "itcs355-lab3-env", "id": "/subscriptions/test/env", "tags": tags}
    app_lists = [[app], []]
    commands = []
    identity_deleted = False
    environment_deleted = False

    def run_az(*args, **kwargs):
        nonlocal identity_deleted, environment_deleted
        commands.append(args)
        if args[:2] == ("containerapp", "list"):
            return app_lists.pop(0) if app_lists else []
        if args[:2] == ("identity", "list"):
            return [] if identity_deleted else [identity]
        if args[:3] == ("containerapp", "env", "list"):
            return [] if environment_deleted else [environment]
        if args[:2] == ("identity", "delete"):
            identity_deleted = True
            return None
        if args[:3] == ("containerapp", "env", "delete"):
            environment_deleted = True
            return None
        return None

    monkeypatch.setattr(instance, "_env", lambda name, default=None: "rg-test" if name == "AZURE_RESOURCE_GROUP" else default)
    monkeypatch.setattr(instance, "_run_az", run_az)
    deleted = instance._teardown_lab3_resources(tags)
    assert deleted == [
        "deleted Container App itcs355-lab3",
        "deleted managed identity itcs355-lab3-id",
        "deleted Container Apps environment itcs355-lab3-env",
        "verified no non-shared Lab 3 Container Apps resources remain",
    ]
    assert all(command[0] != "ml" for command in commands)


def test_canary_weights_target_named_candidate_not_an_unrelated_revision():
    active = [
        {"name": "app--stable"},
        {"name": "app--old"},
        {"name": "app--candidate"},
    ]
    assert weighted(active, "app--stable", 90, "app--candidate") == {
        "app--stable": 90,
        "app--old": 0,
        "app--candidate": 10,
    }


@pytest.mark.parametrize("canary", [False, True])
def test_update_routes_latest_normally_but_preserves_stable_for_canary(monkeypatch, canary):
    instance = adapter()
    environment_id = "/subscriptions/test/providers/Microsoft.App/managedEnvironments/lab3"
    old_revision = "itcs355-lab3--stable"
    latest_revision = "itcs355-lab3--v2-123"
    old_app = {
        "name": "itcs355-lab3",
        "tags": {"course": "itcs355", "student": "student-test", "lab": "3"},
        "properties": {
            "environmentId": environment_id,
            "configuration": {
                "activeRevisionsMode": "multiple",
                "revisionTransitionThreshold": 0,
                "targetLabel": "preview",
                "ingress": {
                    "external": True, "targetPort": 8080, "targetPortHttpScheme": None,
                    "traffic": [{"revisionName": old_revision, "weight": 100}],
                },
            },
            "template": {
                "containers": [{"name": "itcs355-lab3", "image": "old-image"}],
                "scale": {"minReplicas": 0, "maxReplicas": 3},
            },
        },
    }
    request_documents = []
    traffic_calls = []
    monkeypatch.setattr(
        instance,
        "_env",
        lambda name, default=None: {
            "AZURE_CONTAINERAPP_ENVIRONMENT": "lab3-env",
            "AZURE_SUBSCRIPTION_ID": "test-subscription",
        }.get(name, default),
    )
    monkeypatch.setattr(instance, "_containerapp_identity", lambda: ({"type": "SystemAssigned"}, "system", []))

    def run_az(*args, **kwargs):
        nonlocal latest_revision
        if args[:2] == ("containerapp", "list"):
            return [old_app]
        if args[:3] == ("containerapp", "env", "show"):
            return {"id": environment_id}
        if args[:2] == ("rest", "--method"):
            assert args[args.index("--method") + 1] == "put"
            assert "api-version=2025-07-01" in args[args.index("--url") + 1]
            body_path = args[args.index("--body") + 1].removeprefix("@")
            document = json.loads(Path(body_path).read_text(encoding="utf-8"))
            request_documents.append(document)
            latest_revision = f"itcs355-lab3--{document['properties']['template']['revisionSuffix']}"
            return {}
        if args[:2] == ("containerapp", "show"):
            return {"properties": {
                "latestRevisionName": latest_revision,
                "configuration": {"ingress": {"fqdn": "lab3.example.test"}},
            }}
        if args[:3] == ("containerapp", "revision", "list"):
            return [
                {"name": old_revision, "properties": {
                    "active": True, "trafficWeight": 100,
                    "runningState": "Running",
                    "template": {"containers": [{"env": [{"name": "MODEL_VERSION", "value": "1"}]}]},
                }},
                {"name": latest_revision, "properties": {
                    "active": True, "trafficWeight": 0,
                    "runningState": "Running",
                    "template": {"containers": [{"env": [{"name": "MODEL_VERSION", "value": "2"}]}]},
                }},
            ]
        if args[:4] == ("containerapp", "ingress", "traffic", "set"):
            traffic_calls.append(args)
            return None
        raise AssertionError(f"unexpected Azure CLI call: {args}")

    monkeypatch.setattr(instance, "_run_az", run_az)
    method = instance.deploy_canary_revision if canary else instance.deploy
    assert method("models:/itcs355-test/2", "itcs355-lab3", "0.5cpu/1Gi") == "https://lab3.example.test"
    assert request_documents[0]["properties"]["configuration"]["ingress"]["traffic"] == [
        {"revisionName": old_revision, "weight": 100}
    ]
    assert "targetPortHttpScheme" not in request_documents[0]["properties"]["configuration"]["ingress"]
    assert "revisionTransitionThreshold" not in request_documents[0]["properties"]["configuration"]
    assert "targetLabel" not in request_documents[0]["properties"]["configuration"]
    assert len(traffic_calls) == (0 if canary else 1)
    if not canary:
        assert f"{latest_revision}=100" in traffic_calls[0]
