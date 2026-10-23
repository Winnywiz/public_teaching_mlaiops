"""Azure adapter for Blob Storage, ACR, Azure ML jobs, and model registry.

The adapter deliberately uses the Azure CLI for most operations and the Azure SDK for
Key Vault secret retrieval. Both use the user's ``az login`` session locally and Azure
ML's job identity remotely, so no storage keys, registry passwords, or service-principal
secrets enter the project.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import quote, urlparse

from cloudlayer.base import CloudAdapter


class AzureAdapter(CloudAdapter):
    """Provider implementation backed by Azure CLI v2 and Azure ML v2."""

    _CONTAINER_APP_API_VERSION = "2025-07-01"

    _AZURE_IMAGE_ALIASES = {
        "ml.m5.large": "Standard_DS3_v2",
        "ml.m5.xlarge": "Standard_F4s_v2",
    }

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._last_training_args: dict[str, Any] = {}
        self._last_training_result: dict[str, Any] = {}
        self._registry_metadata: dict[str, Any] = {}
        self._last_registry_ref: str | None = None

    def get_secret(self, name: str) -> str:
        """Read a named secret from the configured vault using the active identity."""
        vault_url = self._env("AZURE_KEY_VAULT_URL")
        if not vault_url:
            raise RuntimeError("AZURE_KEY_VAULT_URL is required to resolve the configured secret")
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError as exc:
            raise RuntimeError(
                "Key Vault support requires azure-identity and azure-keyvault-secrets"
            ) from exc

        credential = DefaultAzureCredential()
        try:
            secret = SecretClient(vault_url=vault_url, credential=credential).get_secret(name)
            if not secret.value:
                raise RuntimeError(f"Key Vault secret {name!r} is empty")
            return secret.value
        finally:
            credential.close()

    # --- CLI and resource helpers -------------------------------------------
    @staticmethod
    def _env(name: str, default: str | None = None) -> str | None:
        value = os.environ.get(name, default)
        return value.strip() if isinstance(value, str) else value

    def _az_executable(self) -> str:
        configured = self._env("AZURE_CLI_PATH")
        if configured:
            return configured
        for name in ("az", "az.cmd"):
            resolved = shutil.which(name)
            if resolved:
                return resolved
        standard = Path(r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd")
        if standard.is_file():
            return str(standard)
        raise RuntimeError("Azure CLI was not found; install it and run `az login`.")

    def _run_az(self, *args: str, json_output: bool = False) -> Any:
        command = [self._az_executable(), *[str(arg) for arg in args]]
        if json_output:
            command.extend(["--output", "json"])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[1],
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"Azure CLI failed ({' '.join(command[1:])}): {detail}")
        output = completed.stdout.strip()
        if json_output:
            return json.loads(output) if output else {}
        return output

    @property
    def resource_group(self) -> str:
        return self._env("AZURE_RESOURCE_GROUP") or self.cfg.project_id

    @property
    def workspace_name(self) -> str:
        value = self._env("AZURE_ML_WORKSPACE")
        if not value:
            raise ValueError("AZURE_ML_WORKSPACE must identify the Azure ML workspace")
        return value

    @property
    def subscription_id(self) -> str:
        value = self._env("AZURE_SUBSCRIPTION_ID")
        if value:
            return value
        return str(self._run_az("account", "show", "--query", "id", "--output", "tsv"))

    @staticmethod
    def _safe_job_name(value: str | None) -> str:
        raw = value or f"itcs355-lab2-{int(time.time())}"
        cleaned = re.sub(r"[^a-zA-Z0-9-]", "-", raw).strip("-").lower()
        return (cleaned or "itcs355-lab2-job")[:63]

    @classmethod
    def _instance_type(cls, value: str | None) -> str:
        requested = value or "Standard_DS3_v2"
        return cls._AZURE_IMAGE_ALIASES.get(requested, requested)

    @staticmethod
    def _safe_app_name(value: str | None) -> str:
        raw = value or "itcs355-lab3"
        cleaned = re.sub(r"[^a-zA-Z0-9-]", "-", raw).strip("-").lower()
        cleaned = re.sub(r"-+", "-", cleaned)
        return (cleaned or "itcs355-lab3")[:31].rstrip("-")

    @staticmethod
    def _revision_suffix(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9-]", "-", value).strip("-").lower()
        cleaned = re.sub(r"-+", "-", cleaned)
        return (cleaned or f"r{int(time.time())}")[-20:].strip("-")

    @staticmethod
    def _model_name_version(model_ref: str, configured_name: str) -> tuple[str, str]:
        value = model_ref.strip()
        if value.startswith("models:/"):
            value = value.removeprefix("models:/")
        if ":" in value:
            name, version = value.rsplit(":", 1)
        elif "/" in value:
            name, version = value.rsplit("/", 1)
        else:
            name, version = configured_name, value
        if name != configured_name:
            raise ValueError(
                f"model_ref must refer to configured MODEL_REGISTRY_NAME={configured_name!r}"
            )
        if not version or not re.fullmatch(r"\d+", version):
            raise ValueError("model_ref must end with a specific registry version")
        return name, version

    @staticmethod
    def _container_resources(instance: str | None) -> tuple[float, str]:
        value = (instance or "1cpu/2Gi").strip().lower().replace(" ", "")
        match = re.fullmatch(r"(0\.25|0\.5|1|2)(?:cpu)?/(0\.5|1|2|4)(?:gi)?", value)
        if not match:
            raise ValueError("instance must be one of 0.25cpu/0.5Gi, 0.5cpu/1Gi, 1cpu/2Gi, or 2cpu/4Gi")
        cpu, memory = float(match.group(1)), f"{match.group(2)}Gi"
        valid = {(0.25, "0.5Gi"), (0.5, "1Gi"), (1.0, "2Gi"), (2.0, "4Gi")}
        if (cpu, memory) not in valid:
            raise ValueError("instance CPU and memory must use a supported pair")
        return cpu, memory

    @property
    def containerapp_environment(self) -> str:
        value = self._env("AZURE_CONTAINERAPP_ENVIRONMENT")
        if not value:
            raise ValueError(
                "AZURE_CONTAINERAPP_ENVIRONMENT must name an existing Azure Container Apps environment"
            )
        return value

    def _containerapp_environment(self, lab: int) -> str:
        if lab == 5:
            value = self._env("AZURE_LAB5_CONTAINERAPP_ENVIRONMENT")
            if not value:
                raise ValueError(
                    "AZURE_LAB5_CONTAINERAPP_ENVIRONMENT must name the Lab 5 staging environment"
                )
            return value
        return self.containerapp_environment

    def create_container_apps_environment(self) -> str:
        """Create a scale-to-zero environment with platform logging disabled."""
        name = self.containerapp_environment
        existing = self._run_az(
            "containerapp", "env", "list", "--resource-group", self.resource_group,
            json_output=True,
        )
        for item in existing if isinstance(existing, list) else []:
            if str(item.get("name", "")).lower() == name.lower():
                return name
        self._run_az(
            "containerapp", "env", "create",
            "--name", name,
            "--resource-group", self.resource_group,
            "--location", self.cfg.region,
            "--logs-destination", "none",
            "--tags", *[f"{key}={value}" for key, value in self.cfg.tags(3).items()],
            json_output=True,
        )
        return name

    def create_containerapp_identity(self) -> str:
        """Create or reuse this lab's tagged user-assigned identity (no role grants)."""
        name = self._safe_app_name(
            self._env("AZURE_CONTAINERAPP_IDENTITY_NAME") or f"{self.cfg.project_id}-lab3-id"
        )
        identities = self._run_az(
            "identity", "list", "--resource-group", self.resource_group, json_output=True,
        )
        for item in identities if isinstance(identities, list) else []:
            if str(item.get("name", "")).lower() != name.lower():
                continue
            if item.get("tags", {}) != self.cfg.tags(3):
                raise RuntimeError(
                    f"Managed identity {name} already exists without this lab's exact tags; "
                    "set AZURE_CONTAINERAPP_IDENTITY to an identity you own instead."
                )
            resource_id = str(item.get("id", ""))
            if not resource_id:
                raise RuntimeError(f"Azure did not return the resource ID for identity {name}")
            return resource_id
        item = self._run_az(
            "identity", "create",
            "--name", name,
            "--resource-group", self.resource_group,
            "--location", self.cfg.region,
            "--tags", *[f"{key}={value}" for key, value in self.cfg.tags(3).items()],
            json_output=True,
        )
        resource_id = str(item.get("id", ""))
        if not resource_id:
            raise RuntimeError(f"Azure did not return the resource ID for identity {name}")
        return resource_id

    @staticmethod
    def _aca_probes() -> list[dict[str, Any]]:
        return [
            {
                "type": "Liveness",
                "httpGet": {"path": "/health", "port": 8080},
                "initialDelaySeconds": 30,
                "periodSeconds": 10,
                "failureThreshold": 3,
            },
            {
                "type": "Readiness",
                "httpGet": {"path": "/ready", "port": 8080},
                "initialDelaySeconds": 5,
                "periodSeconds": 5,
                "failureThreshold": 3,
            },
            {
                "type": "Startup",
                "httpGet": {"path": "/health", "port": 8080},
                "periodSeconds": 5,
                "failureThreshold": 10,
            },
        ]

    def _serving_image(self) -> tuple[str, str]:
        image = (self.cfg.serving_image_uri or self._env("SERVING_IMAGE_URI") or "").strip()
        if "@sha256:" not in image:
            raise ValueError(
                "SERVING_IMAGE_URI must be the digest printed by `make serve-image-push`"
            )
        host, registry, _ = self._registry_parts()
        if not image.startswith(f"{host}/"):
            raise ValueError("SERVING_IMAGE_URI must be in the configured Azure Container Registry")
        return image, registry

    def _containerapp_identity(self, lab: int = 3) -> tuple[dict[str, Any], str, list[dict[str, str]]]:
        """Use a pre-authorized user identity when configured; otherwise system identity."""
        suffix = "LAB5_" if lab == 5 else ""
        identity_id = (self._env(f"AZURE_{suffix}CONTAINERAPP_IDENTITY") or "").strip()
        if not identity_id:
            return {"type": "SystemAssigned"}, "system", []
        if not identity_id.startswith("/subscriptions/"):
            raise ValueError(f"AZURE_{suffix}CONTAINERAPP_IDENTITY must be a full managed identity resource ID")
        client_id = str(self._run_az(
            "identity", "show", "--ids", identity_id,
            "--query", "clientId", "--output", "tsv",
        )).strip()
        if not client_id:
            raise RuntimeError("Azure did not return the configured managed identity client ID")
        return (
            {"type": "UserAssigned", "userAssignedIdentities": {identity_id: {}}},
            identity_id,
            [{"name": "AZURE_CLIENT_ID", "value": client_id}],
        )

    def _containerapp_name(self, endpoint: str, lab: int = 3) -> str:
        suffix = "LAB5_" if lab == 5 else ""
        configured = self._env(f"AZURE_{suffix}CONTAINERAPP_NAME")
        candidate = configured or endpoint
        if candidate.startswith(("https://", "http://")):
            host = urlparse(candidate).hostname or ""
            candidate = host.split(".", 1)[0]
        return self._safe_app_name(candidate)

    def _containerapp_detail(self, endpoint: str, lab: int = 3) -> dict[str, Any]:
        return self._run_az(
            "containerapp", "show",
            "--name", self._containerapp_name(endpoint, lab),
            "--resource-group", self.resource_group,
            json_output=True,
        )

    @staticmethod
    def _environment_values(container: dict[str, Any]) -> dict[str, str]:
        return {
            str(item.get("name")): str(item.get("value", ""))
            for item in container.get("env", []) or []
            if item.get("name")
        }

    def _app_yaml(self, endpoint: str, model_name: str, version: str, instance: str | None,
                  suffix: str, current: dict[str, Any] | None = None, *, lab: int = 3) -> dict[str, Any]:
        image, _registry = self._serving_image()
        identity, registry_identity, identity_env = self._containerapp_identity(lab)
        cpu, memory = self._container_resources(instance)
        app_name = self._containerapp_name(endpoint, lab)
        environment = self._run_az(
            "containerapp", "env", "show",
            "--name", self._containerapp_environment(lab),
            "--resource-group", self.resource_group,
            json_output=True,
        )
        environment_id = str(environment.get("id", ""))
        if not environment_id:
            raise RuntimeError("Azure did not return an ID for the Container Apps environment")

        env_vars = [
            {"name": "MODEL_REGISTRY_NAME", "value": model_name},
            {"name": "MODEL_VERSION", "value": version},
            {"name": "MLFLOW_TRACKING_URI", "value": self.cfg.mlflow_tracking_uri},
            {"name": "APP_ENVIRONMENT", "value": "production"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
        ] + identity_env
        if current:
            properties = current.get("properties", {})
            existing_configuration = dict(properties.get("configuration", {}) or {})
            existing_template = dict(properties.get("template", {}) or {})
            containers = list(existing_template.get("containers", []) or [])
            if not containers:
                raise RuntimeError("Existing Container App has no container template")
            container = {
                "name": containers[0].get("name", app_name),
                "image": image,
                "resources": {"cpu": cpu, "memory": memory},
                "env": env_vars,
                "probes": self._aca_probes(),
            }
            existing_ingress = dict(existing_configuration.get("ingress", {}) or {})
            ingress = {
                "external": True,
                "targetPort": 8080,
                "transport": existing_ingress.get("transport", "auto"),
                "allowInsecure": bool(existing_ingress.get("allowInsecure", False)),
            }
            traffic = existing_ingress.get("traffic")
            if isinstance(traffic, list):
                ingress["traffic"] = [
                    {key: value for key, value in rule.items()
                     if key in {"revisionName", "label", "latestRevision", "weight"}}
                    for rule in traffic if isinstance(rule, dict)
                ]
            configuration = {
                "activeRevisionsMode": "multiple",
                "ingress": ingress,
                "registries": [{"server": self._registry_parts()[0], "identity": registry_identity}],
            }
            if existing_configuration.get("secrets"):
                configuration["secrets"] = existing_configuration["secrets"]
            template = {
                "revisionSuffix": suffix,
                "containers": [container],
                "scale": {"minReplicas": 0, "maxReplicas": 3},
            }
            env_id = str(properties.get("environmentId") or properties.get("managedEnvironmentId") or environment_id)
            resource_tags = {**(current.get("tags", {}) or {}), **self.cfg.tags(lab)}
        else:
            configuration = {
                "activeRevisionsMode": "multiple",
                "ingress": {"external": True, "targetPort": 8080, "transport": "auto"},
                "registries": [{"server": self._registry_parts()[0], "identity": registry_identity}],
            }
            template = {
                "revisionSuffix": suffix,
                "containers": [{
                    "name": app_name,
                    "image": image,
                    "resources": {"cpu": cpu, "memory": memory},
                    "env": env_vars,
                    "probes": self._aca_probes(),
                }],
                "scale": {"minReplicas": 0, "maxReplicas": 3},
            }
            env_id = environment_id
            resource_tags = self.cfg.tags(lab)

        return {
            "location": self.cfg.region,
            "tags": resource_tags,
            "identity": identity,
            "properties": {
                "environmentId": env_id,
                "configuration": configuration,
                "template": template,
            },
        }

    def deploy(self, model_ref: str, endpoint: str, instance: str) -> str:
        """Deploy a registry version and route all normal traffic to its new revision."""
        return self._deploy(model_ref, endpoint, instance, route_latest=True)

    def deploy_lab5_staging(self, model_ref: str, instance: str = "0.5cpu/1Gi") -> str:
        """Update only the pre-created Lab 5 staging app through ARM and Entra ID."""
        endpoint = self._env("AZURE_LAB5_CONTAINERAPP_NAME")
        if not endpoint:
            raise ValueError("AZURE_LAB5_CONTAINERAPP_NAME must name the Lab 5 staging app")
        if not self.cfg.mlflow_tracking_uri.startswith("azureml://"):
            raise ValueError("Lab 5 staging requires the configured Azure ML model registry")
        model_name, version = self._model_name_version(model_ref, self.cfg.model_registry_name)
        app_name = self._containerapp_name(endpoint, 5)
        environment_name = self._containerapp_environment(5)
        subscription = quote(self.subscription_id, safe="")
        resource_group = quote(self.resource_group, safe="")
        app_path = (
            f"/subscriptions/{subscription}/resourceGroups/{resource_group}"
            f"/providers/Microsoft.App/containerApps/{quote(app_name, safe='')}"
        )
        environment_path = (
            f"/subscriptions/{subscription}/resourceGroups/{resource_group}"
            f"/providers/Microsoft.App/managedEnvironments/{quote(environment_name, safe='')}"
        )
        environment = self._arm_request(
            "GET", environment_path + "?api-version=" + self._CONTAINER_APP_API_VERSION
        )
        current = self._arm_request(
            "GET", app_path + "?api-version=" + self._CONTAINER_APP_API_VERSION
        )
        if (current.get("tags") or {}) != self.cfg.tags(5):
            raise RuntimeError(
                f"Refusing to update {app_name}: the existing app must have the exact Lab 5 tags"
            )
        properties = current.get("properties", {}) or {}
        if str(properties.get("environmentId", "")).lower() != str(environment.get("id", "")).lower():
            raise RuntimeError("the Lab 5 staging app is not attached to the configured Lab 5 environment")
        configuration = properties.get("configuration", {}) or {}
        ingress = configuration.get("ingress")
        if not isinstance(ingress, dict) or not ingress.get("fqdn"):
            raise RuntimeError("the pre-created Lab 5 staging app must have configured ingress")
        containers = properties.get("template", {}).get("containers", []) or []
        if not containers:
            raise RuntimeError("the pre-created Lab 5 staging app has no container template")

        identity_id = (self._env("AZURE_LAB5_CONTAINERAPP_IDENTITY") or "").strip()
        client_id = ""
        if identity_id:
            if not identity_id.startswith("/subscriptions/"):
                raise ValueError("AZURE_LAB5_CONTAINERAPP_IDENTITY must be a full resource ID")
            assigned = ((current.get("identity") or {}).get("userAssignedIdentities", {}) or {})
            if identity_id not in assigned:
                raise RuntimeError("the configured serving identity is not attached to the Lab 5 app")
            identity = self._arm_request(
                "GET", identity_id + "?api-version=2023-01-31"
            )
            client_id = str(identity.get("properties", {}).get("clientId", ""))
            if not client_id:
                raise RuntimeError("Azure did not return the serving identity client ID")

        image, _registry = self._serving_image()
        container = containers[0]
        container["image"] = image
        env_vars = {str(item.get("name")): item for item in container.get("env", []) or []
                    if item.get("name")}
        values = {
            "MODEL_REGISTRY_NAME": model_name,
            "MODEL_VERSION": version,
            "MLFLOW_TRACKING_URI": self.cfg.mlflow_tracking_uri,
            "APP_ENVIRONMENT": "staging",
            "PYTHONUNBUFFERED": "1",
        }
        if client_id:
            values["AZURE_CLIENT_ID"] = client_id
        for key, value in values.items():
            env_vars[key] = {"name": key, "value": value}
        container["env"] = list(env_vars.values())

        suffix = self._revision_suffix(f"lab5-v{version}-{int(time.time())}")
        properties.setdefault("template", {})["revisionSuffix"] = suffix
        configuration["activeRevisionsMode"] = "multiple"
        ingress["traffic"] = [{
            "revisionName": f"{app_name}--{suffix}",
            "weight": 100,
        }]
        current["properties"] = properties
        current["tags"] = self.cfg.tags(5)
        current["location"] = self.cfg.region
        self._arm_request(
            "PUT",
            app_path + "?api-version=" + self._CONTAINER_APP_API_VERSION,
            payload=current,
        )

        expected_revision = f"{app_name}--{suffix}"
        deadline = time.monotonic() + 900
        while True:
            detail = self._arm_request(
                "GET", app_path + "?api-version=" + self._CONTAINER_APP_API_VERSION
            )
            live = detail.get("properties", {}) or {}
            state = str(live.get("provisioningState", "")).lower()
            if state in {"failed", "canceled", "cancelled"}:
                raise RuntimeError(f"Lab 5 staging app provisioning failed: {state}")
            ingress = live.get("configuration", {}).get("ingress", {}) or {}
            fqdn = ingress.get("fqdn") or live.get("latestRevisionFqdn")
            revision = live.get("latestRevisionName") or live.get("latestReadyRevisionName")
            if fqdn and revision == expected_revision and state in {"", "succeeded"}:
                return f"https://{fqdn}"
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Lab 5 staging revision {expected_revision}; "
                    f"state={state or 'unknown'}, revision={revision or 'unknown'}"
                )
            time.sleep(10)

    def _arm_request(self, method: str, path_and_query: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call the Azure Resource Manager API using the active Entra identity."""
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:
            raise RuntimeError("ARM deployment requires azure-identity") from exc
        credential = DefaultAzureCredential()
        try:
            token = credential.get_token("https://management.azure.com/.default").token
            body = json.dumps(payload).encode("utf-8") if payload is not None else None
            request = Request(
                "https://management.azure.com" + path_and_query,
                data=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method=method,
            )
            try:
                with urlopen(request, timeout=60) as response:
                    result = response.read().decode("utf-8")
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Azure ARM request failed with HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                raise RuntimeError(f"Could not reach Azure Resource Manager: {exc.reason}") from exc
            if not result:
                return {}
            decoded = json.loads(result)
            if not isinstance(decoded, dict):
                raise RuntimeError("Azure ARM returned a non-object JSON response")
            return decoded
        finally:
            credential.close()

    def deploy_canary_revision(self, model_ref: str, endpoint: str, instance: str) -> str:
        """Create a canary revision while preserving the explicitly pinned stable weights."""
        return self._deploy(model_ref, endpoint, instance, route_latest=False)

    def _deploy(
        self, model_ref: str, endpoint: str, instance: str, *,
        route_latest: bool, lab: int = 3,
    ) -> str:
        """Deploy a registry version to Azure Container Apps with multi-revision routing."""
        if not self.cfg.mlflow_tracking_uri.startswith("azureml://"):
            raise ValueError(
                "Azure Container Apps must load the shared Azure ML registry; "
                "set MLFLOW_TRACKING_URI to its azureml:// URI in cloud.env"
            )
        model_name, version = self._model_name_version(model_ref, self.cfg.model_registry_name)
        app_name = self._containerapp_name(endpoint, lab)
        self._serving_image()
        current_apps = self._run_az(
            "containerapp", "list", "--resource-group", self.resource_group,
            json_output=True,
        )
        current = next((
            item for item in current_apps if isinstance(item, dict)
            and str(item.get("name", "")).lower() == app_name.lower()
        ), None) if isinstance(current_apps, list) else None
        if current and (current.get("tags") or {}) != self.cfg.tags(lab):
            raise RuntimeError(
                f"Refusing to update Container App {app_name}: it does not have the exact Lab {lab} tags"
            )
        suffix = self._revision_suffix(f"v{version}-{int(time.time())}")
        expected_revision = f"{app_name}--{suffix}"
        document = self._app_yaml(endpoint, model_name, version, instance, suffix, current, lab=lab)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="itcs355-aca-", delete=False, encoding="utf-8"
            ) as stream:
                json.dump(document, stream, indent=2)
                temporary_path = stream.name
            resource_path = (
                f"/subscriptions/{quote(self.subscription_id, safe='')}/resourceGroups/"
                f"{quote(self.resource_group, safe='')}/providers/Microsoft.App/containerApps/"
                f"{quote(app_name, safe='')}"
            )
            resource_url = (
                "https://management.azure.com" + resource_path
                + f"?api-version={self._CONTAINER_APP_API_VERSION}"
            )
            self._run_az(
                "rest", "--method", "put", "--url", resource_url,
                "--body", f"@{temporary_path}", json_output=True,
            )
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

        deadline = time.monotonic() + 900
        detail: dict[str, Any] = {}
        while True:
            detail = self._containerapp_detail(app_name, lab)
            properties = detail.get("properties", {}) or {}
            provisioning = str(properties.get("provisioningState", "")).lower()
            if provisioning in {"failed", "canceled", "cancelled"}:
                raise RuntimeError(
                    f"Container App {app_name} provisioning {provisioning}: "
                    f"{properties.get('provisioningError') or properties.get('runningStatus') or 'see Azure activity log'}"
                )
            latest_revision = str(
                properties.get("latestRevisionName") or properties.get("latestReadyRevisionName") or ""
            )
            fqdn = (
                properties.get("configuration", {}).get("ingress", {}).get("fqdn")
                or properties.get("latestRevisionFqdn")
            )
            if fqdn and latest_revision == expected_revision:
                revisions = self.list_revisions(app_name, lab=lab)
                if any(
                    item.get("active") and item.get("name") == latest_revision
                    and item.get("model_version") == version
                    for item in revisions
                ):
                    break
            if time.monotonic() >= deadline:
                states = [
                    f"{item.get('name')}={item.get('running_state')}"
                    for item in self.list_revisions(app_name)
                ]
                raise TimeoutError(
                    f"Timed out waiting for Container App {app_name} revision for model {version}; "
                    f"provisioningState={provisioning or 'unknown'}, revisions={states or 'none'}"
                )
            time.sleep(10)
        properties = detail.get("properties", {}) or {}
        if route_latest and current is not None:
            revisions = self.list_revisions(app_name, lab=lab)
            active_names = {item["name"] for item in revisions if item.get("active") and item.get("name")}
            if not latest_revision:
                latest_revision = next((
                    str(item["name"]) for item in revisions
                    if item.get("active") and item.get("model_version") == version and item.get("name")
                ), "")
            if not latest_revision or latest_revision not in active_names:
                raise RuntimeError(
                    "Azure created/updated the app but did not report an active latest revision; "
                    "inspect revisions before sending traffic."
                )
            weights = {name: 0 for name in active_names}
            weights[latest_revision] = 100
            self.set_traffic_weights(app_name, weights, lab=lab)
        fqdn = (
            properties.get("configuration", {}).get("ingress", {}).get("fqdn")
            or properties.get("latestRevisionFqdn")
        )
        if not fqdn:
            raise RuntimeError(f"Container App {app_name} was deployed without an ingress FQDN")
        return f"https://{fqdn}"

    def invoke(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one prediction and return the decoded JSON response."""
        base = endpoint.rstrip("/") if endpoint.startswith(("http://", "https://")) else None
        if base is None:
            detail = self._containerapp_detail(endpoint)
            fqdn = detail.get("properties", {}).get("configuration", {}).get("ingress", {}).get("fqdn")
            if not fqdn:
                raise RuntimeError("Container App has no ingress FQDN")
            base = f"https://{fqdn}"
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{base}/predict",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=30) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Prediction endpoint returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach prediction endpoint: {exc.reason}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Prediction endpoint returned a non-object JSON response")
        return decoded

    def generate(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call an Azure OpenAI v1 chat deployment using Entra ID and real usage fields.

        Configure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT locally or in the
        managed job environment. No API key is read or written by this adapter.
        """
        endpoint = self._env("AZURE_OPENAI_ENDPOINT")
        deployment = self._env("AZURE_OPENAI_DEPLOYMENT")
        if not endpoint or not deployment:
            raise RuntimeError(
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT to an existing "
                "Azure OpenAI deployment before using --live. This adapter will not "
                "create a model deployment."
            )
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/openai/v1"):
            url = f"{endpoint}/chat/completions?api-version=v1"
        else:
            url = f"{endpoint}/openai/v1/chat/completions?api-version=v1"

        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:
            raise RuntimeError("Azure OpenAI live evaluation requires azure-identity") from exc

        max_output_tokens = int(params.get("max_output_tokens", 200))
        temperature = float(params.get("temperature", 0.0))
        if not 1 <= max_output_tokens <= 4096:
            raise ValueError("max_output_tokens must be between 1 and 4096")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        credential = DefaultAzureCredential()
        started = time.monotonic()
        try:
            token = credential.get_token("https://ai.azure.com/.default").token
            messages: list[dict[str, str]] = []
            system_prompt = str(params.get("system_prompt", "")).strip()
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            body = json.dumps({
                "model": deployment,
                "messages": messages,
                "max_completion_tokens": max_output_tokens,
                "temperature": temperature,
            }).encode("utf-8")
            request = Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=90) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Azure OpenAI request failed with HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                raise RuntimeError(f"Could not reach Azure OpenAI endpoint: {exc.reason}") from exc
        finally:
            credential.close()

        usage = result.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if input_tokens is None or output_tokens is None:
            raise RuntimeError(
                "Azure OpenAI response omitted usage.prompt_tokens or "
                "usage.completion_tokens; refusing to estimate token counts"
            )
        choices = result.get("choices") or []
        if not choices or not isinstance(choices[0].get("message", {}).get("content"), str):
            raise RuntimeError("Azure OpenAI response did not contain a text completion")
        details = usage.get("prompt_tokens_details") or {}
        cached_input_tokens = int(details.get("cached_tokens", 0) or 0)
        return {
            "response": choices[0]["message"]["content"],
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cached_input_tokens": cached_input_tokens,
            "latency_ms": round((time.monotonic() - started) * 1000),
            "model": str(result.get("model") or deployment),
            "usage_source": "provider",
        }

    def list_revisions(self, endpoint: str, *, lab: int = 3) -> list[dict[str, Any]]:
        """Return revision names, traffic weights, readiness, and model versions."""
        revisions = self._run_az(
            "containerapp", "revision", "list",
            "--name", self._containerapp_name(endpoint, lab),
            "--resource-group", self.resource_group,
            json_output=True,
        )
        result: list[dict[str, Any]] = []
        for item in revisions if isinstance(revisions, list) else []:
            properties = item.get("properties", {}) or {}
            containers = properties.get("template", {}).get("containers", []) or []
            env_vars = self._environment_values(containers[0]) if containers else {}
            result.append({
                "name": str(item.get("name", "")),
                "active": bool(properties.get("active", False)),
                "traffic_weight": int(properties.get("trafficWeight", 0) or 0),
                "running_state": str(properties.get("runningState", "Unknown")),
                "model_version": env_vars.get("MODEL_VERSION", "unknown"),
            })
        return result

    def set_traffic_weights(self, endpoint: str, weights: dict[str, int], *, lab: int = 3) -> None:
        """Set explicit revision traffic weights that sum to 100 percent."""
        if not weights or any(not isinstance(weight, int) or weight < 0 or weight > 100 for weight in weights.values()):
            raise ValueError("traffic weights must be non-negative integer percentages")
        if sum(weights.values()) != 100:
            raise ValueError("traffic weights must sum to 100")
        revisions = {item["name"] for item in self.list_revisions(endpoint, lab=lab)}
        missing = set(weights) - revisions
        if missing:
            raise ValueError("unknown Container Apps revisions: " + ", ".join(sorted(missing)))
        self._run_az(
            "containerapp", "ingress", "traffic", "set",
            "--name", self._containerapp_name(endpoint, lab),
            "--resource-group", self.resource_group,
            "--revision-weight", *[f"{name}={weight}" for name, weight in weights.items()],
            "--output", "none",
        )

    # --- Blob Storage --------------------------------------------------------
    @staticmethod
    def _blob_parts(uri: str) -> tuple[str, str, str]:
        parsed = urlparse(uri.strip())
        if parsed.scheme == "wasbs":
            account_host = parsed.netloc.split("@", 1)[-1]
            container = parsed.netloc.split("@", 1)[0]
            account = account_host.split(".", 1)[0]
            prefix = parsed.path.strip("/")
        elif parsed.scheme in {"https", "http"}:
            host = parsed.netloc.split(":", 1)[0]
            if ".blob.core.windows.net" not in host:
                raise ValueError("Azure Blob URI must use an account.blob.core.windows.net host")
            account = host.split(".", 1)[0]
            pieces = parsed.path.strip("/").split("/", 1)
            if not pieces or not pieces[0]:
                raise ValueError("Azure Blob URI must include a container")
            container = pieces[0]
            prefix = pieces[1] if len(pieces) == 2 else ""
        else:
            raise ValueError("Azure Blob URI must use https:// or wasbs://")
        if not account or not container:
            raise ValueError("Azure Blob URI must include an account and container")
        return account, container, prefix.strip("/")

    def _blob_key(self, key: str) -> tuple[str, str, str]:
        account, container, prefix = self._blob_parts(self.cfg.blob_uri)
        blob = "/".join(part.strip("/") for part in (prefix, key) if part.strip("/"))
        if not blob:
            raise ValueError("object key must not be empty")
        return account, container, blob

    @staticmethod
    def _blob_https_uri(account: str, container: str, blob: str) -> str:
        return f"https://{account}.blob.core.windows.net/{container}/{blob}"

    @staticmethod
    def _blob_wasbs_uri(uri: str) -> str:
        account, container, prefix = AzureAdapter._blob_parts(uri)
        suffix = f"/{prefix}" if prefix else "/"
        return f"wasbs://{container}@{account}.blob.core.windows.net{suffix}"

    def upload(self, local_path: str, key: str, *, lab: int = 1) -> str:
        source = Path(local_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        account, container, blob = self._blob_key(key)
        self._run_az(
            "storage", "blob", "upload",
            "--account-name", account,
            "--container-name", container,
            "--name", blob,
            "--file", str(source),
            "--auth-mode", "login",
            "--overwrite", "true",
            "--output", "none",
        )
        return self._blob_https_uri(account, container, blob)

    def download(self, uri: str, local_path: str) -> None:
        account, container, blob = self._blob_parts(uri)
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run_az(
            "storage", "blob", "download",
            "--account-name", account,
            "--container-name", container,
            "--name", blob,
            "--file", str(destination),
            "--auth-mode", "login",
            "--overwrite", "true",
            "--output", "none",
        )

    # --- Container Registry --------------------------------------------------
    def _registry_parts(self) -> tuple[str, str, str]:
        value = self.cfg.container_registry.strip().rstrip("/")
        if "/" not in value:
            raise ValueError("CONTAINER_REGISTRY must be registry.azurecr.io/repository")
        host, repository = value.split("/", 1)
        registry = self._env("AZURE_ACR_NAME") or host.split(".", 1)[0]
        if not registry or not repository:
            raise ValueError("CONTAINER_REGISTRY must include a registry and repository")
        return host, registry, repository

    @staticmethod
    def _tag_from_local_image(local_tag: str) -> str:
        last_component = local_tag.rsplit("/", 1)[-1]
        if "@" in last_component:
            raise ValueError("local_tag must be tag-pinned, not digest-pinned")
        return last_component.rsplit(":", 1)[1] if ":" in last_component else "latest"

    def push_image(self, local_tag: str) -> str:
        host, registry, repository = self._registry_parts()
        tag = self._tag_from_local_image(local_tag)
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError("Docker CLI was not found; install/start Docker before image-push")

        self._run_az("acr", "login", "--name", registry, "--output", "none")
        remote_tag = f"{host}/{repository}:{tag}"
        subprocess.run([docker, "tag", local_tag, remote_tag], check=True)
        subprocess.run([docker, "push", remote_tag], check=True)
        try:
            digest = self._run_az(
                "acr", "manifest", "show-metadata",
                "--registry", registry,
                "--name", f"{repository}:{tag}",
                "--query", "digest",
                "--output", "tsv",
            ).strip()
        except RuntimeError:
            digest = self._run_az(
                "acr", "repository", "show",
                "--name", registry,
                "--image", f"{repository}:{tag}",
                "--query", "digest",
                "--output", "tsv",
            ).strip()
        if not digest.startswith("sha256:"):
            raise RuntimeError("ACR push completed without a retrievable image digest")
        return f"{host}/{repository}@{digest}"

    # --- Azure ML jobs -------------------------------------------------------
    @staticmethod
    def _rewrite_container_args(container_args: list[str]) -> list[str]:
        replacements = {
            "--data-path": "${{inputs.training_data}}/sensors.csv",
            "--model-out": "${{outputs.model}}/model.joblib",
            "--checkpoint-dir": "${{outputs.checkpoint}}",
        }
        rewritten: list[str] = []
        index = 0
        while index < len(container_args):
            token = str(container_args[index])
            rewritten.append(token)
            if token in replacements and index + 1 < len(container_args):
                rewritten.append(replacements[token])
                index += 2
                continue
            index += 1
        return rewritten

    @staticmethod
    def _command(tokens: list[str]) -> str:
        # Azure ML expressions must remain unquoted so the service substitutes them.
        rendered = [token if token.startswith("${{") else shlex.quote(str(token)) for token in tokens]
        return "python -m src.train " + " ".join(rendered)

    def _job_spec(self, image_uri: str, args: dict[str, Any]) -> dict[str, Any]:
        job_name = self._safe_job_name(args.get("job_name"))
        instance = self._instance_type(args.get("instance_type"))
        data_uri = str(args.get("input_s3_uri") or self.cfg.blob_uri)
        container_args = self._rewrite_container_args([
            str(value) for value in args.get("container_args", [])
        ])
        if not container_args:
            container_args = ["--data-path", "${{inputs.training_data}}/sensors.csv"]
        tags = {**self.cfg.tags(2), **{str(k): str(v) for k, v in args.get("tags", {}).items()}}
        spec: dict[str, Any] = {
            "$schema": "https://azuremlschemas.azureedge.net/latest/commandJob.schema.json",
            "type": "command",
            "display_name": job_name,
            "experiment_name": str(args.get("experiment", "itcs355-lab2")),
            "code": ".",
            "command": self._command(container_args),
            "environment": {"image": image_uri},
            "environment_variables": {
                "MLFLOW_TRACKING_URI": self.cfg.mlflow_tracking_uri,
                "PYTHONUNBUFFERED": "1",
            },
            "identity": {"type": "user_identity"},
            "inputs": {
                "training_data": {
                    "type": "uri_folder",
                    "path": self._blob_wasbs_uri(data_uri),
                    "mode": "download",
                }
            },
            "outputs": {
                "model": {"type": "uri_folder", "mode": "upload"},
                "checkpoint": {"type": "uri_folder", "mode": "upload"},
            },
            "resources": {"instance_type": instance, "instance_count": 1},
            "limits": {"timeout": int(args.get("max_run_seconds", 3600))},
            "tags": tags,
        }
        compute = self._env("AZURE_ML_COMPUTE")
        if compute:
            spec["compute"] = f"azureml:{compute}"
        elif bool(args.get("spot", False)):
            spec["queue_settings"] = {"job_tier": "spot"}
        return spec

    def submit_training(self, image_uri: str, args: dict[str, Any]) -> str:
        """Submit one digest-pinned Azure ML command job."""
        if "@sha256:" not in image_uri:
            raise ValueError("image_uri must be digest-pinned (registry.azurecr.io/repo@sha256:...)")
        spec = self._job_spec(image_uri, args)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="itcs355-azure-job-", delete=False,
                encoding="utf-8",
            ) as stream:
                json.dump(spec, stream, indent=2)
                temporary_path = stream.name
            job_id = str(self._run_az(
                "ml", "job", "create",
                "--file", temporary_path,
                "--resource-group", self.resource_group,
                "--workspace-name", self.workspace_name,
                "--subscription", self.subscription_id,
                "--query", "name",
                "--output", "tsv",
            )).strip()
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)
        if not job_id:
            raise RuntimeError("Azure ML accepted no job name")
        tracking_job_id = self._safe_job_name(args.get("job_name"))
        self._last_training_args = {
            **args,
            "job_name": job_id,
            "tracking_job_id": tracking_job_id,
            "image_uri": image_uri,
            "instance_type": self._instance_type(args.get("instance_type")),
        }
        return job_id

    def _show_job(self, job_id: str) -> dict[str, Any]:
        return self._run_az(
            "ml", "job", "show",
            "--name", job_id,
            "--resource-group", self.resource_group,
            "--workspace-name", self.workspace_name,
            "--subscription", self.subscription_id,
            json_output=True,
        )

    @staticmethod
    def _job_status(detail: dict[str, Any]) -> str:
        return str(detail.get("status") or detail.get("properties", {}).get("status") or "Unknown")

    @staticmethod
    def _output_uri(detail: dict[str, Any], name: str) -> str:
        value = detail.get("outputs", {}).get(name, {})
        if isinstance(value, dict):
            uri = str(value.get("uri") or value.get("path") or "")
            if uri:
                return uri
        elif value:
            return str(value)
        default = detail.get("outputs", {}).get("default", {})
        if isinstance(default, dict):
            default_path = str(default.get("path") or "").rstrip("/")
            if default_path:
                return f"{default_path}/{name}"
        return ""

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            text = str(value).replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def _mlflow_result(self, tracking_job_id: str) -> dict[str, Any]:
        """Recover metrics/run ID after the managed job logs to Azure ML MLflow."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
            experiment = self._last_training_args.get("experiment", "itcs355-lab2")
            experiment_obj = mlflow.get_experiment_by_name(experiment)
            if not experiment_obj:
                return {}
            runs = mlflow.search_runs(
                experiment_ids=[experiment_obj.experiment_id],
                filter_string=f"tags.training_job_id = '{tracking_job_id}'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if runs.empty:
                return {}
            row = runs.iloc[0]
            result: dict[str, Any] = {"run_id": str(row["run_id"])}
            for key in ("val_roc_auc", "test_roc_auc", "cost_thb"):
                column = f"metrics.{key}"
                if column in row and row[column] == row[column]:
                    result[key] = float(row[column])
            return result
        except Exception:
            # A failed client-side lookup should not hide the provider job result.
            return {}

    def wait_training(self, job_id: str) -> dict[str, Any]:
        """Poll an Azure ML job and return status, metrics, and output URIs."""
        poll_seconds = float(self._last_training_args.get("poll_seconds", 30))
        terminal = {"Completed", "Failed", "Canceled", "CancelCompleted"}
        while True:
            detail = self._show_job(job_id)
            status = self._job_status(detail)
            if status in terminal:
                break
            time.sleep(poll_seconds)

        if status != "Completed":
            error = detail.get("error") or detail.get("properties", {}).get("error") or "no failure reason returned"
            raise RuntimeError(f"Azure ML job {job_id} ended as {status}: {error}")

        properties = detail.get("properties", {})
        started = self._parse_time(
            detail.get("start_time")
            or properties.get("start_time")
            or properties.get("StartTimeUtc")
        )
        ended = self._parse_time(
            detail.get("end_time")
            or properties.get("end_time")
            or properties.get("EndTimeUtc")
        )
        billable_seconds = max(0.0, (ended - started).total_seconds()) if started and ended else None
        hourly_rate = self._last_training_args.get("hourly_rate_thb")
        tracking_job_id = (
            self._last_training_args.get("tracking_job_id")
            or detail.get("display_name")
            or job_id
        )
        metrics = self._mlflow_result(str(tracking_job_id))
        if hourly_rate is not None and billable_seconds is not None:
            metrics["cost_thb"] = float(hourly_rate) * billable_seconds / 3600.0

        result: dict[str, Any] = {
            "job_id": job_id,
            "status": status,
            "model_uri": self._output_uri(detail, "model"),
            "checkpoint_uri": self._output_uri(detail, "checkpoint"),
            "image_uri": self._last_training_args.get("image_uri", ""),
            "image_digest": self._last_training_args.get("image_digest", "unknown"),
            "instance_type": self._last_training_args.get("instance_type", "unknown"),
            "spot": bool(self._last_training_args.get("spot", False)),
            "billable_seconds": billable_seconds,
            **metrics,
        }
        if started:
            result["started_at"] = started.astimezone(timezone.utc).isoformat()
        if ended:
            result["ended_at"] = ended.astimezone(timezone.utc).isoformat()
        self._last_training_result = result
        return result

    # --- Azure ML model registry --------------------------------------------
    def set_registry_metadata(self, metadata: dict[str, Any]) -> None:
        self._registry_metadata = {str(key): str(value) for key, value in metadata.items()}

    def _next_model_version(self, name: str) -> str:
        models = self._run_az(
            "ml", "model", "list",
            "--name", name,
            "--resource-group", self.resource_group,
            "--workspace-name", self.workspace_name,
            "--subscription", self.subscription_id,
            json_output=True,
        )
        versions: list[int] = []
        for model in models if isinstance(models, list) else []:
            try:
                versions.append(int(model.get("version")))
            except (TypeError, ValueError):
                continue
        return str(max(versions, default=0) + 1)

    def register_model(self, model_uri: str, name: str) -> str:
        """Register the managed-job output as a versioned Azure ML custom model."""
        if not model_uri:
            raise ValueError("Azure model registration requires a managed output URI")
        required = (
            "git_commit", "data_version", "mlflow_run_id", "training_job_id",
            "image_digest", "seed", "metric_val", "metric_test",
        )
        metadata = {
            **self._registry_metadata,
        }
        metadata.setdefault(
            "training_job_id", self._last_training_result.get("job_id", "unknown")
        )
        missing = [key for key in required if not metadata.get(key)]
        if missing:
            raise ValueError("Missing registry lineage fields: " + ", ".join(missing))
        version = self._next_model_version(name)
        payload = {
            "$schema": "https://azuremlschemas.azureedge.net/latest/model.schema.json",
            "name": name,
            "version": version,
            "path": model_uri,
            "type": "custom_model",
            "description": "ITCS355 Lab 2 lineage-tracked selected model",
            "tags": {**self.cfg.tags(2), **metadata, "stage": "PendingManualApproval"},
            "properties": metadata,
        }
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix="itcs355-azure-model-", delete=False,
                encoding="utf-8",
            ) as stream:
                json.dump(payload, stream, indent=2)
                temporary_path = stream.name
            registered = self._run_az(
                "ml", "model", "create",
                "--file", temporary_path,
                "--resource-group", self.resource_group,
                "--workspace-name", self.workspace_name,
                "--subscription", self.subscription_id,
                "--query", "version",
                "--output", "tsv",
            ).strip()
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)
        version = registered or version
        self._last_registry_ref = f"{name}:{version}"
        return version

    def promote_model(self, model_ref: str, stage: str = "Staging") -> str:
        """Record the review stage on the Azure ML model version.

        Azure ML model versions do not expose SageMaker-style approval stages, so the
        stage is represented as review tags on the version.
        """
        if ":" in model_ref:
            name, version = model_ref.rsplit(":", 1)
        elif self._last_registry_ref:
            name, version = self._last_registry_ref.rsplit(":", 1)
        else:
            raise ValueError("model_ref must be name:version")
        self._run_az(
            "ml", "model", "update",
            "--name", name,
            "--version", version,
            "--resource-group", self.resource_group,
            "--workspace-name", self.workspace_name,
            "--subscription", self.subscription_id,
            "--set",
            f"tags.stage={stage}",
            "tags.approval_status=Approved",
            "tags.approval_note=Promoted after lineage comparison and reload review",
            "--force-string",
        )
        return stage

    def _teardown_lab3_resources(self, tags: dict[str, str]) -> list[str]:
        """Delete only tagged Lab 3 apps, identities, and environments."""
        def matches(resource: dict[str, Any]) -> bool:
            resource_tags = resource.get("tags", {}) or {}
            return all(str(resource_tags.get(key)) == str(value) for key, value in tags.items())

        changed: list[str] = []
        kept_identity_names: set[str] = set()
        kept_environment_names: set[str] = set()
        apps = self._run_az(
            "containerapp", "list", "--resource-group", self.resource_group, json_output=True,
        )
        for app in apps if isinstance(apps, list) else []:
            name = str(app.get("name") or "")
            if name and matches(app):
                self._run_az(
                    "containerapp", "delete", "--name", name,
                    "--resource-group", self.resource_group, "--yes", "--output", "none",
                )
                changed.append(f"deleted Container App {name}")

        remaining_apps = self._run_az(
            "containerapp", "list", "--resource-group", self.resource_group, json_output=True,
        )
        identities = self._run_az(
            "identity", "list", "--resource-group", self.resource_group, json_output=True,
        )
        for identity in identities if isinstance(identities, list) else []:
            name = str(identity.get("name") or "")
            resource_id = str(identity.get("id") or "").lower()
            if not name or not matches(identity):
                continue
            referenced = any(
                resource_id in {
                    str(key).lower()
                    for key in (app.get("identity", {}).get("userAssignedIdentities", {}) or {})
                }
                for app in remaining_apps if isinstance(remaining_apps, list)
            )
            if referenced:
                changed.append(f"kept managed identity {name}; another Container App still uses it")
                kept_identity_names.add(name)
                continue
            self._run_az(
                "identity", "delete", "--name", name,
                "--resource-group", self.resource_group, "--output", "none",
            )
            changed.append(f"deleted managed identity {name}")

        environments = self._run_az(
            "containerapp", "env", "list", "--resource-group", self.resource_group,
            json_output=True,
        )
        for environment in environments if isinstance(environments, list) else []:
            name = str(environment.get("name") or "")
            environment_id = str(environment.get("id") or "").lower()
            if not name or not matches(environment):
                continue
            apps_still_using_environment = [
                str(app.get("name") or "")
                for app in remaining_apps if isinstance(remaining_apps, list)
                if str(
                    app.get("properties", {}).get("environmentId")
                    or app.get("properties", {}).get("managedEnvironmentId") or ""
                ).lower() == environment_id
            ]
            if apps_still_using_environment:
                changed.append(
                    f"kept Container Apps environment {name}; still contains: "
                    + ", ".join(apps_still_using_environment)
                )
                kept_environment_names.add(name)
                continue
            self._run_az(
                "containerapp", "env", "delete", "--name", name,
                "--resource-group", self.resource_group, "--yes", "--output", "none",
            )
            changed.append(f"deleted Container Apps environment {name}")

        remaining_apps = self._run_az(
            "containerapp", "list", "--resource-group", self.resource_group, json_output=True,
        )
        remaining_app_names = [
            str(app.get("name") or "") for app in remaining_apps if isinstance(remaining_apps, list)
            if matches(app)
        ]
        remaining_identities = self._run_az(
            "identity", "list", "--resource-group", self.resource_group, json_output=True,
        )
        remaining_identity_names = [
            str(identity.get("name") or "")
            for identity in remaining_identities if isinstance(remaining_identities, list)
            if matches(identity) and str(identity.get("name") or "") not in kept_identity_names
        ]
        remaining_environments = self._run_az(
            "containerapp", "env", "list", "--resource-group", self.resource_group,
            json_output=True,
        )
        remaining_environment_names = [
            str(environment.get("name") or "")
            for environment in remaining_environments if isinstance(remaining_environments, list)
            if matches(environment) and str(environment.get("name") or "") not in kept_environment_names
        ]
        if remaining_app_names or remaining_identity_names or remaining_environment_names:
            raise RuntimeError(
                "Lab 3 teardown was not confirmed; remaining tagged resources: "
                f"apps={remaining_app_names}, identities={remaining_identity_names}, "
                f"environments={remaining_environment_names}. Re-run teardown and verify in Azure."
            )
        changed.append("verified no non-shared Lab 3 Container Apps resources remain")
        return changed

    @staticmethod
    def _tag_match(resource: dict[str, Any], tags: dict[str, str]) -> bool:
        found = resource.get("tags", {}) or {}
        return all(str(found.get(key)) == str(value) for key, value in tags.items())

    def _lab5_resource_lists(self, tags: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
        common = [
            "--resource-group", self.resource_group,
            "--workspace-name", self.workspace_name,
            "--subscription", self.subscription_id,
        ]
        schedules = self._run_az("ml", "schedule", "list", *common, json_output=True)
        jobs = self._run_az("ml", "job", "list", "--all-results", *common, json_output=True)
        computes = self._run_az(
            "ml", "compute", "list", "--resource-group", self.resource_group,
            "--workspace-name", self.workspace_name, "--subscription", self.subscription_id,
            json_output=True,
        )
        apps = self._run_az(
            "containerapp", "list", "--resource-group", self.resource_group, json_output=True
        )
        environments = self._run_az(
            "containerapp", "env", "list", "--resource-group", self.resource_group,
            json_output=True,
        )
        return {
            "schedules": [r for r in schedules if isinstance(r, dict) and self._tag_match(r, tags)]
            if isinstance(schedules, list) else [],
            "jobs": [r for r in jobs if isinstance(r, dict) and self._tag_match(r, tags)]
            if isinstance(jobs, list) else [],
            "computes": [r for r in computes if isinstance(r, dict) and self._tag_match(r, tags)]
            if isinstance(computes, list) else [],
            "apps": [r for r in apps if isinstance(r, dict) and self._tag_match(r, tags)]
            if isinstance(apps, list) else [],
            "environments": [r for r in environments if isinstance(r, dict) and self._tag_match(r, tags)]
            if isinstance(environments, list) else [],
        }

    def _teardown_lab5_resources(self, tags: dict[str, str]) -> list[str]:
        if tags != self.cfg.tags(5):
            raise ValueError("Lab 5 teardown requires the exact configured course/student/lab tags")
        resources = self._lab5_resource_lists(tags)
        common = [
            "--resource-group", self.resource_group,
            "--workspace-name", self.workspace_name,
            "--subscription", self.subscription_id,
        ]
        changed: list[str] = []
        for schedule in resources["schedules"]:
            name = str(schedule.get("name") or "")
            if not name:
                continue
            if schedule.get("is_enabled", schedule.get("isEnabled", True)):
                self._run_az("ml", "schedule", "disable", "--name", name, *common, "--output", "none")
            self._run_az("ml", "schedule", "delete", "--name", name, *common, "--output", "none")
            changed.append(f"disabled and deleted Lab 5 schedule {name}")

        for job in resources["jobs"]:
            name = str(job.get("name") or "")
            status = str(job.get("status") or "").lower()
            if not name:
                continue
            if status not in {"completed", "failed", "canceled", "cancelled", "archived"}:
                self._run_az("ml", "job", "cancel", "--name", name, *common, "--output", "none")
                changed.append(f"cancelled Lab 5 job {name}")
            elif status != "archived":
                self._run_az("ml", "job", "archive", "--name", name, *common, "--output", "none")
                changed.append(f"archived completed Lab 5 job {name}")

        for app in resources["apps"]:
            name = str(app.get("name") or "")
            if name:
                self._run_az(
                    "containerapp", "delete", "--name", name,
                    "--resource-group", self.resource_group, "--yes", "--output", "none",
                )
                changed.append(f"deleted Lab 5 Container App {name}")

        for environment in resources["environments"]:
            name = str(environment.get("name") or "")
            if name:
                self._run_az(
                    "containerapp", "env", "delete", "--name", name,
                    "--resource-group", self.resource_group, "--yes", "--output", "none",
                )
                changed.append(f"deleted Lab 5 Container Apps environment {name}")

        for compute in resources["computes"]:
            name = str(compute.get("name") or "")
            if name:
                self._run_az(
                    "ml", "compute", "delete", "--name", name, "--yes", *common,
                    "--output", "none",
                )
                changed.append(f"deleted Lab 5 compute {name}")
        return changed

    def verify_lab5_teardown(self, tags: dict[str, str]) -> list[str]:
        if tags != self.cfg.tags(5):
            raise ValueError("Lab 5 verification requires the exact configured course/student/lab tags")
        remaining = self._lab5_resource_lists(tags)
        active_jobs = [
            str(job.get("name")) for job in remaining["jobs"]
            if str(job.get("status", "")).lower() not in {
                "completed", "failed", "canceled", "cancelled", "archived"
            }
        ]
        resources = {
            key: [str(item.get("name") or item.get("id") or "unnamed") for item in value]
            for key, value in remaining.items()
        }
        resources["jobs"] = active_jobs
        return [f"{key}: {', '.join(names)}" for key, names in resources.items() if names]

    def teardown(self, tags: dict[str, str]) -> list[str]:
        """Archive tagged Lab 2 jobs and delete tagged Lab 2/3 resources only."""
        if not tags or tags.get("course") != "itcs355":
            raise ValueError("teardown requires the exact course=itcs355 resource tags")
        if tags.get("lab") == "3":
            return self._teardown_lab3_resources(tags)
        if tags.get("lab") == "5":
            return self._teardown_lab5_resources(tags)

        def matches(resource: dict[str, Any]) -> bool:
            resource_tags = resource.get("tags", {}) or {}
            return all(str(resource_tags.get(key)) == str(value) for key, value in tags.items())

        changed: list[str] = []
        jobs = self._run_az(
            "ml", "job", "list",
            "--all-results",
            "--resource-group", self.resource_group,
            "--workspace-name", self.workspace_name,
            "--subscription", self.subscription_id,
            json_output=True,
        )
        for job in jobs if isinstance(jobs, list) else []:
            name = str(job.get("name") or "")
            if not name or not matches(job) or str(job.get("status")) not in {
                "Completed", "Failed", "Canceled", "CancelCompleted"
            }:
                continue
            self._run_az(
                "ml", "job", "archive",
                "--name", name,
                "--resource-group", self.resource_group,
                "--workspace-name", self.workspace_name,
                "--subscription", self.subscription_id,
                "--output", "none",
            )
            changed.append(f"archived job {name}")

        computes = self._run_az(
            "ml", "compute", "list",
            "--resource-group", self.resource_group,
            "--workspace-name", self.workspace_name,
            "--subscription", self.subscription_id,
            json_output=True,
        )
        for compute in computes if isinstance(computes, list) else []:
            name = str(compute.get("name") or "")
            if not name or not matches(compute):
                continue
            self._run_az(
                "ml", "compute", "delete",
                "--name", name,
                "--yes",
                "--resource-group", self.resource_group,
                "--workspace-name", self.workspace_name,
                "--subscription", self.subscription_id,
                "--output", "none",
            )
            changed.append(f"deleted compute {name}")

        return changed
