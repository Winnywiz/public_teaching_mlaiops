"""AWS adapter for Lab 1 storage/images and Lab 2 SageMaker/registry operations.

The adapter is the only module that knows AWS names. Callers pass a digest-pinned image,
an S3 input prefix, and provider-neutral job arguments; all AWS request shapes stay here.
"""
from __future__ import annotations

import base64
import re
import subprocess
import time
from datetime import timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import boto3
from botocore.exceptions import ClientError

from cloudlayer.base import CloudAdapter


class AwsAdapter(CloudAdapter):
    """AWS implementation of the storage, image, training, and registry seams."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._last_training_args: dict[str, Any] = {}
        self._last_training_result: dict[str, Any] = {}
        self._registry_metadata: dict[str, Any] = {}
        self._last_registry_arn: str | None = None

    def _s3_parts(self, uri: str | None = None) -> tuple[str, str]:
        value = (uri or self.cfg.blob_uri).strip()
        parsed = urlparse(value)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("BLOB_URI and object URIs must use the s3:// scheme")
        return parsed.netloc, parsed.path.strip("/")

    def _s3_key(self, key: str) -> tuple[str, str]:
        bucket, prefix = self._s3_parts()
        object_key = "/".join(part.strip("/") for part in (prefix, key) if part.strip("/"))
        if not object_key:
            raise ValueError("object key must not be empty")
        return bucket, object_key

    def _join_s3(self, suffix: str) -> str:
        bucket, prefix = self._s3_parts()
        parts = [part.strip("/") for part in (prefix, suffix) if part.strip("/")]
        return f"s3://{bucket}/{'/'.join(parts)}"

    def _ecr_parts(self) -> tuple[str, str]:
        registry = self.cfg.container_registry.strip().rstrip("/")
        if "/" not in registry:
            raise ValueError("CONTAINER_REGISTRY must be a full ECR registry/repository reference")
        return registry.split("/", 1)

    @staticmethod
    def _tag_from_local_image(local_tag: str) -> str:
        last_component = local_tag.rsplit("/", 1)[-1]
        if "@" in last_component:
            raise ValueError("local_tag must be tag-pinned, not digest-pinned")
        return last_component.rsplit(":", 1)[1] if ":" in last_component else "latest"

    @staticmethod
    def _job_name(value: str | None) -> str:
        raw = value or f"itcs355-lab2-{int(time.time())}"
        cleaned = re.sub(r"[^a-zA-Z0-9-]", "-", raw).strip("-").lower()
        return (cleaned or "itcs355-lab2-job")[:63]

    def _s3_client(self):
        return boto3.client("s3", region_name=self.cfg.region or None)

    def _ecr_client(self):
        return boto3.client("ecr", region_name=self.cfg.region or None)

    def _sagemaker_client(self):
        return boto3.client("sagemaker", region_name=self.cfg.region or None)

    def upload(self, local_path: str, key: str) -> str:
        source = Path(local_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        bucket, object_key = self._s3_key(key)
        with source.open("rb") as stream:
            self._s3_client().put_object(
                Bucket=bucket,
                Key=object_key,
                Body=stream,
                Tagging=urlencode(self.cfg.tags(1)),
            )
        return f"s3://{bucket}/{object_key}"

    def download(self, uri: str, local_path: str) -> None:
        bucket, object_key = self._s3_parts(uri)
        if not object_key:
            raise ValueError("object URI must include an object key")
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._s3_client().download_file(bucket, object_key, str(destination))

    def push_image(self, local_tag: str) -> str:
        registry, repository = self._ecr_parts()
        tag = self._tag_from_local_image(local_tag)
        ecr = self._ecr_client()
        auth = ecr.get_authorization_token()["authorizationData"][0]
        username, credential = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
        login_host = urlparse(auth["proxyEndpoint"]).netloc or registry
        subprocess.run(
            ["docker", "login", login_host, "--username", username, "--password-stdin"],
            input=credential + "\n",
            text=True,
            check=True,
            capture_output=True,
        )

        remote_tag = f"{self.cfg.container_registry.rstrip('/')}:{tag}"
        subprocess.run(["docker", "tag", local_tag, remote_tag], check=True)
        subprocess.run(["docker", "push", remote_tag], check=True)

        details = ecr.describe_images(
            repositoryName=repository,
            imageIds=[{"imageTag": tag}],
        )["imageDetails"]
        if not details or not details[0].get("imageDigest"):
            raise RuntimeError("ECR push completed without a retrievable image digest")
        repo_arn = ecr.describe_repositories(repositoryNames=[repository])["repositories"][0]["repositoryArn"]
        ecr.tag_resource(
            resourceArn=repo_arn,
            tags=[{"Key": key, "Value": value} for key, value in self.cfg.tags(1).items()],
        )
        return f"{self.cfg.container_registry.rstrip('/')}@{details[0]['imageDigest']}"

    def submit_training(self, image_uri: str, args: dict[str, Any]) -> str:
        """Submit one digest-pinned image as a SageMaker managed training job."""
        if "@sha256:" not in image_uri:
            raise ValueError("image_uri must be digest-pinned (repo@sha256:...)")
        if not self.cfg.identity_ref:
            raise ValueError("IDENTITY_REF must contain the SageMaker execution role ARN")

        sm = self._sagemaker_client()
        job_name = self._job_name(args.get("job_name"))
        instance = args.get("instance_type", "ml.m5.large")
        input_uri = args.get("input_s3_uri") or self._join_s3("lab2/data")
        output_uri = args.get("output_s3_uri") or self._join_s3(f"lab2/jobs/{job_name}")
        spot = bool(args.get("spot", True))
        container_args = [str(value) for value in args.get("container_args", [])]
        if not container_args:
            container_args = ["--data-path", "/opt/ml/input/data/training/sensors.csv"]

        algorithm = {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "-m", "src.train"],
            "ContainerArguments": container_args,
            "MetricDefinitions": [
                {"Name": "val_roc_auc", "Regex": r'"val_roc_auc":\s*([0-9.]+)'},
                {"Name": "test_roc_auc", "Regex": r'"test_roc_auc":\s*([0-9.]+)'},
                {"Name": "cost_thb", "Regex": r'"cost_thb":\s*([0-9.]+)'},
            ],
        }
        tags = {**self.cfg.tags(2), **{str(k): str(v) for k, v in args.get("tags", {}).items()}}
        request: dict[str, Any] = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": algorithm,
            "RoleArn": self.cfg.identity_ref.removeprefix("role:"),
            "InputDataConfig": [{
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
            }],
            "OutputDataConfig": {"S3OutputPath": output_uri},
            "ResourceConfig": {
                "InstanceType": instance,
                "InstanceCount": int(args.get("instance_count", 1)),
                "VolumeSizeInGB": int(args.get("volume_size_gb", 30)),
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": int(args.get("max_run_seconds", 3600))},
            "EnableManagedSpotTraining": spot,
            "Tags": [{"Key": key, "Value": value} for key, value in tags.items()],
            "Environment": {
                "MLFLOW_TRACKING_URI": str(args.get("mlflow_tracking_uri", self.cfg.mlflow_tracking_uri)),
                "PYTHONUNBUFFERED": "1",
            },
        }
        if spot:
            request["CheckpointConfig"] = {
                "S3Uri": args.get("checkpoint_s3_uri") or self._join_s3(f"lab2/checkpoints/{job_name}"),
                "LocalPath": "/opt/ml/checkpoints",
            }
        if args.get("hyperparameters"):
            request["HyperParameters"] = {
                str(key): str(value) for key, value in args["hyperparameters"].items()
            }
        sm.create_training_job(**request)
        self._last_training_args = {**args, "job_name": job_name, "image_uri": image_uri}
        return job_name

    def wait_training(self, job_id: str) -> dict[str, Any]:
        """Poll a SageMaker job and return its status, metrics, and model artifact."""
        sm = self._sagemaker_client()
        poll_seconds = float(self._last_training_args.get("poll_seconds", 30))
        terminal = {"Completed", "Failed", "Stopped"}
        while True:
            detail = sm.describe_training_job(TrainingJobName=job_id)
            status = detail["TrainingJobStatus"]
            if status in terminal:
                break
            time.sleep(poll_seconds)

        if status != "Completed":
            reason = detail.get("FailureReason", "no failure reason returned")
            raise RuntimeError(f"SageMaker training job {job_id} ended as {status}: {reason}")

        metrics = {
            item["MetricName"]: float(item["Value"])
            for item in detail.get("FinalMetricDataList", [])
            if "MetricName" in item and "Value" in item
        }
        started = detail.get("TrainingStartTime")
        ended = detail.get("TrainingEndTime")
        billable_seconds = detail.get("BillableTimeInSeconds")
        if billable_seconds is None and started and ended:
            billable_seconds = max(0.0, (ended - started).total_seconds())
        hourly_rate = self._last_training_args.get("hourly_rate_thb")
        if hourly_rate is not None and billable_seconds is not None:
            metrics["cost_thb"] = float(hourly_rate) * float(billable_seconds) / 3600.0

        result = {
            "job_id": job_id,
            "status": status,
            "training_job_arn": detail.get("TrainingJobArn", ""),
            "model_uri": detail.get("ModelArtifacts", {}).get("S3ModelArtifacts", ""),
            "output_uri": detail.get("OutputDataConfig", {}).get("S3OutputPath", ""),
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

    def set_registry_metadata(self, metadata: dict[str, Any]) -> None:
        self._registry_metadata = {str(key): str(value) for key, value in metadata.items()}

    def register_model(self, model_uri: str, name: str) -> str:
        """Create a SageMaker model package with all eight required lineage fields."""
        if not model_uri.startswith("s3://"):
            raise ValueError("AWS model registration requires an s3:// model artifact URI")
        metadata = {
            **self._registry_metadata,
            "training_job_id": self._last_training_result.get("job_id", "unknown"),
        }
        required = (
            "git_commit", "data_version", "mlflow_run_id", "training_job_id",
            "image_digest", "seed", "metric_val", "metric_test",
        )
        missing = [key for key in required if not metadata.get(key)]
        if missing:
            raise ValueError("Missing registry lineage fields: " + ", ".join(missing))
        image_uri = metadata.get("image_uri") or self._last_training_args.get("image_uri")
        if not image_uri:
            raise ValueError("image_uri is required to register a SageMaker model package")

        sm = self._sagemaker_client()
        try:
            sm.describe_model_package_group(ModelPackageGroupName=name)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in {"ResourceNotFoundException", "ValidationException"}:
                raise
            sm.create_model_package_group(
                ModelPackageGroupName=name,
                ModelPackageGroupDescription="ITCS355 Lab 2 lineage-tracked model",
                Tags=[{"Key": key, "Value": value} for key, value in self.cfg.tags(2).items()],
            )

        response = sm.create_model_package(
            ModelPackageGroupName=name,
            ModelPackageDescription="ITCS355 Lab 2 selected model",
            ModelApprovalStatus="PendingManualApproval",
            InferenceSpecification={
                "Containers": [{"Image": image_uri, "ModelDataUrl": model_uri}],
                "SupportedContentTypes": ["application/json", "text/csv"],
                "SupportedResponseMIMETypes": ["application/json"],
            },
            CustomerMetadataProperties=metadata,
            Tags=[
                {"Key": key, "Value": value}
                for key, value in {**self.cfg.tags(2), **metadata}.items()
            ],
        )
        arn = response["ModelPackageArn"]
        self._last_registry_arn = arn
        return str(response.get("ModelPackageVersion", arn.rsplit("/", 1)[-1]))

    def promote_model(self, model_ref: str, stage: str = "Staging") -> str:
        """Move a package to the requested approval stage after review evidence exists."""
        arn = self._last_registry_arn
        if model_ref.startswith("arn:"):
            arn = model_ref
        elif ":" in model_ref:
            group, version = model_ref.rsplit(":", 1)
            response = self._sagemaker_client().list_model_packages(
                ModelPackageGroupName=group,
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=100,
            )
            for package in response.get("ModelPackageSummaryList", []):
                if str(package.get("ModelPackageVersion")) == version:
                    arn = package["ModelPackageArn"]
                    break
        if not arn:
            raise ValueError("model_ref must be a model package ARN or group:version")
        status = "Approved" if stage.lower() in {"staging", "approved", "production"} else stage
        self._sagemaker_client().update_model_package(
            ModelPackageArn=arn,
            ModelApprovalStatus=status,
            ApprovalDescription=f"Promoted to {stage} after lineage and comparison review",
        )
        return status

    # deploy / invoke                   -> Lab 3 (SageMaker real-time endpoint)
    # emit_metric                       -> Lab 4 (CloudWatch put_metric_data)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resourcegroupstaggingapi to find by tag)
