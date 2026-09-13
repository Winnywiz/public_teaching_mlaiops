"""AWS adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install boto3
Docs: S3 -> boto3 client("s3"); ECR -> boto3 client("ecr") for the auth token,
      then `docker push` through subprocess.

Hints for Lab 1:
  * BLOB_URI looks like s3://bucket/prefix — parse it here, never in src/.
  * ECR login expires. If a push that worked yesterday fails today, re-authenticate:
        aws ecr get-login-password --region $REGION | docker login --username AWS \
            --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
  * Return the DIGEST reference from push_image, not the tag. `docker inspect` or the
    push output gives you the sha256.
  * Tag the bucket objects and the ECR repository with cfg.tags(1).
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from urllib.parse import urlencode, urlparse

import boto3

from cloudlayer.base import CloudAdapter


class AwsAdapter(CloudAdapter):
    """AWS implementation of the Lab 1 storage and image seam."""

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

    def _s3_client(self):
        return boto3.client("s3", region_name=self.cfg.region or None)

    def _ecr_client(self):
        return boto3.client("ecr", region_name=self.cfg.region or None)

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

    # submit_training / register_model  -> Lab 2 (SageMaker training job + model package group)
    # deploy / invoke                   -> Lab 3 (SageMaker real-time endpoint)
    # emit_metric                       -> Lab 4 (CloudWatch put_metric_data)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resourcegroupstaggingapi to find by tag)
