# ITCS355 Lab 1 — training image
#
# The manifest-list digest pins the complete multi-architecture base. The build command
# selects linux/amd64, so the same immutable reference resolves to the intended platform.
FROM python@sha256:4d1caded1f729ae443eb803f26ffde7b61e696aeaef62f099abb6dd6b14257c7 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies first so this layer caches independently of your source.
COPY requirements.txt ./
RUN pip install --prefix=/install --require-hashes -r requirements.txt


FROM python@sha256:4d1caded1f729ae443eb803f26ffde7b61e696aeaef62f099abb6dd6b14257c7 AS runtime

# Non-root. A training container has no reason to run as root, and graders check.
RUN useradd --create-home --uid 10001 runner
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=runner:runner src/ ./src/
COPY --chown=runner:runner cloudlayer/ ./cloudlayer/
COPY --chown=runner:runner scripts/ ./scripts/
COPY --chown=runner:runner data/raw.dvc ./data/raw.dvc

USER runner

# Credentials NEVER enter an image layer. They arrive at runtime from SECRET_STORE_PATH
# or from the platform's identity. If you find yourself adding an ARG for a key, stop.
# Keep a runnable default for the Labs 1–3 `docker run` workflow, while allowing
# Azure ML command jobs to replace CMD with any pipeline-step command.
CMD ["python", "-m", "src.train", "--n-estimators", "200", "--max-depth", "8"]
