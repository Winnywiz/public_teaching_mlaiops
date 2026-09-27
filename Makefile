# ITCS355 Labs 1–3
# Lab 3 adds registry-backed serving, load testing, canary, rollback, and teardown.

SHELL := /bin/bash
IMAGE ?= itcs355-lab1
TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
PLATFORM ?= linux/amd64
SEED ?= 20260101
INSTANCE ?= ml.m5.large
TUNE_FLAGS ?=
IMAGE_URI ?=
MODEL_REGISTRY_NAME ?= itcs355
ENDPOINT ?= itcs355-lab3
TARGET ?= http://127.0.0.1:8080
SERVE_INSTANCE ?= 0.5cpu/1Gi
LOAD_INSTANCE ?= local
P95_TARGET_MS ?= 200
LOAD_DURATION ?= 30s
PAYLOAD_BYTES ?= 0
UTILIZATION ?= 0.25
HOURLY_RATE_THB ?=
THROUGHPUT_RPS ?=
BATCH_COST_PER_1000_THB ?=

.PHONY: help setup cloud-check data test portability-audit train image image-push reproduce verify clean teardown teardown-lab2 \
        tune train-remote compare register-model reload-check serve serve-image serve-image-push aca-env-create deploy smoke \
        canary-model canary loadtest payload-test batch-test cold-start-test cost-report aca-identity-create drift inject-drift pipeline cost swap-check llm-eval llm-gate

help:
	@grep -E "^[a-zA-Z_-]+:.*?## .*$$" $(MAKEFILE_LIST) | awk -F":.*?## " "{printf \"  %-20s %s\\n\", \$$1, \$$2}"

setup: ## Install dependencies and print environment status
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "environment ok"

cloud-check: ## Resolve the eight capability slots
	python scripts/cloud_check.py

data: ## Generate the default dataset (deterministic)
	python scripts/make_dataset.py --seed $(SEED)

test: ## Run data contract and split property tests
	pytest -q tests/

portability-audit: ## Fail if provider strings leak into src/
	python scripts/portability_audit.py

train: ## Train locally, outside the container
	python -m src.train --seed $(SEED) --metrics-out reports/metrics.json

image: ## Build the training image for linux/amd64
	docker buildx build --platform $(PLATFORM) -t $(IMAGE):$(TAG) --load .

image-push: image ## Push to CONTAINER_REGISTRY via your adapter
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	print(get_adapter(config.load()).push_image(\"$(IMAGE):$(TAG)\"))"

reproduce: data image ## THE ONE COMMAND. Grader runs this.
	docker run --rm \
	  -v "$$PWD/data:/app/data:ro" \
	  -v "$$PWD/reports:/app/reports" \
	  -e MLFLOW_TRACKING_URI=sqlite:////app/reports/mlflow.db \
	  $(IMAGE):$(TAG) --seed $(SEED) --metrics-out /app/reports/metrics.json

verify: ## Check the produced metric against the README claim
	python scripts/verify_metric.py

teardown: ## Delete only resources tagged course=itcs355 and lab=3
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	cfg=config.load(strict=False); \
	print('No Azure Lab 3 resources configured; nothing deleted.' if cfg.provider != 'azure' else get_adapter(cfg).teardown(cfg.tags(3)))"

teardown-lab2: ## Remove Lab 2 tagged compute (kept separate from Lab 3 teardown)
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	cfg=config.load(); print(get_adapter(cfg).teardown(cfg.tags(2)))"

clean: ## Remove local artifacts
	rm -rf mlruns mlartifacts mlflow.db reports/metrics.json .pytest_cache

# --- Lab 2 -------------------------------------------------------------------
tune: ## Budgeted hyperparameter study (>=12 trials)
	python -m src.tune --trials 12 --budget-thb 150 --instance $(INSTANCE) --image-uri "$(IMAGE_URI)" $(TUNE_FLAGS)

train-remote: ## Submit one digest-pinned training job to managed compute
	python scripts/train_remote.py --image-uri "$(IMAGE_URI)" --instance $(INSTANCE) $(TUNE_FLAGS)

compare: ## Rank runs by metric and by cost per point
	python scripts/compare_runs.py --experiment itcs355-lab2

register-model: ## Register and promote a selected run with lineage
	python scripts/register_model.py --run-id "$(RUN_ID)" --name "$(MODEL_REGISTRY_NAME)" \
	  --training-job-id "$(TRAINING_JOB_ID)" --image-digest "$(IMAGE_DIGEST)" --seed $(SEED)

reload-check: ## Load the registered model by version and score rows
	python scripts/reload_check.py --name $(MODEL_REGISTRY_NAME) --version $(VERSION)

# --- Lab 3 -------------------------------------------------------------------
serve: ## Export a local-only model and run the inference service on :8080
	python scripts/export_model.py --out reports/model.joblib
	MODEL_PATH=reports/model.joblib MODEL_VERSION=local uvicorn service.app:app --port 8080

serve-image: ## Build the serving image locally
	docker buildx build --platform $(PLATFORM) -f service/Dockerfile.serve -t itcs355-serve:$(TAG) --load .

serve-image-push: serve-image ## Push the serving image and print its immutable digest
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	print(get_adapter(config.load()).push_image('itcs355-serve:$(TAG)'))"

aca-env-create: ## Create the tagged, log-disabled Azure Container Apps environment (billable if active)
	python scripts/manage_azure.py create-environment

aca-identity-create: ## Create a tagged user-assigned identity; roles remain explicit user/admin actions
	python scripts/manage_azure.py create-identity

deploy: ## Deploy a registered model version; set VERSION first
	python scripts/deploy_service.py --version "$(VERSION)" --endpoint "$(ENDPOINT)" --instance "$(SERVE_INSTANCE)"

smoke: ## Send three predictions and verify response/version/request-id contracts
	python scripts/smoke_service.py --endpoint "$(ENDPOINT)"

canary-model: ## Train and register a slightly weaker model; set STABLE_VERSION first
	python scripts/register_canary_model.py --stable-version "$(STABLE_VERSION)"

canary: ## 90/10 metric-only canary with automatic rollback; set CANARY_VERSION
	python scripts/canary_rollback.py --endpoint "$(ENDPOINT)" --candidate-version "$(CANARY_VERSION)" \
	  --instance "$(SERVE_INSTANCE)" --p95-target-ms $(P95_TARGET_MS)

loadtest: ## Measure p50/p95/p99, throughput, and errors at 1/10/50 users
	python scripts/run_loadtest.py --scenario concurrency --target "$(TARGET)" --p95-target-ms $(P95_TARGET_MS) \
	  --instance "$(LOAD_INSTANCE)" --duration "$(LOAD_DURATION)"

payload-test: ## Compare normal, 4 KiB, 16 KiB, and 60 KiB requests at 10 users
	python scripts/run_loadtest.py --scenario payload --target "$(TARGET)" --instance "$(LOAD_INSTANCE)" --duration "$(LOAD_DURATION)"

batch-test: ## Compare 100 single calls with one batch of 100
	python scripts/benchmark_batch.py --endpoint "$(TARGET)" --rows 100

cold-start-test: ## Measure first successful prediction after Azure Container Apps reaches zero replicas
	python scripts/measure_cold_start.py --endpoint "$(ENDPOINT)"

cost-report: ## Calculate cost/1k and batch break-even after supplying measured cost inputs
	HOURLY_RATE_THB="$(HOURLY_RATE_THB)" THROUGHPUT_RPS="$(THROUGHPUT_RPS)" \
	BATCH_COST_PER_1000_THB="$(BATCH_COST_PER_1000_THB)" REQUEST_RATE_THB_PER_MILLION="$(REQUEST_RATE_THB_PER_MILLION)" \
	WARM_HOURLY_RATE_THB="$(WARM_HOURLY_RATE_THB)" \
	python scripts/lab3_cost_report.py --utilization $(UTILIZATION) --instance "$(SERVE_INSTANCE)"

# --- Lab 4 -------------------------------------------------------------------
inject-drift: ## Shift a feature's distribution on purpose
	python scripts/inject_drift.py --feature temp_c --mode shift --magnitude 6

drift: ## Score drift against the reference window
	python -m monitoring.drift --current data/current.csv

# --- Lab 5 -------------------------------------------------------------------
pipeline: ## Compile pipeline/pipeline.yaml for your provider
	python -c "from cloudlayer.pipelines import compile_for; from src import config; \
	compile_for(config.load().provider)"

llm-eval: ## Run the LLM golden set against recorded responses (offline, free)
	python scripts/llm_eval.py --out reports/llm_eval-baseline.json

llm-gate: ## Prove the gate fails on a degraded set — expected to exit non-zero
	python scripts/llm_eval.py --out reports/llm_eval-baseline.json >/dev/null
	python scripts/llm_eval.py --responses evals/fixtures/triage-regressed.jsonl \
	  --out reports/llm_eval.json --baseline reports/llm_eval-baseline.json

cost: ## Build the cost report
	python scripts/cost_report.py --estimate $(EST) --actual $(ACT) --rps $(RPS) --instance $(INSTANCE)

swap-check: ## Prove the portability seam against a second provider
	python scripts/portability_swap_check.py --second-provider $(SECOND)
