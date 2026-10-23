# ITCS355 Lab 5 — Cloud MLOps, Cost, and LLM Operations

## Lab 5 — Cloud MLOps, cost, and LLM operations

Lab 5 is maintained in this separate checkout on branch `codex/lab5-cloud-mlops-cost`. The local implementation is prepared, but account-side evidence is deliberately marked pending rather than simulated.

### Implemented and checked locally

- `cloudlayer/pipelines.py` compiles the neutral DAG into Azure ML CLI v2 pipeline and weekly schedule YAML. The graph connects ingest, contract validation, checkpointed training on a separately named low-priority compute target, a real ROC-AUC threshold gate, registration, and Lab 5-only staging deployment. The gate writes its success receipt only after passing. Azure ML is configured to stop downstream work on any failed step, so registration cannot run after a failed evaluation. The compiler writes files only; it does not submit jobs or create resources. Azure ML's pipeline YAML supports inline command jobs and `continue_on_step_failure`; see the [pipeline job schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-pipeline) and [CLI pipeline guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipelines-cli?view=azureml-api-2).
- The evaluation floor is test ROC-AUC `0.82`, based on the Lab 2 selected model's measured `0.8533` test ROC-AUC. The threshold catches a material drop while leaving room for seed variation. If an incumbent score is supplied, the gate also requires a `0.002` improvement.
- The Azure adapter can call an existing Azure OpenAI v1 deployment using Entra ID and records provider-returned token counts, cached tokens, and request latency. It fails if actual usage fields are absent. It never estimates tokens from prompt length and does not create an endpoint or deployment.
- The golden set has 10 cases. Offline baseline/degraded fixtures exercise the per-case regression gate and a safety-refusal failure. AWS now implements the invocation method needed alongside its existing object upload/download methods, with mocked tests that make no AWS calls.
- `src/llmcost.py` has no invented provider price table. Costs require the exact model's current regional input/output rates. See [`reports/lab5-llm-eval.md`](reports/lab5-llm-eval.md).

### Cloud evidence and required account actions

No Lab 5 job, retraining schedule, staging app, AWS endpoint, or live LLM call has been created from this checkout. I also did not change Azure role assignments, request quota, or create billable resources. The inherited ignored `cloud.env` is retained locally, but it has no Lab 5 spot compute or LLM deployment configured. In the earlier quota check, the Azure for Students subscription showed zero low-priority vCPU quota in Japan East, so the checkpointed spot stage cannot run until the subscription has eligible quota and a suitable compute target.

When Lab 5 resources do exist, use `make teardown-lab5` to remove only resources tagged for this lab. The inherited `make teardown` target is still the Lab 4 drift-schedule cleanup and must not be used as Lab 5 cleanup. The teardown verifier does not remove shared storage, ACR, the workspace, or managed identities; inspect those separately and retain the screenshot of the empty Lab 5 resource list.

Before a cloud run, add the Lab 5 values from `cloud.env.example` to the ignored local `cloud.env`: a digest-pinned training image, an existing `Standard_B2s` general compute target, an existing low-priority training target, two assigned job identities, a separately tagged Lab 5 Container Apps environment/app, and its serving identity. Attach the training identity to the spot target; attach the pipeline/deployment identity to the general target and grant only its required job and staging-app permissions. `PIPELINE_IMAGE_URI` must point to the training image, not the Lab 3 serving image. For live LLM evaluation, set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT` for a deployment you already have, then ensure the signed-in identity is authorized for inference. This repo will not deploy a model to Azure OpenAI.

After those prerequisites are available, `make pipeline` generates `pipeline/generated/azureml-pipeline.yml` and `azureml-schedule.yml`. Review both before submitting with Azure ML CLI. Creating the job and schedule may incur charges. The selected trigger is weekly Sunday at 02:00 UTC, because a seven-day failure-label horizon makes weekly label maturity safer than daily retraining.

| Strategy | When it fits this sensor-failure problem | How it can fail |
|---|---|---|
| Fixed schedule (chosen: weekly) | Retrain after the seven-day label horizon so recent examples have time to receive verified outcomes. | It can retrain unnecessarily during stable periods, or wait until the next run after a sudden real change. |
| Data-volume threshold (5,000 newly validated readings) | Request an earlier run when enough new readings accumulate, but only once their labels have matured. | Duplicate bursts or a fast data spike can trigger on low-quality or not-yet-labelled readings. |
| Drift-triggered (PSI above 0.20 is a review alert only) | Ask an operator to inspect a sustained feature shift before choosing to retrain. | The worst case is a calibration, unit-conversion, or upstream labeling fault that looks like drift; automatic retraining on that corrupted data can replace a working model with a worse one faster than waiting for the schedule. |

The existing contract gate must pass before training, and the evaluation gate must pass before registration or deployment.

### Identity design (not yet assigned or tested)

| Identity | Minimum intended access | Why |
|---|---|---|
| Training job identity | Read the input data container; write only the run's model/metrics/checkpoint outputs; pull the pinned image | Attached only to the low-priority training compute; no staging deployment or registry-approval rights |
| Serving identity | Pull the serving image; read the selected model artifact; write only runtime logs/metrics | Attached only to the Lab 5 staging app; no training-data or model-registration rights |
| Pipeline/deployment job identity | Read/write the pipeline's required data and registry artifacts; update only the Lab 5 staging app | Selected on the general compute for the deploy step; no subscription-wide role |
| CI submitter identity | Push the pinned image to the target ACR; submit the pipeline and schedule | Used by the authorized CLI or CI runner to start the workflow, not passed into the serving container |

No role assignment or permission-removal test has been performed. The assessed test still needs an account owner: remove the training identity's data-read grant, run the validation/ingest stage, record the exact authorization failure, then restore the grant. Keep the staging app, compute, registry, and shared workspace scopes separate.

### Portability write-up

The AWS adapter's upload/download round trip is the smallest part of the swap. `invoke` is the hardest because SageMaker Runtime calls a named endpoint with an AWS-specific request envelope, while the Azure app uses an HTTP `/predict` route; response schemas and model-container contracts can also differ. That is a genuine abstraction leak: this adapter normalizes JSON transport, not model packaging, IAM, endpoint lifecycle, or prediction semantics. A full migration from the current Azure setup would still require several engineering days for identity policies, image/build workflow, registry lineage, pipeline and deployment integration, load tests, and teardown evidence. The abstraction is worth keeping for stable storage and invocation boundaries, but it does not make cloud migration cheap. The AWS methods have offline mocked unit tests only; the required real S3 round trip and SageMaker invocation are not demonstrated because no AWS bucket, credentials, or endpoint is configured.

### Cost and LLM evidence

[`reports/lab5-cost.md`](reports/lab5-cost.md) records the current evidence gap. There is no Lab 5 estimate committed before a run, tagged billing export, measured Lab 5 throughput, applied optimization, teardown verification, or empty-resource screenshot. Do not substitute Lab 2/3 estimates or call the absence of a run “zero cost.” For the token bill, run `make llm-live` only after configuring an existing authorized deployment; then pass its exact model rates and dated regional pricing source to the evaluator. Offline fixture token counts are synthetic and are never priced.

Local checks:

```powershell
python -m pytest -q
python scripts/portability_audit.py
python scripts/llm_eval.py --out reports/llm_eval-baseline.json
```

`python scripts/llm_eval.py --responses evals/fixtures/triage-regressed.jsonl --out reports/llm_eval-regressed.json --baseline reports/llm_eval-baseline.json` should fail by design and list the cases that regressed.

---

# ITCS355 Lab 4 — CI/CD, Monitoring, and Drift

This is a separate Lab 4 checkout on branch `codex/lab4-cicd-monitoring-drift`. Lab 4 adds
full-history secret scanning, hash-locked CI, a SHA-tagged serving image, a green-main-only Azure
staging workflow, Prometheus service metrics, a provisioned Grafana dashboard, explicit SLOs, and a
PSI/KS drift detector. It leaves the previous Lab 1–3 material in this fork intact.

## Lab 4 status and remaining evidence

The local implementation and generated-data tests run without Azure credentials. No GitHub Actions
run, deliberately failing pull request, Azure ML schedule, hosted Pushgateway, Slack alert, or staging
deployment has been created from this checkout. Those require your repository, Azure workspace, and
notification-channel setup. The local drift simulation is evidence of the detector only, not proof
that a real channel received an alert. See [`reports/lab4-drift.md`](reports/lab4-drift.md) for the
measured local result and outstanding evidence.

### Why the data contracts matter

The contract tests in `tests/test_data.py` map to concrete production incidents:

- `test_schema_columns_present_and_typed`: a sensor firmware/export change renames `pressure_kpa`
  or serializes numeric readings as strings, otherwise silently feeding the wrong values downstream.
- `test_no_nulls_in_required_columns`: a partial telemetry outage or broken join emits null feature
  values; reject the batch instead of imputing a plausible-looking but misleading prediction.
- `test_features_within_plausible_ranges`: a Fahrenheit/Celsius, psi/kPa, or 0–1/0–100 unit change
  crosses domain bounds before corrupt readings reach the model.
- `test_target_is_binary_and_not_degenerate`: an upstream label mapping change turns all labels into
  one class, making a training run appear successful while learning no useful distinction.
- `test_identifier_is_unique`: replayed or duplicated readings inflate counts and can distort the
  training set and operational reporting.

The separate group-split property test catches machine leakage between train, validation, and test.
The model behavior test checks probability bounds and p95 latency on a fixed sample. CI also builds
and starts the serving image, checks the `/predict` response and SHA model version, and verifies the
`/metrics` endpoint.

### Run locally (PowerShell)

```powershell
python -m pip install --require-hashes -r requirements.txt
python scripts/make_dataset.py
python scripts/export_model.py --out reports/model.joblib
$env:MODEL_PATH = (Resolve-Path reports/model.joblib).Path
$env:MODEL_VERSION = 'local'
docker compose -f docker-compose.monitoring.yml up --build -d
```

Grafana is at `http://127.0.0.1:3000` and Prometheus at `http://127.0.0.1:9090`; both are bound to
localhost. Grafana's initial login is `admin` / `admin`, so set `$env:GRAFANA_ADMIN_PASSWORD` before
starting the stack if you want a different local password. Inject and score a local drift sample:

```powershell
python scripts/inject_drift.py --feature temp_c --mode shift --magnitude 6
python -m monitoring.drift --current data/current.csv --pushgateway-url http://localhost:9091
```

This local detector call has no alert channel unless `DRIFT_ALERT_WEBHOOK_URL` is set in the current
process environment. Do not paste the webhook into a tracked file. To stop the local dashboard stack,
run `docker compose -f docker-compose.monitoring.yml down`.

The alert threshold is PSI `0.20`. Calibration on the checked-in deterministic data produced PSI
`0.38333` for a +6°C shift in the 6,000-row `temp_c` reference. Across 100 same-distribution
6,000-row comparisons, the largest observed PSI was `0.00726`. Thus this threshold is below the
material injected shift and about 27 times the largest observed sampling-noise score in that check.
It is a starting threshold for this synthetic dataset, not a universal production constant. PSI is
the primary page/alert gate; KS is reported alongside it to distinguish location shifts from shape
changes.

### Azure schedule and alert setup

The schedule definition is under `monitoring/azureml/`. It runs daily at 02:00 UTC and compares the
latest registered reference and current-window CSV assets. Before creating it, register the
`itcs355-lab4-drift` environment, set the real compute name and data asset names in
`drift-pipeline.yml`, provide an authenticated TLS Pushgateway URL, and configure the Key Vault URL.
The Azure ML job's managed identity needs permission to read the two assets and the two Key Vault
secrets named for the Slack webhook and Pushgateway token. The current production feature window must
also be exported and registered as `itcs355-current-inputs` before each run; this checkout does not
create that production export.

From PowerShell, after `az login` and after filling in the workspace and asset prerequisites:

```powershell
Push-Location monitoring/azureml
az ml environment create --file environment.yml --resource-group <RESOURCE_GROUP> --workspace-name <WORKSPACE>
az ml schedule create --file drift-schedule.yml --resource-group <RESOURCE_GROUP> --workspace-name <WORKSPACE>
Pop-Location
```

Those commands create billable Azure ML resources and were not run. Once configured, run
`python scripts/teardown_lab4.py` or `make teardown` to disable and delete only the named Lab 4
schedule tagged `course=itcs355, lab=4`. It does not delete the shared workspace, compute, data assets,
or any Lab 1–3 resources. The existing Lab 3 cleanup remains available as `make teardown-lab3`.

### GitHub staging setup and required graded evidence

CI runs for pull requests and pushes to `main`. CD checks out the exact successful `main` SHA, builds
and pushes the `itcs355-serve:<SHA>` image, updates the pre-created staging Container App, and smoke
tests readiness plus a prediction. Configure an Azure federated identity for the GitHub repository's
`staging` environment and give it only the required ACR push and staging-app deployment permissions.
Add GitHub Actions secrets `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, and these
repository variables: `AZURE_RESOURCE_GROUP`, `ACR_NAME`, `ACR_LOGIN_SERVER`, `ACA_NAME`,
`MODEL_REGISTRY_NAME`, and `MODEL_VERSION`. The staging app must already have an identity that can
pull from ACR and read the selected MLflow model, plus an ACA secret named `mlflow-tracking-uri`.

Two assignment artifacts still need your accounts: (1) create a temporary PR that deliberately
breaks a data contract, save its failing Actions run and named assertion, then close it without
merging; (2) configure the real recent-input export, Azure ML schedule, Pushgateway, and Slack Key
Vault secret, then capture the real alert timestamp and dashboard. They cannot be truthfully
substituted by this local simulation. Use the five-line postmortem template in
[`reports/lab4-drift.md`](reports/lab4-drift.md) after that cloud run.

---

# ITCS355 Lab 3 — Serving, Load Testing, and Rollback

This is a separate Lab 3 checkout, based on the completed Lab 2 branch. It adds a registry-backed
FastAPI service, a digest-pinned Azure Container Apps deployment, three-level load testing, batch and
payload experiments, metric-gated canary rollback, cost calculations, and tagged teardown. The
assignment specification is [`course/labs/lab-03-serving-and-rollback.md`](course/labs/lab-03-serving-and-rollback.md).

## Status and required user action

The service, Azure Container Apps adapter, managed deployment, load-test tooling, canary workflow,
cost worksheet, and tagged teardown are implemented. This checkout uses the existing Lab 2 Azure
ML registry and a separate Lab 3 Container Apps environment; `cloud.env` is local and gitignored.

**Cloud evidence completed (2026-09-28):** the public HTTPS endpoint was deployed and smoke-tested
with MLflow model version `1`. The selected `1cpu/2Gi` revision met the predeclared 200 ms p95 target at
10 concurrent Locust users (190 ms, zero errors); it first exceeded the target at 50 users (620 ms,
zero errors). The prior `0.5cpu/1Gi` run crossed the target at 10 users (370 ms). Batch and payload
experiments also ran against Azure. The 90/10 canary detected degradation in 129.9 seconds and
verified stable traffic returned to 100%. Cold start from confirmed zero replicas was 31.6 seconds;
the warm follow-up was 296 ms. The retail-price cost estimate is not the account's eventual invoice.

The app, identity, and both Lab 3 Container Apps environments have now been deleted and verified.
**Required user action:** no additional setup is needed to submit the code/evidence. Check Azure Cost
Analysis after usage posts to reconcile the estimate with your student subscription's actual charges.
Shared Lab 2 models, registry, ACR, and unrelated resources were retained.

### Azure deployment record

- App and endpoint: `itcs355-lab3` · Japan East · [HTTPS endpoint](https://itcs355-lab3.agreeablesand-112bc772.japaneast.azurecontainerapps.io) (decommissioned after evidence capture)
- Selected revision size: `1cpu/2Gi`; minimum replicas were zero; external ingress was temporary.
- Container image: `itcs3556688040.azurecr.io/itcs355@sha256:1a9275a36667e663f5fd1fd83ecf71cf1e33e771ce2331a364d805a86ed0c872`.
- Image was built/pushed locally with Docker because ACR Tasks are not enabled for this subscription.
- The standard Consumption-only environment supported managed identity and multi-revision canary traffic. The earlier Express environment was not used. Teardown verified both Lab 3 environments were deleted.
- The app pulls the registered MLflow artifact using its user-assigned identity. The serving loader trusts only the reviewed `sklearn.tree._tree.Tree` skops type required by model version 1; other unreviewed serialized types are rejected.

The endpoint is no longer running. Redeploy with `scripts/deploy_service.py` only if you intentionally
need it again; that can incur cloud charges.

### Local development (PowerShell)

Install the pinned load-generator dependencies after the locks below are available, create the
ignored synthetic dataset/model, and run the server in one terminal:

```powershell
python -m pip install -r requirements.txt
python -m venv .venv-loadtest
.\.venv-loadtest\Scripts\python.exe -m pip install --require-hashes -r requirements-loadtest.txt
python scripts/make_dataset.py
python scripts/export_model.py --out reports/model.joblib
$env:MODEL_PATH = (Resolve-Path reports/model.joblib).Path
$env:MODEL_VERSION = 'local'
python -m uvicorn service.app:app --host 127.0.0.1 --port 8080
```

In another terminal, run local verification and use the endpoint for development-only tests. The
Locust scripts are committed; generated CSVs are ignored.

```powershell
python -m pytest -q tests/
python scripts/run_loadtest.py --scenario concurrency --target http://127.0.0.1:8080 --p95-target-ms 200 --instance local
python scripts/run_loadtest.py --scenario payload --target http://127.0.0.1:8080 --instance local
python scripts/benchmark_batch.py --endpoint http://127.0.0.1:8080 --rows 100
python scripts/lab3_cost_report.py --utilization 0.25
```

Managed endpoint evidence used the pinned load-test requirements and a 200 ms p95 target declared
before measurement. At `1cpu/2Gi`, the endpoint met it at 1 and 10 users; the first tested breaking
level was 50 users. The 90/10 canary report records model metrics, revision weights, detection time,
and verified rollback. The cold-start report separately records scale-out and warm latency.

### Lab 3 evidence

| Deliverable | File / state |
|---|---|
| Service, schemas, probes, structured logs | `service/app.py`, `service/schemas.py`, `service/Dockerfile.serve` |
| Azure managed deployment and invoke/traffic adapter | `cloudlayer/azure.py` |
| 1/10/50 concurrency percentiles and breaking point | `reports/lab3-load.md`; `reports/lab3-load-small.md` retains the smaller-instance comparison |
| Batch-size and payload-size findings | `reports/lab3-batch.md`, `reports/lab3-payload.md` |
| Instance-size latency and retail cost delta | `reports/lab3-instance-size.md` |
| Canary metric, detection time, timestamped rollback | `reports/lab3-rollback-20260927T182921Z.md` (passed; rollback verified) |
| Scale-to-zero cold-start timing | `reports/lab3-cold-start.md` (31.6 s cold, 296 ms warm) |
| Cost / 1,000 and batch break-even | `reports/lab3-cost.md`; retail-price estimate, reconcile with actual Azure billing |
| Cloud deletion | `scripts/manage_azure.py teardown`; verified no Lab 3 app, identity, or environment remains |

The unit tests do not create cloud resources. `make teardown-lab3` is scoped to exact Lab 3 tags; it
leaves Lab 2 jobs, models, ACR, and unrelated resources untouched. Azure Cost Analysis may update
after teardown rather than immediately.

---

## Previous Lab 2 baseline — Experiment Tracking and Model Registry

> **Course materials live in [`course/`](course/README.md)** — syllabus, slides, the faculty
> specification, all five lab handouts, and the project brief. Every document is Markdown and
> renders on GitHub, diagrams included. New to the repo? Start with the
> [portability reference](course/reference/cloud-portability-reference.md).
> Keep this block when you edit the rest of this file; it is not part of the Lab 1 deliverable.

The Lab 2 baseline in this checkout carries the completed Lab 1 reproducibility foundation and
adds a resumable, budgeted study, comparison report, lineage-aware registration, staging promotion,
and registry reload check.

The selected cloud adapter is Azure. The storage, DVC remote, MLflow endpoint, digest-pinned image,
12-trial managed study, model registry, promotion, and registry reload are complete. Serverless and
low-priority Azure ML quotas are zero, so the study used the available dedicated `Standard_DS3_v2`
cluster with scale-to-zero. The exact quota and permission evidence is recorded in
[`reports/lab2-cloud-status.md`](reports/lab2-cloud-status.md).

---

## Local verification

The Lab 1 smoke test remains:

```bash
make reproduce
python scripts/verify_metric.py
```

The Lab 2 local study uses the same MLflow tracking and checkpoint logic as the managed study:

```bash
make tune
make compare
```

The checked-in comparison contains 12 distinct Azure ML trials varying `n_estimators`, `max_depth`,
and `min_samples_leaf`. It selected run `funny_berry_bk66p02czt` with validation ROC-AUC
`0.84259990` and test ROC-AUC `0.85328579`. Five seed runs for that configuration measured
validation ROC-AUC standard deviation `0.0141` and range `0.8426–0.8781`.

The comparison is in [`reports/lab2-comparison.md`](reports/lab2-comparison.md). It contains the
trial table, cost-per-point ranking, seed variance, training/monthly retraining cost, the model
choice, and the required failure mode. The cloud study's recorded managed-job duration cost is
`1.57275 THB`, below the `150 THB` budget. It is a provider-duration calculation; the Azure billing
page remains the source of truth for the invoice. Because the low-priority quota is zero, this is
dedicated-compute evidence rather than discounted-compute evidence.

The final local evidence was generated from committed code `18e9dbb8e21b3c315698ba16c074a12ed9f24627`
using an isolated SQLite tracking database. The selected local registry entry is
`itcs355-lab2-local` version `1`; [`reports/lab2-registry.md`](reports/lab2-registry.md) records
its lineage and reload result.

The Azure image was built for `linux/amd64` and pushed by digest as
`itcs3556688040.azurecr.io/itcs355@sha256:8d1de4c30566664467d3dfbf658c40e8d6d977f2d61dd9ba4e8bda0cdc6600cb`.
The DVC remote is the Azure Blob prefix `azure://itcs355/lab2/dvc`, and its push completed with
two objects transferred.

---

## Lab 2 implementation

| Area | Implementation |
|---|---|
| Managed training | `cloudlayer/azure.py` submits and polls Azure ML command jobs with a digest-pinned image, Blob input prefix, checkpoint output, compute target, tags, and model artifact. The configured target scales from 0 to 1 node. |
| Job context | `.amlignore` excludes local MLflow history, reports, caches, and credentials before Azure ML packages the command-job code. |
| Resumption | `src/tune.py` writes an atomic checkpoint before submission, records the job ID, and waits on an in-flight job after interruption. |
| Tracking | `src/train.py` logs hyperparameters, validation/test metrics, duration, cost, seed, Git SHA, data fingerprint, DVC version, training job ID, and image digest. |
| Comparison | `scripts/compare_runs.py` writes the table and a sub-200-word justification instead of leaving a grading placeholder. |
| Registry | `scripts/register_model.py` registers the MLflow model and writes all eight lineage fields; the Azure adapter also registers a versioned Azure ML custom model. |
| Promotion | MLflow staging/alias promotion is implemented locally; Azure promotion records an approved staging tag on the Azure ML model version. |
| Reload | `scripts/reload_check.py` loads `models:/MODEL_REGISTRY_NAME/VERSION` from the registry and scores five held-out rows. |

The eight registry fields are `git_commit`, `data_version`, `mlflow_run_id`, `training_job_id`,
`image_digest`, `seed`, `metric_val`, and `metric_test`. The staging owner should be a model owner
or release approver, not the person who trained the candidate alone. They should require the
comparison, seed-variance, data-version, image-digest, cost, and reload evidence before approving.

---

## Cloud run required before submission

These steps need user-owned cloud resources and credentials:

1. Sign in with `az login`, install the Azure ML CLI extension with `az extension add -n ml`, and
   confirm `python scripts/cloud_check.py` reports all checks passing. The configured `cloud.env`
   is local and ignored; never paste credentials into source files or Docker build arguments.
2. Use the existing Azure ML workspace, Blob Storage container, and ACR named in `cloud.env`. The
   submitting identity needs permission to read/write the storage data and create/describe jobs and
   model versions.
3. Start Docker, build for `linux/amd64`, push the image, and record the returned digest:

   ```bash
   make image-push
   ```

4. Configure a DVC object-storage remote under the configured `BLOB_URI`, run `dvc add data/raw`
   to populate the local DVC cache, then run `dvc push`. The training job must read the data prefix
   from object storage rather than from the laptop.
5. The managed study was run with the available dedicated quota. If low-priority quota is later
   granted, rerun the same checkpointed study with the discounted target; otherwise retain the
   documented dedicated fallback and confirm the invoice in Azure:

   ```bash
   make tune INSTANCE=Standard_DS3_v2 IMAGE_URI="$TRAINING_IMAGE_URI" \
     TUNE_FLAGS="--remote"
   make compare
   ```

6. The selected run is already registered and promoted: MLflow model `itcs355-6688040` version `1`
   and Azure ML custom model version `2`. A model owner or release approver should review the
   comparison, seed variance, data version, image digest, cost, and reload result before repeating
   that promotion in a real release. The equivalent command shape is:

   ```bash
   python scripts/register_model.py --run-id "$SELECTED_RUN_ID" \
     --name "$MODEL_REGISTRY_NAME" --training-job-id "$TRAINING_JOB_ID" \
     --image-uri "$TRAINING_IMAGE_URI" --image-digest "$IMAGE_DIGEST" \
     --provider-model-uri "$MODEL_ARTIFACT_URI" --stage Staging
   python scripts/reload_check.py --name "$MODEL_REGISTRY_NAME" --version "$VERSION"
   make cost EST="$ESTIMATE_THB" ACT="$ACTUAL_THB" RPS="$THROUGHPUT_RPS" INSTANCE=Standard_DS3_v2
   make teardown
   ```

The first managed submission may fail because submit-time and run-time identities are different.
Record the specific missing permission, fix only that permission, then repeat. A successful local
dry run cannot substitute for the managed-job and billing evidence.

---

## Evidence checklist

- [x] Lab 1 base image, dependency lock, DVC metadata, grouped split, and tracking foundation
- [x] 12-trial search space with three meaningful hyperparameters
- [x] Atomic checkpoint and in-flight job ID support for interruption/resumption
- [x] Comparison artifact and justification under 200 words
- [x] Five-seed variance evidence for the selected configuration
- [x] Local MLflow model version with all eight lineage fields
- [x] Local staging promotion and registry reload check
- [x] 12+ trials completed on managed compute with atomic checkpoint/resume
- [ ] 12+ trials completed on discounted managed compute (low-priority quota is currently zero)
- [ ] Actual Azure billing export/portal confirmation under 150 THB
- [x] Image pushed by digest and DVC remote push completed
- [x] Cloud registry version promoted and reloaded from the MLflow/Azure ML registry
- [x] Teardown performed: tagged jobs archived and tagged compute deleted
- [x] No `cloud.env` or credentials in Git history
