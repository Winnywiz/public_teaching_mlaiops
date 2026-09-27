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

The unit tests do not create cloud resources. `make teardown` is scoped to exact Lab 3 tags; it
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
