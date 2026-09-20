# ITCS355 Lab 2 — Experiment Tracking and Model Registry

> **Course materials live in [`course/`](course/README.md)** — syllabus, slides, the faculty
> specification, all five lab handouts, and the project brief. Every document is Markdown and
> renders on GitHub, diagrams included. New to the repo? Start with the
> [portability reference](course/reference/cloud-portability-reference.md).
> Keep this block when you edit the rest of this file; it is not part of the Lab 1 deliverable.

This checkout is a separate Lab 2 repository. It carries the completed Lab 1 reproducibility
foundation and adds a resumable, budgeted study, comparison report, lineage-aware registration,
staging promotion, and registry reload check.

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
