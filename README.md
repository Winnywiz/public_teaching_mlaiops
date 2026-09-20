# ITCS355 Lab 2 — Experiment Tracking and Model Registry

> **Course materials live in [`course/`](course/README.md)** — syllabus, slides, the faculty
> specification, all five lab handouts, and the project brief. Every document is Markdown and
> renders on GitHub, diagrams included. New to the repo? Start with the
> [portability reference](course/reference/cloud-portability-reference.md).
> Keep this block when you edit the rest of this file; it is not part of the Lab 1 deliverable.

This checkout is a separate Lab 2 repository. It carries the completed Lab 1 reproducibility
foundation and adds a resumable, budgeted study, comparison report, lineage-aware registration,
staging promotion, and registry reload check.

The selected cloud adapter is AWS. The local evidence below is explicitly labelled as a dry run;
managed compute, object storage, image pushing, and cloud registry promotion require the user's AWS
account and are not claimed without those external actions.

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

The checked-in comparison contains 12 distinct trials varying `n_estimators`, `max_depth`, and
`min_samples_leaf`. The local evidence selected run `3091d8f8` with validation ROC-AUC `0.8426`
and test ROC-AUC `0.8533`. Five seed runs for that configuration measured validation ROC-AUC
standard deviation `0.0139` and range `0.8364–0.8736`.

The comparison is in [`reports/lab2-comparison.md`](reports/lab2-comparison.md). It contains the
trial table, cost-per-point ranking, seed variance, training/monthly retraining cost, the model
choice, and the required failure mode. The local study estimated cost from discounted AWS rates;
it did not create a cloud job.

The final local evidence was generated from committed code `18e9dbb8e21b3c315698ba16c074a12ed9f24627`
using an isolated SQLite tracking database. The selected local registry entry is
`itcs355-lab2-local` version `1`; [`reports/lab2-registry.md`](reports/lab2-registry.md) records
its lineage and reload result.

---

## Lab 2 implementation

| Area | Implementation |
|---|---|
| Managed training | `cloudlayer/aws.py` submits and polls SageMaker jobs with a digest-pinned image, S3 input prefix, checkpoint path, spot flag, tags, and output artifact. |
| Resumption | `src/tune.py` writes an atomic checkpoint before submission, records the job ID, and waits on an in-flight job after interruption. |
| Tracking | `src/train.py` logs hyperparameters, validation/test metrics, duration, cost, seed, Git SHA, data fingerprint, DVC version, training job ID, and image digest. |
| Comparison | `scripts/compare_runs.py` writes the table and a sub-200-word justification instead of leaving a grading placeholder. |
| Registry | `scripts/register_model.py` registers the MLflow model and writes all eight lineage fields; `cloudlayer/aws.py` also supports SageMaker Model Package Groups. |
| Promotion | MLflow staging/alias promotion is implemented locally; AWS promotion maps to an approved SageMaker model package. |
| Reload | `scripts/reload_check.py` loads `models:/MODEL_REGISTRY_NAME/VERSION` from the registry and scores five held-out rows. |

The eight registry fields are `git_commit`, `data_version`, `mlflow_run_id`, `training_job_id`,
`image_digest`, `seed`, `metric_val`, and `metric_test`. The staging owner should be a model owner
or release approver, not the person who trained the candidate alone. They should require the
comparison, seed-variance, data-version, image-digest, cost, and reload evidence before approving.

---

## Cloud run required before submission

These steps need user-owned cloud resources and credentials:

1. Copy `cloud.env.example` to `cloud.env` and fill all eight capability slots. Keep the file
   untracked; never paste credentials into source files or Docker build arguments.
2. Give the SageMaker execution role read access to the data prefix, write access to the output and
   checkpoint prefixes, pull access to the container registry, and permission to write the MLflow
   tracking backend. Give the submitting identity permission to create/describe training jobs and
   model packages.
3. Start Docker, build for `linux/amd64`, push the image, and record the returned digest:

   ```bash
   make image-push
   ```

4. Configure a DVC object-storage remote under the configured `BLOB_URI`, run `dvc add data/raw`
   to populate the local DVC cache, then run `dvc push`. The training job must read the data prefix
   from object storage rather than from the laptop.
5. Run the discounted managed study and record actual billing rather than the local estimate:

   ```bash
   make tune INSTANCE=ml.m5.large IMAGE_URI="$TRAINING_IMAGE_URI" \
     TUNE_FLAGS="--remote --spot"
   make compare
   ```

6. Register the selected run and promote it only after review. Use the model artifact URI from the
   remote job result when provider registration is enabled:

   ```bash
   python scripts/register_model.py --run-id "$SELECTED_RUN_ID" \
     --name "$MODEL_REGISTRY_NAME" --training-job-id "$TRAINING_JOB_ID" \
     --image-uri "$TRAINING_IMAGE_URI" --image-digest "$IMAGE_DIGEST" \
     --provider-model-uri "$MODEL_ARTIFACT_URI" --stage Staging
   python scripts/reload_check.py --name "$MODEL_REGISTRY_NAME" --version "$VERSION"
   make cost EST="$ESTIMATE_THB" ACT="$ACTUAL_THB" RPS="$THROUGHPUT_RPS" INSTANCE=ml.m5.large
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
- [ ] 12+ trials completed on discounted managed compute
- [ ] Actual cloud billing recorded and confirmed under 150 THB
- [ ] Image pushed by digest and DVC remote push completed
- [ ] Cloud registry version promoted and reloaded from the provider registry
- [x] No `cloud.env` or credentials in Git history
