# Lab 2 registry evidence

This evidence was generated after implementation commit
`18e9dbb8e21b3c315698ba16c074a12ed9f24627`.

The local MLflow tracking database was isolated at `reports/lab2-final.db`; it is ignored and is
not a submission artifact. The provider registry path is implemented in `cloudlayer/aws.py`, but
the local run below does not claim that an AWS model package was created.

## Selected version

| Field | Value |
|---|---|
| Model name | `itcs355-lab2-local` |
| Version | `1` |
| URI | `models:/itcs355-lab2-local/1` |
| Stage/alias | `Staging` / `staging` |
| MLflow run | `3091d8f88b5b4409983473f40a127774` |
| Training job | `local-trial-01` |
| Validation ROC-AUC | `0.84259990` |
| Test ROC-AUC | `0.85328579` |

## Required lineage tags

| Tag | Value |
|---|---|
| `git_commit` | `18e9dbb8e21b3c315698ba16c074a12ed9f24627` |
| `data_version` | `1c886b512c8a5c9bf723da1cd119fc80.dir` |
| `mlflow_run_id` | `3091d8f88b5b4409983473f40a127774` |
| `training_job_id` | `local-trial-01` |
| `image_digest` | `local-development` |
| `seed` | `20260101` |
| `metric_val` | `0.84259990` |
| `metric_test` | `0.85328579` |

## Reload result

`python scripts/reload_check.py --name itcs355-lab2-local --version 1 --rows 5` passed. It loaded
`models:/itcs355-lab2-local/1` from the registry and scored five held-out readings; it did not load
`reports/model.joblib` or another local model file.

The cloud action still required for submission is to run the same flow with a real MLflow tracking
server and AWS Model Package Group, then replace the local development image digest and job ID with
the managed values.
