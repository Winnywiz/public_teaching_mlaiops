# Lab 2 Azure cloud status

The Azure adapter, storage path, DVC remote, MLflow tracking endpoint, ACR image, managed
training, registry, promotion, and reload check were exercised against the configured Azure
subscription and workspace.

## Resources and permissions

- Workspace: `itcs3556688040-ml` in `japaneast`
- DVC remote: `azure://itcs355/lab2/dvc` — two objects pushed
- Training image: `itcs3556688040.azurecr.io/itcs355@sha256:8d1de4c30566664467d3dfbf658c40e8d6d977f2d61dd9ba4e8bda0cdc6600cb`
- Compute: `itcs355-cpu`, `Standard_DS3_v2`, dedicated, min 0 / max 1, idle scale-down 120 seconds
- Compute identity: system-assigned identity with `AcrPull` on the ACR

The first dedicated attempt reached Azure but failed to pull the private image because the compute
had no registry identity. The specific fix was to enable the compute's system-assigned identity and
grant only `AcrPull` on the registry. No registry password or credential was added to the repo.

## Managed study

Two serverless attempts (`quirky_fennel_n562hjxjw6` and `frosty_cup_cxcwtyj682`) were rejected by
the subscription's zero-core serverless quota. A low-priority cluster was also tested and Azure
reported a zero low-priority vCPU quota; the failed `itcs355-spot` target was removed.

The study therefore ran on the available dedicated Dsv2 quota. Twelve trials completed with
checkpoint/resume support and measured job-duration cost of `1.57275 THB` against the `150 THB`
study budget. The study is managed-compute evidence, but it is not discounted/low-priority
evidence because that quota is unavailable.

Selected run: `funny_berry_bk66p02czt`

- Configuration: 100 trees, depth 4, minimum leaf 5
- Validation ROC-AUC: `0.84259990`
- Test ROC-AUC: `0.85328579`
- Training job: `funny_berry_bk66p02czt`
- Five-seed validation standard deviation: `0.0141`, range `0.8426–0.8781`

## Registry and reload

- MLflow model: `itcs355-6688040` version `1`, staged
- Azure ML custom model: `itcs355-6688040` version `2`, `Staging` / `Approved`
- `scripts/reload_check.py --name itcs355-6688040 --version 1 --rows 5`: passed
- Teardown: all tagged Lab 2 jobs archived and the tagged compute target deleted; registry and artifacts remain

Both registry entries carry the eight required lineage fields: Git commit, DVC data version,
MLflow run ID, training job ID, image digest, seed, validation metric, and test metric.

## Remaining user action

If the course marker requires the literal discounted-compute criterion, request low-priority Azure
ML vCPUs for `japaneast` (at least 4) and rerun the same checkpointed study with the low-priority
compute target. Otherwise, the dedicated fallback is complete and under budget. Azure billing should
still be checked in the portal because the cost above is calculated from the managed-job durations,
not an invoice export.
