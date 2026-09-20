# Lab 2 study evidence

Study mode: Azure ML managed command jobs on `itcs355-cpu` (`Standard_DS3_v2`, dedicated,
autoscaling 0–1). Low-priority quota was tested and Azure returned zero available vCPUs, so the
available dedicated quota was used as the documented fallback.

## Search design

The study completed all 12 combinations of three behavior-changing hyperparameters:

- `n_estimators`: 100, 300
- `max_depth`: 4, 8, 12
- `min_samples_leaf`: 1, 5

Each managed trial used the DVC-backed Azure Blob input, the digest-pinned ACR image, an atomic
checkpoint, Azure ML MLflow tracking, and uploaded model/checkpoint artifacts. The comparison table
and the under-200-word selection argument are in [`lab2-comparison.md`](lab2-comparison.md).

## Cost

The recorded managed-job duration estimate is `1.57275 THB`, below the `150 THB` budget. It is a
provider-duration calculation; the Azure billing page remains the source of truth for the invoice.

## Seed variance

The selected configuration was evaluated with seeds `20260101` through `20260105` in the
`itcs355-lab2-seed-variance` experiment. Validation ROC-AUC standard deviation is `0.0141`, with a
range of `0.8426–0.8781`.

## Resume and failure evidence

The checkpoint recorded an in-flight Azure job before submission. After an interrupted local
controller, it resumed the existing job rather than launching a duplicate. The first serverless
failure documented the zero-core quota, the first dedicated failure documented missing ACR pull
permission, and the subsequent run completed after those provider-specific issues were resolved.

## Registry evidence

The selected run is `funny_berry_bk66p02czt`. MLflow model version `1` and Azure ML custom model
version `2` carry all eight lineage fields. The registry reload check loaded version `1` and scored
five held-out rows successfully.

## Teardown

All tagged Lab 2 jobs were archived and the tagged Azure ML compute target was deleted after the
study. The workspace, registry versions, MLflow runs, and stored artifacts remain available.
