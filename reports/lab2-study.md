# Lab 2 study evidence

Implementation commit: `18e9dbb8e21b3c315698ba16c074a12ed9f24627`
Experiment: `itcs355-lab2`
Study mode: local dry run using the discounted AWS rate table; no cloud job was created.
Budget: `150 THB`  
Recorded estimate: `0.0014 THB`

## Search design

The study ran all 12 combinations of three behavior-changing hyperparameters:

- `n_estimators`: 100, 300
- `max_depth`: 4, 8, 12
- `min_samples_leaf`: 1, 5

Each trial logged parameters, validation and test metrics, duration, estimated cost, seed, data
fingerprint, DVC version, Git SHA, training job ID, and a model artifact to MLflow.

## Seed variance

The selected configuration was evaluated with seeds `20260101` through `20260105`.

| Statistic | Value |
|---|---:|
| Validation ROC-AUC standard deviation | 0.0139 |
| Minimum | 0.8364 |
| Maximum | 0.8736 |
| Number of seeds | 5 |

## Resume evidence

After the 12 trials completed, rerunning the same command with the checkpoint printed
`already done, skipping (resumed from checkpoint)` for trials 0 through 11 and preserved the
recorded spend. The checkpoint stores an in-flight job ID before a managed submission, so an
interruption can wait on the existing job rather than silently launch a duplicate.

The comparison table and under-200-word selection argument are in
[`lab2-comparison.md`](lab2-comparison.md). Cloud execution, real billing, and AWS registry
promotion remain user-owned actions because they require the account, permissions, and image/data
endpoints listed in the root README.
