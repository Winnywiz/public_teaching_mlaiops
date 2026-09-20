# Lab 2 — Run comparison

Experiment `itcs355-lab2` · 12 trials · total spend 0.0014 THB

`thb_per_point` is cost per percentage point of validation ROC-AUC above the worst trial.

| run_id   |   val_roc_auc |   cost_thb |   thb_per_point |   n_estimators |   max_depth |   min_samples_leaf |     seed | training_job_id   |
|:---------|--------------:|-----------:|----------------:|---------------:|------------:|-------------------:|---------:|:------------------|
| 3091d8f8 |        0.8426 |     0.0001 |          0      |            100 |           4 |                  5 | 20260101 | local-trial-01    |
| b603eec3 |        0.8424 |     0.0001 |          0.0001 |            100 |           4 |                  1 | 20260101 | local-trial-00    |
| 8959f6ac |        0.8411 |     0.0002 |          0.0001 |            300 |           4 |                  5 | 20260101 | local-trial-07    |
| 21b5046d |        0.8404 |     0.0002 |          0.0001 |            300 |           4 |                  1 | 20260101 | local-trial-06    |
| 65547595 |        0.8397 |     0.0001 |          0      |            100 |           8 |                  5 | 20260101 | local-trial-03    |
| 1f4a6a1d |        0.8377 |     0.0001 |          0.0001 |            300 |           8 |                  5 | 20260101 | local-trial-09    |
| 3dee8c02 |        0.8354 |     0.0002 |          0.0002 |            300 |          12 |                  5 | 20260101 | local-trial-11    |
| aa056ccc |        0.8338 |     0.0002 |          0.0002 |            300 |           8 |                  1 | 20260101 | local-trial-08    |
| dbeac59f |        0.8322 |     0.0001 |          0.0001 |            100 |          12 |                  5 | 20260101 | local-trial-05    |
| 9e710a7e |        0.8312 |     0.0001 |          0.0001 |            100 |           8 |                  1 | 20260101 | local-trial-02    |
| fd199559 |        0.8268 |     0.0001 |          0.0019 |            100 |          12 |                  1 | 20260101 | local-trial-04    |
| 29f547f3 |        0.8265 |     0.0002 |       1535.12   |            300 |          12 |                  1 | 20260101 | local-trial-10    |

## Selected model

- Full MLflow run ID: `3091d8f88b5b4409983473f40a127774`
- Validation ROC-AUC: `0.84259990`
- Test ROC-AUC: `0.85328579`
- Training job: `local-trial-01`
- Seed variance runs: `5`

## Justification (under 200 words)

I selected run 3091d8f8 (100 trees, depth 4, leaf 5) with validation ROC-AUC 0.8426. The selected run is also the highest-scoring run, but its margin is shown against the seed variance rather than treated as proof by itself. Five-seed variance for this configuration is std=0.0139, range 0.8364–0.8736. Its measured or estimated training cost is 0.0001 THB and retraining it 30 times per month would cost about 0.0020 THB. This choice could be wrong if future machines operate outside the grouped hold-out distribution; the model may then rank risk poorly despite a clean offline score.

## Registration and promotion

Register the selected run with `scripts/register_model.py`; it writes all eight lineage fields to the model version. Promotion to staging must be performed by the model owner or release approver after checking the comparison table, seed variance, data version, image digest, and reload result. The approver should reject a version with missing lineage, a failed reload, or a cost outside the budget.

## Cost note

The study budget is 150 THB. The reported total is 0.0014 THB. Cloud runs must use discounted compute and replace estimates with the provider billing figure before submission.
