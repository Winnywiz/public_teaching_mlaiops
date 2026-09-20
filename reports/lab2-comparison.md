# Lab 2 — Run comparison

Experiment `itcs355-lab2` · 12 trials · total spend 1.5727 THB

`thb_per_point` is cost per percentage point of validation ROC-AUC above the worst trial.

| run_id   |   val_roc_auc |   cost_thb |   thb_per_point |   n_estimators |   max_depth |   min_samples_leaf |     seed | training_job_id                          |
|:---------|--------------:|-----------:|----------------:|---------------:|------------:|-------------------:|---------:|:-----------------------------------------|
| funny_be |        0.8426 |     0.1102 |      0.0683     |            100 |           4 |                  5 | 20260101 | itcs355-6688040-lab2-trial-01-1789907661 |
| frank_ch |        0.8424 |     0.3398 |      0.2126     |            100 |           4 |                  1 | 20260101 | itcs355-6688040-lab2-trial-00-1789907289 |
| frosty_s |        0.8411 |     0.117  |      0.08       |            300 |           4 |                  5 | 20260101 | itcs355-6688040-lab2-trial-07-1789908301 |
| frank_be |        0.8404 |     0.1125 |      0.0809     |            300 |           4 |                  1 | 20260101 | itcs355-6688040-lab2-trial-06-1789908192 |
| sad_matc |        0.8397 |     0.1102 |      0.0833     |            100 |           8 |                  5 | 20260101 | itcs355-6688040-lab2-trial-03-1789907911 |
| musing_b |        0.8377 |     0.1125 |      0.1001     |            300 |           8 |                  5 | 20260101 | itcs355-6688040-lab2-trial-09-1789908537 |
| polite_s |        0.8354 |     0.1148 |      0.1288     |            300 |          12 |                  5 | 20260101 | itcs355-6688040-lab2-trial-11-1789908729 |
| placid_c |        0.8338 |     0.1148 |      0.1569     |            300 |           8 |                  1 | 20260101 | itcs355-6688040-lab2-trial-08-1789908429 |
| plum_spa |        0.8322 |     0.108  |      0.1893     |            100 |          12 |                  5 | 20260101 | itcs355-6688040-lab2-trial-05-1789908105 |
| dreamy_b |        0.8312 |     0.108  |      0.2285     |            100 |           8 |                  1 | 20260101 | itcs355-6688040-lab2-trial-02-1789907747 |
| khaki_ha |        0.8268 |     0.1102 |      2.9988     |            100 |          12 |                  1 | 20260101 | itcs355-6688040-lab2-trial-04-1789908017 |
| strong_b |        0.8265 |     0.1148 |      1.1475e+06 |            300 |          12 |                  1 | 20260101 | itcs355-6688040-lab2-trial-10-1789908641 |

## Selected model

- Full MLflow run ID: `funny_berry_bk66p02czt`
- Validation ROC-AUC: `0.84259990`
- Test ROC-AUC: `0.85328579`
- Training job: `itcs355-6688040-lab2-trial-01-1789907661`
- Seed variance runs: `5`

## Justification (under 200 words)

I selected run funny_be (100 trees, depth 4, leaf 5) with validation ROC-AUC 0.8426. The selected run is also the highest-scoring run, but its margin is shown against the seed variance rather than treated as proof by itself. Five-seed variance for this configuration is std=0.0141, range 0.8426–0.8781. Its measured or estimated training cost is 0.1103 THB and retraining it 30 times per month would cost about 3.3075 THB. This choice could be wrong if future machines operate outside the grouped hold-out distribution; the model may then rank risk poorly despite a clean offline score.

## Registration and promotion

Register the selected run with `scripts/register_model.py`; it writes all eight lineage fields to the model version. Promotion to staging must be performed by the model owner or release approver after checking the comparison table, seed variance, data version, image digest, and reload result. The approver should reject a version with missing lineage, a failed reload, or a cost outside the budget.

## Cost note

The study budget is 150 THB. The reported total is 1.5727 THB. Low-priority quota was unavailable in this Azure subscription, so the study used dedicated scale-to-zero compute. Reconcile the provider-duration cost with the Azure billing portal before submission.
