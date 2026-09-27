# Lab 3 — load-test results

> These are managed endpoint measurements. Cold-start is recorded separately; actual tagged
> Azure billing must still be reconciled in the account.


## Measurement setup

- Target: `https://itcs355-lab3.agreeablesand-112bc772.japaneast.azurecontainerapps.io`
- Load generator: Locust, headless, 1 / 10 / 50 users, `30s` per run
- Stated p95 target, set before measurement: **200 ms**
- Service size: `0.5cpu/1Gi` (managed endpoint)
- Approximate request payload: 0 bytes
- Host: Windows-11-10.0.26200-SP0 · Python 3.12.10

| Concurrent users | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 270 | 9.270 | 110.0 | 110.0 | 120.0 | 0.000% |
| 10 | 1538 | 52.782 | 140.0 | 370.0 | 410.0 | 0.000% |
| 50 | 3096 | 106.189 | 200.0 | 1300.0 | 1400.0 | 0.000% |

Breaking point: First exceeded at 10 concurrent users. A failed request or p95 above the stated target counts as breaking.
The smallest tested configuration meeting the target is 1 concurrent user.
Locust mixes single and 50-row batch calls (9:1 task weight); its request throughput counts HTTP
requests, not individual predictions. Use `make batch-test` for prediction-throughput comparison.

## Related evidence

| Variable | Current result | Remaining cloud measurement |
|---|---|---|
| Cold start | `reports/lab3-cold-start.md` | Captured separately from steady-state percentiles |
| Batch size 100 vs. 100 singles | `reports/lab3-batch.md` | Managed comparison recorded
| Payload size | `reports/lab3-payload.md` | Managed comparison recorded
| Instance-size latency and cost | `reports/lab3-instance-size.md` | Comparison recorded
| Canary detection and rollback | `reports/lab3-rollback-20260927T182921Z.md` | Timestamped traffic and rollback evidence recorded
| Cost per 1,000 predictions | `reports/lab3-cost.md` | Reconcile retail estimate with actual Azure billing
