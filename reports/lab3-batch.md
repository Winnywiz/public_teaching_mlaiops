# Lab 3 — batch-size experiment

Managed Endpoint HTTP measurements.

- Target: `https://itcs355-lab3.agreeablesand-112bc772.japaneast.azurecontainerapps.io`
- Replica size: `1cpu/2Gi`
- Rows per comparison: 100
- Repetitions: 5
- 100 individual calls (concurrent fan-out) median wall time: **1725.389 ms**
- One `/predict/batch` call median wall time: **313.637 ms**
- Batch speedup: **5.501×**

The single-call baseline fans requests out concurrently with at most 32 client workers; it is not a
sequential-loop comparison. This distinction is intentional so it does not exaggerate batch gains.
The request metrics include network/JSON overhead and therefore differ from model-only inference time.
