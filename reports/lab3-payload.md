# Lab 3 — payload-size experiment

Managed Endpoint Locust measurements on `https://itcs355-lab3.agreeablesand-112bc772.japaneast.azurecontainerapps.io`, size `0.5cpu/1Gi`, 10 users, 90:10
single:batch task ratio. Payload bytes are approximate;
the optional `metadata.padding` field is ignored by the model but validated and parsed by the API.


| Target payload bytes | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2651 | 90.983 | 110.0 | 120.0 | 140.0 | 0.000% |
| 4096 | 2581 | 88.574 | 110.0 | 130.0 | 190.0 | 0.000% |
| 16384 | 2494 | 85.549 | 110.0 | 140.0 | 210.0 | 0.000% |
| 60000 | 2142 | 73.326 | 120.0 | 220.0 | 310.0 | 0.000% |

At 60,000 bytes, p95 was 220.0 ms versus 120.0 ms at baseline (change +100.0 ms). This observed change includes run-to-run and service variation as well as request parsing/serialization.
