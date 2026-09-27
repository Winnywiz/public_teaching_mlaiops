# Lab 3 — instance-size and cost comparison

Managed Azure Container Apps runs in Japan East. Both tests used the same image/model, request mix,
30-second Locust duration, and predeclared **200 ms p95 target**. The request mix is 90% single-row
calls and 10% 50-row batch calls (5.9 predictions per HTTP request on average).

| Users | `0.5cpu/1Gi` req/s | p95 (ms) | Errors | `1cpu/2Gi` req/s | p95 (ms) | Errors |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.270 | 110 | 0 | 9.361 | 110 | 0 |
| 10 | 52.782 | 370 | 0 | 70.084 | 190 | 0 |
| 50 | 106.189 | 1300 | 0 | 196.613 | 620 | 0 |

At 10 users, the larger instance improved p95 by **180 ms (48.6%)** and request throughput by
**32.8%**; it was the first tested size to meet the 200 ms target at this concurrency. At 50 users,
throughput rose **85.2%** and p95 improved **52.3%**, but the larger replica still exceeded the
target. No request errors occurred in either series.

## Retail-price estimate

Microsoft's Retail Prices API returned these Japan East Consumption rates in THB on 2026-09-28:

- Active vCPU: 0.000789 THB per vCPU-second.
- Active memory: 0.000099 THB per GiB-second.
- HTTP requests: 13.155 THB per million requests.
- Idle vCPU: 0.000099 THB per vCPU-second; idle memory: 0.000099 THB per GiB-second.

| Replica size | Active rate / hour | One-replica idle rate / hour |
|---|---:|---:|
| `0.5cpu/1Gi` | 1.7766 THB | 0.5346 THB |
| `1cpu/2Gi` | 3.5532 THB | 1.0692 THB |
| Delta | +1.7766 THB (+100%) | +0.5346 THB (+100%) |

Cost per 1,000 predictions uses the 10-user observed request throughput, 5.9 predictions per HTTP
request, and a stated **25% utilization assumption**. The calculation includes HTTP request retail
charges and gives approximately **0.00857 THB/1,000 predictions** for `0.5cpu/1Gi` and
**0.01178 THB/1,000 predictions** for `1cpu/2Gi` (about 37.5% higher at the selected size). This
is a steady-active, public-retail estimate, not the Azure for Students invoice. Monthly subscription
grants, other subscription consumption, discounts, startup, storage, and egress can change actual
charges. Microsoft lists monthly per-subscription grants of 180,000 vCPU-seconds, 360,000 GiB-seconds,
and 2 million requests; scale-to-zero has no app-usage charge at zero replicas.

Price sources: [Azure Container Apps pricing](https://azure.microsoft.com/en-us/pricing/details/container-apps/),
[Container Apps billing](https://learn.microsoft.com/en-us/azure/container-apps/billing), and the
[Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices).
The API query was filtered to `serviceName=Azure Container Apps`, `armRegionName=japaneast`,
Consumption rates, and `currencyCode=THB`.
