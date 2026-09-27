# Lab 3 — serving cost report

Instance configuration: `1cpu/2Gi` · utilisation assumption: **25%**.
The request mix is represented by **5.900 predictions/HTTP request**.
The hourly rate is the active-replica estimate; request charges are added when a retail rate is provided.
Price basis: public Azure retail rates for **Japan East**, retrieved **2026-09-28**.
This is not the account invoice. Azure Consumption has monthly per-subscription grants of 180,000
vCPU-seconds, 360,000 GiB-seconds, and 2 million HTTP requests; whether they remain available is
not visible to this worksheet.

| Input / result | Value |
|---|---:|
| Active replica hourly cost (THB/hour) | 3.553200 |
| One-replica idle hourly cost if kept warm (THB/hour) | 1.069200 |
| Measured HTTP throughput (requests/second) | 70.084000 |
| Implied prediction throughput (predictions/second) | 413.495600 |
| HTTP request retail price (THB / million requests) | 13.155000 |
| Measured batch request duration (ms) | 313.637000 |
| Predictions per batch request | 100 |
| Active hourly rate used for batch estimate (THB/hour) | 3.553200 |
| Utilisation assumption | 25% |
| Cost per 1,000 predictions (THB) | 0.011778 |
| Batch inference cost per 1,000 predictions (THB) | 0.003227 |

Method: `hourly_rate × 1000 ÷ (HTTP requests/s × predictions/request × 3600 × utilisation)`,
plus the request-meter cost for `1000 ÷ predictions/request` HTTP calls. This steady-active estimate
excludes scale-to-zero, startup overhead, storage, and egress; compare it with actual tagged Azure billing.
When derived from batch timing, the batch estimate is `calls_per_1000 × (active_hourly_rate ×
median_call_seconds ÷ 3600 + HTTP_request_price_per_call)`.
Sources: [Azure Container Apps pricing](https://azure.microsoft.com/en-us/pricing/details/container-apps/),
[Container Apps billing](https://learn.microsoft.com/en-us/azure/container-apps/billing), and the
[Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices).

Batch vs. one continuously idle (warm) replica: approximately **7,951,543 predictions/day**.
Batch vs. one continuously active replica: approximately **26,424,825 predictions/day**.

These break-even values compare the measured batch unit cost with one continuously allocated replica; scale-to-zero can be cheaper when idle, while startup latency and service-level requirements still matter.
