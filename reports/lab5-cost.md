# Lab 5 — Cost report

Status: cloud cost evidence is pending. No Lab 5 pipeline, schedule, staging app, or LLM request has been created or run from this checkout. This is not evidence of zero account spend; shared Lab 1–4 resources and earlier usage must still be checked separately.

| Required item | Evidence |
|---|---|
| Estimate committed before the first Lab 5 run | Not recorded before a run. This requirement is not yet satisfied. |
| Actual billing filtered to `course=itcs355`, the student tag, and `lab=5` | Not available because there has been no Lab 5 run. |
| Estimate-to-actual gap and explanation | Not measurable yet. |
| Training, storage, serving, pipeline, and monitoring breakdown | Not available. Export the tagged Azure Cost Management data after usage posts. |
| Cost per 1,000 predictions at 5%, 25%, and 80% utilisation | Not reported: no Lab 5 serving throughput or verified regional rate was measured. |
| Applied optimisation with before/after cost and p95 | Not demonstrated. |

Do not replace the missing values with the Lab 2 job-duration estimate or Lab 3 retail-price estimate. Those are different workloads and are not Lab 5 invoice evidence. After collecting the real inputs, use `scripts/cost_report.py`; provide the exact rate source, billing export, five component totals, and measured before/after values. The script intentionally refuses to guess a regional Container Apps rate.
