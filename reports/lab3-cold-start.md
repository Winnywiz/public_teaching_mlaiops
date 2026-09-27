# Lab 3 — scale-to-zero cold-start evidence

- Revision observed at zero replicas: `itcs355-lab3--v1-1790532049`
- Zero replicas confirmed (UTC): 2026-09-27T19:12:02.999+00:00
- First request started (UTC): 2026-09-27T19:12:09.335+00:00
- First successful request: HTTP 200 after **31596.7 ms** (2 attempt(s))
- Immediately warm follow-up: HTTP 200 in **295.7 ms**
- Model version: `1`
- First response request ID: `a8e6b596-bedc-4da9-adc2-2fb21824d196`
- Non-success response detail before readiness: `TimeoutError: The read operation timed out`

This includes scale-out and model load/readiness time; it is recorded separately from steady-state
load-test percentiles. The first request kept the app warm and potentially billable during measurement.
The tag-scoped Lab 3 teardown was subsequently completed; Azure verified no tagged app, identity, or
Container Apps environment remains.
