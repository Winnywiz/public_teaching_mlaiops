# Lab 4 local drift evidence

This is a local synthetic-data detector run. It is not evidence of a Slack delivery, a hosted
Pushgateway write, an Azure schedule run, or a production dashboard capture.

- Injection: `temp_c` shifted upward by 6°C in 6,000 generated rows.
- Injected at: `2026-10-11T13:26:23.510Z`.
- Detector evaluated at: `2026-10-11T13:26:23.993Z`.
- Observed detector latency: `0.484 seconds` from injection to threshold detection. There was no
  delivered channel alert, so this must not be reported as injection-to-alert time.
- Result: `temp_c` PSI `0.38333`, KS `0.24567`, against PSI threshold `0.20`; the other five
  features had PSI `0.00000`.
- Metric delivery: not configured for this local evidence run. Alert delivery: not configured.

## Five-line postmortem

```text
What fired: The local detector crossed PSI 0.20 for temp_c at 2026-10-11T13:26:23.993Z; no external notification was sent.
True cause: The exercise deliberately added 6°C to the generated temperature feature. It was not an upstream failure.
Retrain, roll back, or no action, and why: No retrain. Restore the unmodified input and keep the known-good model because the shift was injected solely to test detection.
What this would have cost if unnoticed for a week: This local run cost $0 in cloud. Production loss cannot be priced from synthetic rows; estimate affected weekly predictions times the measured excess failure rate times the cost per missed failure.
How to prevent or detect it faster: Keep schema/range CI gates, export a daily production feature window, run the scheduled PSI check, and page Slack on the threshold breach.
```

## Still required for submission

1. Create the deliberately broken data-contract PR, save the failing GitHub Actions run and exact
   assertion, then close it without merging.
2. Configure the Azure ML workspace, compute, reference/current data assets, Key Vault identity and
   secrets, and reachable authenticated Pushgateway. Run the schedule and save its real Slack alert
   timestamp plus the dashboard showing the shift.
3. Replace the no-cost local-only statement above with a justified production weekly-cost estimate
   once traffic volume and the cost of a missed failure are known.
4. Start Docker Desktop and run the serving-image integration check and local Grafana stack if you
   want container-level verification on this machine. The engine was unavailable during this run.
