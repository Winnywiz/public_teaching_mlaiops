# Lab 5 — LLM evaluation evidence

The repository contains ten triage cases and offline recorded response fixtures. The baseline fixture passes 10/10; the deliberately degraded fixture makes the regression gate fail, including on an invented machine ID, an unsafe automatic decision, a part-number hallucination, an injection, and a failed safety refusal.

This verifies that the evaluation code can detect those regressions. It is not a live-model baseline: the fixture token counts and latencies are synthetic examples and are excluded from cost calculations. No Azure OpenAI endpoint or deployment was configured in the inherited `cloud.env`, so no provider request, actual token usage, output-cap experiment, or cache hit rate is claimed.

Run the offline checks with:

```powershell
python scripts/llm_eval.py --out reports/llm_eval-baseline.json
python scripts/llm_eval.py --responses evals/fixtures/triage-regressed.jsonl `
  --out reports/llm_eval-regressed.json `
  --baseline reports/llm_eval-baseline.json
```

The second command is expected to exit with code 1. Live evidence still requires an existing Azure OpenAI endpoint and deployment, Entra ID inference access, and a reviewed prompt/model change. No deployment is created by this project.
