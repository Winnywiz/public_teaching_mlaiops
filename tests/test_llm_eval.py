"""Contract tests for the LLM cost model and the evaluation gate. Lab 5 Part B.

The gate is the thing that has to be trustworthy. A gate that cannot fail is decoration,
so most of what follows checks that it fails when it should — the same reasoning as
tests/test_data.py, applied to text.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import llmcost  # noqa: E402

GOLDEN = ROOT / "evals" / "golden" / "triage.jsonl"
BASELINE = ROOT / "evals" / "fixtures" / "triage-baseline.jsonl"
REGRESSED = ROOT / "evals" / "fixtures" / "triage-regressed.jsonl"


def run_eval(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "llm_eval.py"), *args],
        capture_output=True, text=True, cwd=ROOT,
    )


# --- the golden set is a data contract of its own ----------------------------

def test_golden_set_is_wellformed():
    ids = []
    for line in GOLDEN.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        assert case["id"] and case["prompt"], "every case needs an id and a prompt"
        assert case["checks"], f"{case['id']} asserts nothing, so it can never fail"
        ids.append(case["id"])
    assert len(ids) >= 10, "Lab 5 requires at least 10 golden cases"
    assert len(ids) == len(set(ids)), "duplicate case ids"


def test_every_golden_case_has_a_recorded_response():
    cases = {json.loads(x)["id"] for x in GOLDEN.read_text().splitlines() if x.strip()}
    for fixture in (BASELINE, REGRESSED):
        got = {json.loads(x)["id"] for x in fixture.read_text().splitlines() if x.strip()}
        assert cases <= got, f"{fixture.name} is missing {sorted(cases - got)}"


# --- the gate ----------------------------------------------------------------

def test_baseline_fixture_passes_every_case(tmp_path):
    out = tmp_path / "report.json"
    proc = run_eval("--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out.read_text())
    assert report["pass_rate"] == 1.0, [r for r in report["results"] if not r["passed"]]


def test_gate_fails_on_a_regressed_response(tmp_path):
    base = tmp_path / "base.json"
    assert run_eval("--out", str(base)).returncode == 0

    proc = run_eval("--responses", str(REGRESSED),
                    "--out", str(tmp_path / "now.json"), "--baseline", str(base))
    assert proc.returncode == 1, "a regressed fixture must fail the gate"
    assert "GATE FAILED" in proc.stdout
    # Regressions cover unsafe automatic decisions, hallucinated facts, injection,
    # missing identity, and a guardrail refusal that no longer fires.
    for case_id in ("triage-003", "triage-004", "triage-007", "triage-009", "triage-010"):
        assert case_id in proc.stdout


def test_gate_passes_against_itself(tmp_path):
    base = tmp_path / "base.json"
    assert run_eval("--out", str(base)).returncode == 0
    proc = run_eval("--out", str(tmp_path / "now.json"), "--baseline", str(base))
    assert proc.returncode == 0
    assert "GATE PASSED" in proc.stdout


# --- cost model --------------------------------------------------------------

def test_output_tokens_cost_more_than_input():
    rates = llmcost.token_rates("azure", "test-deployment", rates=(1.0, 4.0))
    input_only = llmcost.Usage(input_tokens=1000, output_tokens=0)
    output_only = llmcost.Usage(input_tokens=0, output_tokens=1000)
    assert llmcost.request_cost("azure", "test", output_only, rates=rates) > llmcost.request_cost(
        "azure", "test", input_only, rates=rates
    )


def test_cached_input_is_cheaper_than_fresh_input():
    fresh = llmcost.Usage(input_tokens=1000, output_tokens=100)
    cached = llmcost.Usage(input_tokens=1000, output_tokens=100, cached_input_tokens=900)
    rates = (10.0, 40.0)
    assert llmcost.request_cost(
        "azure", "test", cached, rates=rates, cache_read_factor=0.1
    ) < llmcost.request_cost(
        "azure", "test", fresh, rates=rates
    )


def test_cached_tokens_cannot_exceed_input():
    with pytest.raises(ValueError, match="cannot exceed"):
        llmcost.Usage(input_tokens=100, output_tokens=10, cached_input_tokens=101)


def test_unknown_model_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="No verified token rates"):
        llmcost.token_rates("gcp", "definitely-not-a-model")


def test_capping_output_saves_money_and_capping_upward_saves_nothing():
    usage = llmcost.Usage(input_tokens=400, output_tokens=800)
    rates = (100.0, 500.0)
    assert llmcost.output_cap_saving("azure", "test", usage, 200, rates=rates) > 0
    assert llmcost.output_cap_saving("azure", "test", usage, 900, rates=rates) == 0.0


def test_cache_breakeven_is_a_fraction():
    rate = llmcost.cache_breakeven_hit_rate(
        "azure", "test", prefix_tokens=2000,
        cache_read_factor=0.1, cache_write_factor=1.25,
    )
    assert 0.0 < rate < 1.0


def test_fixture_token_counts_are_never_priced_as_real_usage(tmp_path):
    out = tmp_path / "fixture-report.json"
    proc = run_eval(
        "--input-rate-thb-per-million", "10",
        "--output-rate-thb-per-million", "40",
        "--pricing-source", "test pricing source",
        "--out", str(out),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(out.read_text())
    assert report["usage_source"] == "synthetic-fixture"
    assert report["thb_per_1k_requests"] is None
