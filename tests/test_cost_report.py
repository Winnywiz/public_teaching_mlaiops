"""Offline contract tests for Lab 5's evidence-led cost report."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cost_report_keeps_sources_and_reports_applied_change(tmp_path):
    out = tmp_path / "cost.md"
    command = [
        sys.executable, str(ROOT / "scripts" / "cost_report.py"),
        "--estimate", "120", "--actual", "130", "--rps", "42",
        "--instance", "0.5cpu/1Gi", "--hourly-rate-thb", "0.1",
        "--price-source", "test-only", "--billing-source", "test-export",
        "--gap-explanation", "The test-only example differs by ten THB.",
        "--training-thb", "20", "--storage-thb", "10", "--serving-thb", "80",
        "--pipeline-thb", "15", "--monitoring-thb", "5",
        "--optimization", "scale-to-zero", "--before-config", "always-on",
        "--after-config", "scale-to-zero", "--before-rps", "42", "--after-rps", "42",
        "--before-hourly-rate-thb", "0.2", "--after-hourly-rate-thb", "0.1",
        "--before-p95-ms", "150", "--after-p95-ms", "180", "--out", str(out),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = out.read_text()
    assert "test-export" in report
    assert "scale-to-zero" in report
    assert "5%" in report and "25%" in report and "80%" in report
    assert "TODO" not in report
