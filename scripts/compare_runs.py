"""Rank Lab 2 runs by quality and cost, then write the selection justification."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import pandas as pd

from src import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="itcs355-lab2")
    parser.add_argument("--metric", default="val_roc_auc")
    parser.add_argument("--seed-experiment", default="itcs355-lab2-seed-variance")
    parser.add_argument("--monthly-trials", type=int, default=30)
    parser.add_argument("--chosen-run", default=None, help="full or abbreviated run ID")
    parser.add_argument("--out", type=Path, default=Path("reports/lab2-comparison.md"))
    return parser.parse_args()


def _value(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return default if pd.isna(value) else float(value)


def _seed_stats(experiment_name: str, chosen: pd.Series) -> tuple[int, float, float, float]:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return 0, float("nan"), float("nan"), float("nan")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty or "metrics.val_roc_auc" not in runs:
        return 0, float("nan"), float("nan"), float("nan")
    matches = runs
    for param in ("n_estimators", "max_depth", "min_samples_leaf"):
        column = f"params.{param}"
        selected = str(chosen.get(column, ""))
        if column in matches and selected:
            matches = matches[matches[column].astype(str) == selected]
    values = matches["metrics.val_roc_auc"].dropna().astype(float)
    if values.empty:
        values = runs["metrics.val_roc_auc"].dropna().astype(float)
    if values.empty:
        return 0, float("nan"), float("nan"), float("nan")
    return len(values), float(values.std(ddof=1) if len(values) > 1 else 0.0), float(values.min()), float(values.max())


def main() -> int:
    args = parse_args()
    cfg = config.load(strict=False)
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        print(f"No experiment named {args.experiment!r}. Run `make tune` first.")
        return 1

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        print("No runs found.")
        return 1
    if "status" in runs:
        runs = runs[runs["status"] == "FINISHED"].copy()
    if runs.empty:
        print("No finished runs found.")
        return 1
    metric_col = f"metrics.{args.metric}"
    if metric_col not in runs:
        print(f"No metric {args.metric!r} found in experiment {args.experiment!r}.")
        return 1

    runs = runs.dropna(subset=[metric_col]).copy()
    cost_col = "metrics.cost_thb"
    if cost_col not in runs:
        runs[cost_col] = 0.0
    runs[cost_col] = runs[cost_col].fillna(0.0)
    baseline = float(runs[metric_col].min())
    gain = (runs[metric_col] - baseline).clip(lower=1e-9)
    runs["thb_per_point"] = runs[cost_col] / (gain * 100.0)

    if args.chosen_run:
        matches = runs[runs["run_id"].str.startswith(args.chosen_run)]
        if matches.empty:
            print(f"Chosen run {args.chosen_run!r} was not found.")
            return 1
        chosen = matches.iloc[0]
    else:
        highest = float(runs[metric_col].max())
        # Treat scores within 0.5 percentage points as practically tied, then choose the
        # cheapest one. The seed report below makes that tolerance visible to the reader.
        eligible = runs[runs[metric_col] >= highest - 0.005]
        chosen = eligible.sort_values(["thb_per_point", metric_col], ascending=[True, False]).iloc[0]

    table = pd.DataFrame({
        "run_id": runs["run_id"].str[:8],
        args.metric: runs[metric_col].round(4),
        "cost_thb": runs[cost_col].round(4),
        "thb_per_point": runs["thb_per_point"].round(4),
        "n_estimators": runs.get("params.n_estimators"),
        "max_depth": runs.get("params.max_depth"),
        "min_samples_leaf": runs.get("params.min_samples_leaf"),
        "seed": runs.get("params.seed"),
        "training_job_id": runs.get("tags.training_job_id"),
    }).sort_values(args.metric, ascending=False)

    seed_count, seed_std, seed_min, seed_max = _seed_stats(args.seed_experiment, chosen)
    chosen_cost = _value(chosen, cost_col)
    monthly_cost = chosen_cost * args.monthly_trials
    highest_run = runs.loc[runs[metric_col].idxmax()]
    highest_id = str(highest_run["run_id"])
    chosen_id = str(chosen["run_id"])
    if chosen_id == highest_id:
        comparison = "The selected run is also the highest-scoring run, but its margin is shown against the seed variance rather than treated as proof by itself."
    else:
        comparison = f"The highest-scoring run is {highest_id[:8]} at {float(highest_run[metric_col]):.4f}; the selected run is within the practical tie band and costs less per percentage point."

    if seed_count:
        variance = f"Five-seed variance for this configuration is std={seed_std:.4f}, range {seed_min:.4f}–{seed_max:.4f}."
    else:
        variance = "A five-seed variance study is not available yet; run the seed-variance command before submission."

    justification = (
        f"I selected run {chosen_id[:8]} ({chosen.get('params.n_estimators')} trees, "
        f"depth {chosen.get('params.max_depth')}, leaf {chosen.get('params.min_samples_leaf')}) "
        f"with validation ROC-AUC {float(chosen[metric_col]):.4f}. {comparison} {variance} "
        f"Its measured or estimated training cost is {chosen_cost:.4f} THB and retraining it "
        f"{args.monthly_trials} times per month would cost about {monthly_cost:.4f} THB. "
        "This choice could be wrong if future machines operate outside the grouped hold-out "
        "distribution; the model may then rank risk poorly despite a clean offline score."
    )

    content = [
        "# Lab 2 — Run comparison",
        "",
        f"Experiment `{args.experiment}` · {len(table)} trials · total spend {runs[cost_col].sum():.4f} THB",
        "",
        "`thb_per_point` is cost per percentage point of validation ROC-AUC above the worst trial.",
        "",
        table.to_markdown(index=False),
        "",
        "## Selected model",
        "",
        f"- Full MLflow run ID: `{chosen_id}`",
        f"- Validation ROC-AUC: `{float(chosen[metric_col]):.8f}`",
        f"- Test ROC-AUC: `{_value(chosen, 'metrics.test_roc_auc'):.8f}`",
        f"- Training job: `{chosen.get('tags.training_job_id', 'unknown')}`",
        f"- Seed variance runs: `{seed_count}`",
        "",
        "## Justification (under 200 words)",
        "",
        justification,
        "",
        "## Registration and promotion",
        "",
        "Register the selected run with `scripts/register_model.py`; it writes all eight lineage fields to the model version. Promotion to staging must be performed by the model owner or release approver after checking the comparison table, seed variance, data version, image digest, and reload result. The approver should reject a version with missing lineage, a failed reload, or a cost outside the budget.",
        "",
        "## Cost note",
        "",
        f"The study budget is 150 THB. The reported total is {runs[cost_col].sum():.4f} THB. Low-priority quota was unavailable in this Azure subscription, so the study used dedicated scale-to-zero compute. Reconcile the provider-duration cost with the Azure billing portal before submission.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(table)} trials)")
    print(table.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
