"""
Capstone Part 3, Task 6.1 and 6.2 — Model versioning and experiment summary.

* Registers the trained models in the MLflow Model Registry, so that each
  has a version number, a stage alias (``champion`` / ``challenger``) and
  a link back to the tracked run that produced it.
* Writes ``outputs/MODEL_REGISTRY.md``, a human-readable version table
  built from the tracked runs, and ``outputs/experiment_summary.json``.

Usage
-----
    python part3_machine_learning/src/model_registry.py
    mlflow ui --backend-store-uri sqlite:///part3_machine_learning/mlflow.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent.parent / "part2_python"))

import common  # noqa: E402
from logging_config import configure_logging, log_stage_banner  # noqa: E402

logger = logging.getLogger(__name__)

LOG_FILE = common.PART3_DIR / "logs" / "part3.log"

# Which registered name each run feeds, and the metric that ranks versions.
REGISTRY_PLAN = {
    "traffic-accident-risk-classification": {
        "registered_name": "traffic_risk_classifier",
        "rank_metric": "roc_auc", "higher_is_better": True,
    },
    "traffic-volume-regression": {
        "registered_name": "traffic_volume_regressor",
        "rank_metric": "mae", "higher_is_better": False,
    },
    "traffic-lstm-demand-forecast": {
        "registered_name": "traffic_lstm_forecaster",
        "rank_metric": "mae", "higher_is_better": False,
    },
}


def collect_runs(mlflow) -> pd.DataFrame:
    client = mlflow.tracking.MlflowClient()
    frames = []
    for experiment_name in REGISTRY_PLAN:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning("Experiment '%s' not found — run its training script first",
                           experiment_name)
            continue
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                                  order_by=["start_time ASC"])
        if runs.empty:
            logger.warning("Experiment '%s' has no runs", experiment_name)
            continue
        runs["experiment_name"] = experiment_name
        frames.append(runs)
        logger.info("Experiment '%s' — %s run(s)", experiment_name, len(runs))
    if not frames:
        raise FileNotFoundError("No MLflow runs found")
    return pd.concat(frames, ignore_index=True)


def register_models(mlflow, runs: pd.DataFrame) -> list[dict]:
    """Register every run's model and mark the best as champion."""
    client = mlflow.tracking.MlflowClient()
    registered = []

    for experiment_name, plan in REGISTRY_PLAN.items():
        subset = runs[runs["experiment_name"] == experiment_name]
        if subset.empty:
            continue
        metric_col = f"metrics.{plan['rank_metric']}"
        if metric_col not in subset.columns:
            logger.warning("Metric '%s' missing for %s", plan["rank_metric"], experiment_name)
            continue

        ordered = subset.sort_values(metric_col, ascending=not plan["higher_is_better"])
        name = plan["registered_name"]

        for rank, (_, run) in enumerate(ordered.iterrows(), start=1):
            run_id = run["run_id"]
            run_name = run.get("tags.mlflow.runName", run_id[:8])
            model_uri = run.get("tags.model_uri")
            alias = "champion" if rank == 1 else "challenger"
            metric_value = float(run[metric_col])

            if not isinstance(model_uri, str) or not model_uri:
                logger.warning("Run %s (%s) has no logged model URI; recording "
                               "version metadata only", run_id[:8], run_name)
                registered.append({
                    "registered_name": name, "version": None, "run_id": run_id,
                    "run_name": run_name, "alias": alias,
                    plan["rank_metric"]: metric_value,
                    "artifact": "metrics only (no logged model)",
                })
                continue

            existing = client.search_model_versions(f"run_id='{run_id}'")
            if existing:
                version = existing[0]
                logger.info("Run %s already registered as %s v%s; reusing",
                            run_id[:8], name, version.version)
            else:
                version = None
            try:
                if version is None:
                    version = mlflow.register_model(model_uri, name)
            except Exception as exc:  # noqa: BLE001 — MLflow raises a broad base class
                logger.error("Could not register %s from run %s: %s", name, run_id[:8], exc)
                continue

            client.set_registered_model_alias(name, alias, version.version)
            client.set_model_version_tag(name, version.version, "algorithm",
                                         str(run.get("params.algorithm", "")))
            client.set_model_version_tag(name, version.version, plan["rank_metric"],
                                         f"{metric_value:.4f}")
            client.update_model_version(
                name, version.version,
                description=(f"{run_name} — {plan['rank_metric']} {metric_value:.4f} "
                             "on the chronological test split"),
            )
            logger.info("Registered %s v%s from run '%s' as %s (%s=%.4f)",
                        name, version.version, run_name, alias,
                        plan["rank_metric"], metric_value)
            registered.append({
                "registered_name": name, "version": int(version.version),
                "run_id": run_id, "run_name": run_name, "alias": alias,
                plan["rank_metric"]: metric_value, "artifact": model_uri,
            })
    return registered


def write_registry_markdown(runs: pd.DataFrame, registered: list[dict], path: Path) -> None:
    lines = [
        "# Model Registry and Version History\n",
        "Every model version below is tracked in MLflow with its parameters, "
        "metrics and artefacts. Open the tracking UI with:\n",
        "```bash\nmlflow ui --backend-store-uri sqlite:///part3_machine_learning/mlflow.db\n```\n",
        "All metrics are on the **chronological** test split (the final 20% of the "
        "timeline, 2017-10-26 to 2018-09-30), which is the deployment-realistic figure.\n",
    ]

    lines += ["## Registered models\n",
              "| Registered name | Version | Alias | Source run | Key metric | Artefact |",
              "| --- | ---: | --- | --- | ---: | --- |"]
    for r in registered:
        metric_key = [k for k in r if k not in {"registered_name", "version", "run_id",
                                                 "run_name", "alias", "artifact"}][0]
        lines.append(
            f"| `{r['registered_name']}` | {r['version'] if r['version'] else '—'} | "
            f"**{r['alias']}** | `{r['run_name']}` | {metric_key} = {r[metric_key]:.4f} | "
            f"`{r['artifact']}` |"
        )

    lines.append("\n## Version-by-version performance\n")

    clf = runs[runs["experiment_name"] == "traffic-accident-risk-classification"]
    if not clf.empty:
        lines += ["### Classification — proxy accident risk\n",
                  "| Run | Accuracy | Precision | Recall | F1 | ROC AUC | AUC on random split | Train (s) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for _, r in clf.iterrows():
            lines.append(
                f"| `{r['tags.mlflow.runName']}` | {r['metrics.accuracy']:.4f} | "
                f"{r['metrics.precision']:.4f} | {r['metrics.recall']:.4f} | "
                f"{r['metrics.f1']:.4f} | **{r['metrics.roc_auc']:.4f}** | "
                f"{r['metrics.roc_auc_random_split']:.4f} | {r['metrics.training_seconds']:.1f} |"
            )

    reg = runs[runs["experiment_name"] == "traffic-volume-regression"]
    lstm = runs[runs["experiment_name"] == "traffic-lstm-demand-forecast"]
    if not reg.empty or not lstm.empty:
        lines += ["\n### Regression and forecasting — hourly traffic volume\n",
                  "| Run | MAE (vehicles) | MAE % of mean | RMSE | R² | Train (s) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for _, r in pd.concat([reg, lstm]).iterrows():
            lines.append(
                f"| `{r['tags.mlflow.runName']}` | **{r['metrics.mae']:.1f}** | "
                f"{r['metrics.mae_as_pct_of_mean']:.1f}% | {r['metrics.rmse']:.1f} | "
                f"{r['metrics.r2']:.4f} | {r['metrics.training_seconds']:.1f} |"
            )

    lines += [
        "\n## Promotion policy\n",
        "A version is promoted to **champion** when it beats the incumbent on the "
        "ranking metric on the chronological split *and* the monitoring dashboard "
        "shows no open error-drift alert for the incumbent that the challenger would "
        "inherit. Promotion is a human decision recorded as an MLflow alias change; "
        "it is never automatic.\n",
        "## Reproducibility\n",
        "Every run records its hyperparameters (`hp_*`), the feature count, the "
        "train/test row counts and the split method as MLflow parameters, and the "
        "trained estimator as an artefact. Re-running `supervised.py` and "
        "`deep_learning.py` creates new runs alongside the old ones rather than "
        "overwriting them, so the history is preserved.\n",
        "_Generated by `part3_machine_learning/src/model_registry.py`._",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Model registry document written to %s", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register models and document versions.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(log_file=LOG_FILE, debug=args.debug)
    common.ensure_directories()
    log_stage_banner(logger, "Task 6.1 / 6.2 — Model registry and experiment summary")

    try:
        import mlflow
        common.setup_mlflow(mlflow)
        runs = collect_runs(mlflow)
        registered = register_models(mlflow, runs)
        write_registry_markdown(runs, registered, common.OUTPUT_DIR / "MODEL_REGISTRY.md")

        keep = [c for c in runs.columns if c.startswith(("metrics.", "params.")) or
                c in {"run_id", "experiment_name", "tags.mlflow.runName", "start_time"}]
        summary = runs[keep].copy()
        summary["start_time"] = summary["start_time"].astype(str)
        summary.to_json(common.OUTPUT_DIR / "experiment_summary.json",
                        orient="records", indent=2)
        logger.info("Experiment summary written with %s run(s)", len(summary))
    except FileNotFoundError:
        logger.error("No tracked runs found — run supervised.py and deep_learning.py first",
                     exc_info=True)
        return 3
    except (OSError, ValueError, KeyError):
        logger.error("Registry stage failed", exc_info=True)
        return 5
    except ImportError:
        logger.error("mlflow is not installed", exc_info=True)
        return 6

    logger.info("Model registry stage finished — %s version(s) recorded", len(registered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
