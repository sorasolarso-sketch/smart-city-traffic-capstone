"""
Capstone Part 3, Task 1 — Supervised machine learning.

Two problems, two algorithms each, one shared feature set:

* Classification — proxy accident-risk label
    baseline : Logistic Regression (scaled)
    ensemble : Random Forest
* Regression — hourly traffic volume
    baseline : Linear Regression (scaled)
    ensemble : HistGradientBoosting Regressor

Every run is tracked in MLflow (Task 4 and Task 6.2).

Usage
-----
    python part3_machine_learning/src/supervised.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent.parent / "part2_python"))

import common  # noqa: E402
from logging_config import configure_logging, log_stage_banner  # noqa: E402
import viz_style  # noqa: E402

logger = logging.getLogger(__name__)

LOG_FILE = common.PART3_DIR / "logs" / "part3.log"
EXPERIMENT_CLASSIFICATION = "traffic-accident-risk-classification"
EXPERIMENT_REGRESSION = "traffic-volume-regression"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "positive_rate_actual": float(np.mean(y_true)),
        "positive_rate_predicted": float(np.mean(y_pred)),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_actual": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(y_pred)),
        "mae_as_pct_of_mean": float(
            100 * mean_absolute_error(y_true, y_pred) / np.mean(y_true)
        ),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def run_classification(train: pd.DataFrame, test: pd.DataFrame,
                       features: list[str], mlflow_module) -> dict:
    log_stage_banner(logger, "Task 1a — Classification: proxy accident risk")

    X_train, y_train = train[features], train["high_risk"]
    X_test, y_test = test[features], test["high_risk"]
    common.summarise_target_balance(train, test, "high_risk")

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=common.RANDOM_STATE,
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=18, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1,
            random_state=common.RANDOM_STATE,
        ),
    }

    # A stratified-random baseline, so that accuracy can be read against
    # something. On an imbalanced label, "always predict negative" already
    # scores well, and a headline accuracy means nothing without it.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_metrics = {
        "accuracy": float(accuracy_score(y_test, dummy.predict(X_test))),
    }
    logger.info(
        "Majority-class baseline accuracy on the test set: %.4f — any model "
        "must beat this to be useful", dummy_metrics["accuracy"],
    )

    results = {"baseline_majority_class": dummy_metrics, "models": {}}

    for name, model in models.items():
        logger.info("Training classifier: %s", name)
        started = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        train_seconds = time.time() - started

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, y_pred, y_proba)
        metrics["training_seconds"] = float(train_seconds)

        # Random-split comparison, to quantify the optimism a naive split buys.
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([y_train, y_test])
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            X_all, y_all, test_size=common.TEST_FRACTION,
            random_state=common.RANDOM_STATE, stratify=y_all,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_r = clone(model)
            model_r.fit(Xr_tr, yr_tr)
        random_auc = float(roc_auc_score(yr_te, model_r.predict_proba(Xr_te)[:, 1]))
        metrics["roc_auc_random_split"] = random_auc
        metrics["optimism_from_random_split"] = random_auc - metrics["roc_auc"]

        logger.info(
            "%s — accuracy %.4f, precision %.4f, recall %.4f, F1 %.4f, "
            "ROC AUC %.4f (random split %.4f, optimism %+.4f)",
            name, metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["roc_auc"], random_auc,
            metrics["optimism_from_random_split"],
        )
        if metrics["optimism_from_random_split"] > 0.02:
            logger.warning(
                "%s scores %.4f higher on a random split than on the "
                "chronological split. Reporting the random-split figure would "
                "overstate deployment performance.",
                name, metrics["optimism_from_random_split"],
            )

        with mlflow_module.start_run(run_name=f"clf_{name}"):
            mlflow_module.set_tag("task", "classification")
            mlflow_module.set_tag("target", "high_risk (proxy)")
            mlflow_module.set_tag("split", "chronological")
            mlflow_module.log_param("algorithm", name)
            mlflow_module.log_param("n_features", len(features))
            mlflow_module.log_param("n_train", len(X_train))
            mlflow_module.log_param("n_test", len(X_test))
            estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
            for key, value in estimator.get_params().items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    mlflow_module.log_param(f"hp_{key}", value)
            for key, value in metrics.items():
                mlflow_module.log_metric(key, value)
            info = mlflow_module.sklearn.log_model(model, name="model")
            mlflow_module.set_tag("model_uri", info.model_uri)

        path = common.MODEL_DIR / f"classifier_{name}.joblib"
        joblib.dump(model, path)
        logger.info("Model saved to %s", path)

        results["models"][name] = {
            "metrics": metrics,
            "model_path": str(path.relative_to(common.REPO_ROOT)),
        }

        if name == "random_forest":
            importances = pd.Series(model.feature_importances_, index=features)
            results["models"][name]["top_features"] = (
                importances.nlargest(15).round(5).to_dict()
            )
            plot_feature_importance(
                importances, "Random Forest — proxy risk classifier",
                "07_classification_feature_importance.png",
            )

    best = max(results["models"], key=lambda k: results["models"][k]["metrics"]["roc_auc"])
    results["best_model"] = best
    logger.info("Best classifier by ROC AUC: %s (%.4f)",
                best, results["models"][best]["metrics"]["roc_auc"])

    plot_roc_curves(models, test[features], test["high_risk"])
    return results


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def run_regression(train: pd.DataFrame, test: pd.DataFrame,
                   features: list[str], mlflow_module) -> dict:
    log_stage_banner(logger, "Task 1b — Regression: hourly traffic volume")

    X_train, y_train = train[features], train["traffic_volume"]
    X_test, y_test = test[features], test["traffic_volume"]

    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline = regression_metrics(y_test, dummy.predict(X_test))
    logger.info(
        "Mean-prediction baseline — MAE %.1f, R2 %.4f. Any model must beat this.",
        baseline["mae"], baseline["r2"],
    )

    models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.08, max_depth=None,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=common.RANDOM_STATE,
        ),
    }

    results = {"baseline_mean_prediction": baseline, "models": {}}

    for name, model in models.items():
        logger.info("Training regressor: %s", name)
        started = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - started

        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        metrics["training_seconds"] = float(train_seconds)
        metrics["improvement_over_baseline_mae"] = baseline["mae"] - metrics["mae"]

        logger.info(
            "%s — MAE %.1f vehicles (%.1f%% of mean), RMSE %.1f, R2 %.4f",
            name, metrics["mae"], metrics["mae_as_pct_of_mean"],
            metrics["rmse"], metrics["r2"],
        )

        with mlflow_module.start_run(run_name=f"reg_{name}"):
            mlflow_module.set_tag("task", "regression")
            mlflow_module.set_tag("target", "traffic_volume")
            mlflow_module.set_tag("split", "chronological")
            mlflow_module.log_param("algorithm", name)
            mlflow_module.log_param("n_features", len(features))
            mlflow_module.log_param("n_train", len(X_train))
            mlflow_module.log_param("n_test", len(X_test))
            estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
            for key, value in estimator.get_params().items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    mlflow_module.log_param(f"hp_{key}", value)
            for key, value in metrics.items():
                mlflow_module.log_metric(key, value)
            info = mlflow_module.sklearn.log_model(model, name="model")
            mlflow_module.set_tag("model_uri", info.model_uri)

        path = common.MODEL_DIR / f"regressor_{name}.joblib"
        joblib.dump(model, path)
        logger.info("Model saved to %s", path)

        results["models"][name] = {
            "metrics": metrics,
            "model_path": str(path.relative_to(common.REPO_ROOT)),
        }

    best = min(results["models"], key=lambda k: results["models"][k]["metrics"]["mae"])
    results["best_model"] = best
    logger.info("Best regressor by MAE: %s (%.1f vehicles)",
                best, results["models"][best]["metrics"]["mae"])

    best_model = models[best]
    plot_regression_diagnostics(test, best_model.predict(X_test), best)

    # Residual analysis by hour, which feeds the fairness report.
    residuals = pd.DataFrame({
        "hour": test["hour"].values,
        "is_weekend": test["is_weekend"].values,
        "actual": y_test.values,
        "predicted": best_model.predict(X_test),
    })
    residuals["error"] = residuals["predicted"] - residuals["actual"]
    residuals["abs_error"] = residuals["error"].abs()
    by_hour = residuals.groupby("hour")["abs_error"].mean()
    results["mae_by_hour"] = {str(int(k)): float(v) for k, v in by_hour.items()}
    results["mae_weekday"] = float(
        residuals.loc[residuals["is_weekend"] == 0, "abs_error"].mean())
    results["mae_weekend"] = float(
        residuals.loc[residuals["is_weekend"] == 1, "abs_error"].mean())
    logger.info("Error concentration — worst hour %02d:00 (MAE %.0f), best hour "
                "%02d:00 (MAE %.0f)", int(by_hour.idxmax()), by_hour.max(),
                int(by_hour.idxmin()), by_hour.min())
    logger.warning(
        "Error is unevenly distributed across the day: MAE at %02d:00 is %.1fx "
        "the MAE at %02d:00. This is recorded in the bias and fairness report.",
        int(by_hour.idxmax()), by_hour.max() / max(by_hour.min(), 1e-9),
        int(by_hour.idxmin()),
    )
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_feature_importance(importances: pd.Series, title: str, filename: str) -> None:
    top = importances.nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values, color=viz_style.SERIES[0], height=0.7)
    ax.set_title(title)
    viz_style.add_subtitle(ax, "Top 15 features by impurity-based importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    path = common.FIGURE_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    logger.info("Figure saved: %s", path)


def plot_roc_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, model) in enumerate(models.items()):
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=viz_style.SERIES[i], linewidth=2,
                label=f"{name.replace('_', ' ').title()} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=viz_style.TEXT_MUTED, linestyle="--", linewidth=1.2,
            label="Random (AUC 0.500)")
    ax.set_title("Both classifiers separate the proxy label well")
    viz_style.add_subtitle(ax, "ROC curves on the chronological test set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(axis="both")
    path = common.FIGURE_DIR / "08_classification_roc_curves.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Figure saved: %s", path)


def plot_regression_diagnostics(test: pd.DataFrame, y_pred, model_name: str) -> None:
    actual = test["traffic_volume"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(actual, y_pred, s=6, alpha=0.2, color=viz_style.SERIES[0],
                    linewidths=0)
    lims = [0, max(actual.max(), y_pred.max()) * 1.02]
    axes[0].plot(lims, lims, color=viz_style.TEXT_MUTED, linestyle="--", linewidth=1.2)
    axes[0].set_title("Predicted versus actual")
    axes[0].set_xlabel("Actual vehicles per hour")
    axes[0].set_ylabel("Predicted vehicles per hour")
    axes[0].grid(axis="both")

    hourly = pd.DataFrame({"hour": test["hour"].values,
                           "error": y_pred - actual})
    mae_by_hour = hourly.groupby("hour")["error"].apply(lambda s: s.abs().mean())
    axes[1].bar(mae_by_hour.index, mae_by_hour.values, color=viz_style.SERIES[0],
                width=0.72)
    axes[1].set_title("Mean absolute error by hour of day")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("MAE (vehicles)")
    axes[1].set_xticks(range(0, 24, 2))

    fig.suptitle(f"Regression diagnostics — {model_name.replace('_', ' ')}",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", y=1.10)
    path = common.FIGURE_DIR / "09_regression_diagnostics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Part 3 supervised models.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(log_file=LOG_FILE, debug=args.debug)
    viz_style.apply_style()
    common.ensure_directories()

    logger.info("#" * 72)
    logger.info("Capstone Part 3, Task 1 — supervised learning")
    logger.info("#" * 72)

    try:
        import mlflow
        import mlflow.sklearn
        common.setup_mlflow(mlflow)

        df = common.load_features()
        df = common.add_proxy_risk_label(df)
        features = common.get_feature_columns(df)
        train, test = common.chronological_split(df)

        common.set_experiment(mlflow, EXPERIMENT_CLASSIFICATION)
        classification = run_classification(train, test, features, mlflow)

        common.set_experiment(mlflow, EXPERIMENT_REGRESSION)
        regression = run_regression(train, test, features, mlflow)

        results = {
            "feature_count": len(features),
            "features": features,
            "excluded_for_leakage": common.LEAKY_COLUMNS,
            "excluded_by_design": common.EXCLUDED_BY_DESIGN,
            "split": {
                "method": "chronological",
                "test_fraction": common.TEST_FRACTION,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_end": str(train["date_time"].max()),
                "test_start": str(test["date_time"].min()),
            },
            "classification": classification,
            "regression": regression,
        }
        out_path = common.OUTPUT_DIR / "supervised_results.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Results written to %s", out_path)

    except (FileNotFoundError, PermissionError, OSError):
        logger.error("Supervised stage aborted — file access problem", exc_info=True)
        return 3
    except (ValueError, KeyError, TypeError):
        logger.error("Supervised stage aborted — data or modelling problem",
                     exc_info=True)
        return 5
    except ImportError:
        logger.error("Supervised stage aborted — a required library is missing",
                     exc_info=True)
        return 6

    logger.info("Supervised learning stage finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
