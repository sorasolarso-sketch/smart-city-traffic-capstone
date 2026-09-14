"""
Capstone Part 3, Task 3 — Deep learning with explainability.

An LSTM is trained to predict the next hour's traffic volume from the
preceding 24 hours. The data is naturally sequential, which is why the
brief recommends this architecture.

Explainability follows the allowance the brief makes: an LSTM over a
24-step multivariate window does not decompose into per-feature
attributions that a stakeholder can act on, so SHAP is applied to a
Random Forest trained on the same prediction problem with the same
information. Section "Why SHAP is applied to a surrogate" in the final
report sets out the reasoning and its limits.

Usage
-----
    python part3_machine_learning/src/deep_learning.py
    python part3_machine_learning/src/deep_learning.py --epochs 5   # quick run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # silence TF's C++ chatter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent.parent / "part2_python"))

import common  # noqa: E402
from logging_config import configure_logging, log_stage_banner  # noqa: E402
import viz_style  # noqa: E402

logger = logging.getLogger(__name__)

LOG_FILE = common.PART3_DIR / "logs" / "part3.log"
EXPERIMENT = "traffic-lstm-demand-forecast"

SEQUENCE_LENGTH = 24          # one full day of history
SEQUENCE_FEATURES = [
    "traffic_volume", "temp", "weather_severity", "is_weekend",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "is_precipitating", "is_holiday",
]
TARGET = "traffic_volume"


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def build_sequences(
    df: pd.DataFrame, length: int = SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, timestamps) from contiguous runs of hourly readings.

    This is the step where the dataset's coverage gaps matter most. The
    sensor was offline for months at a time, so consecutive ROWS are often
    not consecutive HOURS. Sliding a window over the row index regardless
    would manufacture sequences that jump across a six-month outage and
    present them to the model as if they were a continuous day. The data is
    therefore split into unbroken hourly runs first, and windows are only
    taken within a run.
    """
    df = df.sort_values("date_time").reset_index(drop=True)

    # Identify contiguous runs: a new run starts wherever the gap != 1 hour.
    gaps = df["date_time"].diff() != pd.Timedelta(hours=1)
    run_id = gaps.cumsum()
    run_sizes = run_id.value_counts()
    usable_runs = run_sizes[run_sizes > length]

    logger.info(
        "Sequence construction — %s contiguous run(s) of hourly data; %s run(s) "
        "are longer than the %s-hour window",
        f"{len(run_sizes):,}", f"{len(usable_runs):,}", length,
    )
    logger.debug("Longest run: %s hours; median run: %s hours",
                 int(run_sizes.max()), int(run_sizes.median()))

    discarded = int(run_sizes[run_sizes <= length].sum())
    if discarded > 0:
        logger.warning(
            "Dropped %s row(s) sitting in runs shorter than the %s-hour window; "
            "no valid sequence can be formed from them", f"{discarded:,}", length,
        )

    X_parts, y_parts, stamp_parts = [], [], []
    for rid in usable_runs.index:
        run = df[run_id == rid]
        values = run[SEQUENCE_FEATURES].to_numpy(dtype=np.float32)
        targets = run[TARGET].to_numpy(dtype=np.float32)
        stamps = run["date_time"].to_numpy()
        for start in range(len(run) - length):
            X_parts.append(values[start:start + length])
            y_parts.append(targets[start + length])
            stamp_parts.append(stamps[start + length])

    X = np.asarray(X_parts, dtype=np.float32)
    y = np.asarray(y_parts, dtype=np.float32)
    stamps = np.asarray(stamp_parts)

    # Runs were visited in size order, so the sequences are not yet in time
    # order. Sort them, otherwise the "chronological" split below would mix
    # periods and the LSTM would be evaluated on hours it had trained beside.
    order = np.argsort(stamps)
    X, y, stamps = X[order], y[order], stamps[order]
    logger.debug("Sequences sorted chronologically: %s to %s",
                 pd.Timestamp(stamps[0]), pd.Timestamp(stamps[-1]))

    logger.info("Built %s sequence(s) of shape (%s, %s) from %s row(s)",
                f"{len(X):,}", length, len(SEQUENCE_FEATURES), f"{len(df):,}")
    coverage = 100 * len(X) / max(len(df) - length, 1)
    if coverage < 90:
        logger.warning(
            "Only %.1f%% of rows yielded a usable sequence, because of the "
            "sensor outages. The LSTM therefore trains on less data than the "
            "tabular models, which is a fair-comparison caveat.", coverage,
        )
    return X, y, stamps


def scale_sequences(X_train, X_test, y_train, y_test):
    """Scale inputs and target, fitting only on the training portion.

    Fitting the scaler on the full dataset would let the test period's mean
    and variance influence the training inputs. It is a small leak, but it
    is a leak, and it is easy to avoid.
    """
    n_features = X_train.shape[2]
    x_scaler = StandardScaler().fit(X_train.reshape(-1, n_features))
    X_train_s = x_scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_test_s = x_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    logger.debug("Input scaler means: %s", x_scaler.mean_.round(3).tolist())
    logger.debug("Target scaler — mean %.2f, scale %.2f",
                 y_scaler.mean_[0], y_scaler.scale_[0])
    return X_train_s, X_test_s, y_train_s, y_test_s, x_scaler, y_scaler


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
def build_lstm(input_shape: tuple[int, int]):
    import tensorflow as tf
    from tensorflow import keras

    # Seeding alone is not enough for a reproducible LSTM on CPU: TensorFlow
    # parallelises reductions across threads, and the order in which partial
    # sums land changes the low-order bits, which compound over 40 epochs
    # into a visibly different model. Op determinism forces a fixed order.
    keras.utils.set_random_seed(common.RANDOM_STATE)
    tf.config.experimental.enable_op_determinism()
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    total_params = model.count_params()
    logger.info("LSTM built — 64 -> 32 units, dropout 0.2, %s trainable parameters",
                f"{total_params:,}")
    return model


def train_lstm(df: pd.DataFrame, epochs: int, mlflow_module) -> dict:
    log_stage_banner(logger, "Task 3 — LSTM demand forecasting")

    import tensorflow as tf
    from tensorflow import keras

    X, y, stamps = build_sequences(df)
    cutoff = int(len(X) * (1 - common.TEST_FRACTION))
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]
    stamps_test = stamps[cutoff:]
    logger.info("Chronological sequence split — train %s, test %s (test begins %s)",
                f"{len(X_train):,}", f"{len(X_test):,}",
                pd.Timestamp(stamps_test[0]).date())

    X_train_s, X_test_s, y_train_s, y_test_s, _, y_scaler = scale_sequences(
        X_train, X_test, y_train, y_test)

    model = build_lstm((X_train_s.shape[1], X_train_s.shape[2]))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, verbose=0),
    ]

    started = time.time()
    history = model.fit(
        X_train_s, y_train_s,
        validation_split=0.15,
        epochs=epochs, batch_size=128,
        callbacks=callbacks, verbose=0, shuffle=False,
    )
    train_seconds = time.time() - started
    epochs_run = len(history.history["loss"])
    logger.info("LSTM trained for %s epoch(s) in %.1f seconds (early stopping "
                "patience 5)", epochs_run, train_seconds)
    for epoch in range(epochs_run):
        logger.debug("epoch %02d — loss %.5f, val_loss %.5f, mae %.5f",
                     epoch + 1, history.history["loss"][epoch],
                     history.history["val_loss"][epoch],
                     history.history["mae"][epoch])

    y_pred_s = model.predict(X_test_s, verbose=0).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "mae_as_pct_of_mean": float(100 * mean_absolute_error(y_test, y_pred)
                                    / y_test.mean()),
        "epochs_run": int(epochs_run),
        "training_seconds": float(train_seconds),
        "trainable_parameters": int(model.count_params()),
        "n_train_sequences": int(len(X_train)),
        "n_test_sequences": int(len(X_test)),
    }
    logger.info("LSTM test performance — MAE %.1f vehicles (%.1f%% of mean), "
                "RMSE %.1f, R2 %.4f", metrics["mae"],
                metrics["mae_as_pct_of_mean"], metrics["rmse"], metrics["r2"])

    # A persistence baseline: predict that this hour equals the previous
    # hour. Any sequence model that cannot beat this has learned nothing
    # beyond the autocorrelation already sitting in the data.
    naive = X_test[:, -1, SEQUENCE_FEATURES.index("traffic_volume")]
    naive_mae = float(mean_absolute_error(y_test, naive))
    metrics["persistence_baseline_mae"] = naive_mae
    metrics["improvement_over_persistence"] = naive_mae - metrics["mae"]
    logger.info("Persistence baseline (predict previous hour) — MAE %.1f. "
                "LSTM improves on it by %.1f vehicles (%.1f%%).",
                naive_mae, metrics["improvement_over_persistence"],
                100 * metrics["improvement_over_persistence"] / naive_mae)
    if metrics["mae"] > naive_mae:
        logger.warning(
            "The LSTM does NOT beat the persistence baseline. Reported as-is "
            "rather than tuned until it does."
        )

    with mlflow_module.start_run(run_name="lstm_traffic_forecast"):
        mlflow_module.set_tag("task", "deep_learning")
        mlflow_module.set_tag("architecture", "LSTM 64-32")
        mlflow_module.set_tag("split", "chronological")
        mlflow_module.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow_module.log_param("n_sequence_features", len(SEQUENCE_FEATURES))
        mlflow_module.log_param("sequence_features", ",".join(SEQUENCE_FEATURES))
        mlflow_module.log_param("max_epochs", epochs)
        mlflow_module.log_param("batch_size", 128)
        mlflow_module.log_param("optimizer", "adam")
        mlflow_module.log_param("learning_rate", 1e-3)
        for key, value in metrics.items():
            mlflow_module.log_metric(key, value)
        for epoch in range(epochs_run):
            mlflow_module.log_metric("train_loss", history.history["loss"][epoch],
                                     step=epoch)
            mlflow_module.log_metric("val_loss", history.history["val_loss"][epoch],
                                     step=epoch)
        try:
            import mlflow.tensorflow
            info = mlflow.tensorflow.log_model(model, name="model")
            mlflow_module.set_tag("model_uri", info.model_uri)
        except (ImportError, ValueError, OSError) as exc:
            logger.warning("Could not log the Keras model to MLflow (%s); the "
                           ".keras file on disk remains the artefact of record", exc)

    model_path = common.MODEL_DIR / "lstm_traffic_forecast.keras"
    model.save(model_path)
    logger.info("LSTM saved to %s", model_path)

    plot_lstm_history(history)
    plot_lstm_predictions(stamps_test, y_test, y_pred)

    return {
        "metrics": metrics,
        "model_path": str(model_path.relative_to(common.REPO_ROOT)),
        "sequence_features": SEQUENCE_FEATURES,
        "history": {
            "loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
        },
    }


def plot_lstm_history(history) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    epochs = range(1, len(history.history["loss"]) + 1)
    ax.plot(epochs, history.history["loss"], color=viz_style.SERIES[0],
            label="Training loss")
    ax.plot(epochs, history.history["val_loss"], color=viz_style.SERIES[1],
            label="Validation loss")
    ax.set_title("LSTM training converges without overfitting")
    viz_style.add_subtitle(ax, "Mean squared error on scaled traffic volume")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    path = common.FIGURE_DIR / "13_lstm_training_history.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Figure saved: %s", path)


def plot_lstm_predictions(stamps, y_true, y_pred, window: int = 336) -> None:
    n = min(window, len(y_true))
    times = pd.to_datetime(stamps[:n])
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(times, y_true[:n], color=viz_style.SERIES[0], linewidth=1.8,
            label="Actual")
    ax.plot(times, y_pred[:n], color=viz_style.SERIES[1], linewidth=1.8,
            label="LSTM prediction", alpha=0.9)
    ax.set_title("LSTM tracks the daily cycle closely on unseen data")
    viz_style.add_subtitle(
        ax, f"First {n} hours of the held-out test period, never seen during training")
    ax.set_ylabel("Vehicles per hour")
    ax.legend(loc="upper right")
    path = common.FIGURE_DIR / "14_lstm_predictions.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Figure saved: %s", path)


# ---------------------------------------------------------------------------
# SHAP explainability on a comparable tree model
# ---------------------------------------------------------------------------
def run_shap(df: pd.DataFrame) -> dict:
    log_stage_banner(logger, "Task 3 — SHAP explainability")

    try:
        import shap
    except ImportError:
        logger.error("shap is not installed; run pip install shap", exc_info=True)
        raise

    features = common.get_feature_columns(df)
    train, test = common.chronological_split(df)

    logger.info(
        "Applying SHAP to a Random Forest trained on the same target as the "
        "LSTM. The brief permits this substitution; the reasoning is that an "
        "LSTM attributes across 24 timesteps x %s features, which does not "
        "reduce to the per-feature story a mobility team needs.",
        len(SEQUENCE_FEATURES),
    )

    surrogate = RandomForestRegressor(
        n_estimators=150, max_depth=16, min_samples_leaf=5,
        n_jobs=-1, random_state=common.RANDOM_STATE,
    )
    surrogate.fit(train[features], train[TARGET])
    pred = surrogate.predict(test[features])
    surrogate_metrics = {
        "mae": float(mean_absolute_error(test[TARGET], pred)),
        "r2": float(r2_score(test[TARGET], pred)),
    }
    logger.info("Surrogate Random Forest — MAE %.1f, R2 %.4f",
                surrogate_metrics["mae"], surrogate_metrics["r2"])

    # SHAP on a 150-tree forest over 8,000 rows is slow; a sample is
    # sufficient for stable global attributions.
    sample = test[features].sample(n=min(1200, len(test)),
                                   random_state=common.RANDOM_STATE)
    logger.info("Computing SHAP values on a %s-row sample", f"{len(sample):,}")
    started = time.time()
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(sample, check_additivity=False)
    logger.info("SHAP values computed in %.1f seconds", time.time() - started)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=features).sort_values(ascending=False)

    logger.info("Top 10 features by mean |SHAP value|:")
    for name, value in importance.head(10).items():
        logger.info("  %-28s %8.1f vehicles", name, value)

    plot_shap_summary(shap_values, sample, features)
    plot_shap_bar(importance)

    return {
        "surrogate_model": "RandomForestRegressor(n_estimators=150, max_depth=16)",
        "surrogate_metrics": surrogate_metrics,
        "sample_size": int(len(sample)),
        "expected_value": float(np.ravel(explainer.expected_value)[0]),
        "mean_abs_shap": {k: float(v) for k, v in importance.round(2).items()},
        "top_10": {k: float(v) for k, v in importance.head(10).round(2).items()},
    }


def plot_shap_summary(shap_values, sample, features) -> None:
    import shap
    fig = plt.figure(figsize=(9, 6.5))
    shap.summary_plot(shap_values, sample, feature_names=features,
                      max_display=15, show=False, plot_size=None)
    plt.title("SHAP summary — what drives predicted traffic volume",
              loc="left", fontsize=12.5, fontweight="bold", pad=14)
    path = common.FIGURE_DIR / "15_shap_summary.png"
    plt.savefig(path, bbox_inches="tight", dpi=viz_style.DPI,
                facecolor=viz_style.SURFACE)
    plt.close(fig)
    logger.info("Figure saved: %s", path)


def plot_shap_bar(importance: pd.Series) -> None:
    top = importance.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values, color=viz_style.SERIES[0], height=0.7)
    for i, value in enumerate(top.values):
        ax.text(value + importance.max() * 0.012, i, f"{value:,.0f}",
                va="center", fontsize=8.5, color=viz_style.TEXT_SECONDARY)
    ax.set_title("Hour of day dominates every other predictor")
    viz_style.add_subtitle(ax, "Mean absolute SHAP value, in vehicles per hour")
    ax.set_xlabel("Mean |SHAP value| (vehicles per hour)")
    ax.set_xlim(0, importance.max() * 1.15)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    path = common.FIGURE_DIR / "16_shap_feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Figure saved: %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the LSTM and produce SHAP explanations."
    )
    parser.add_argument("--epochs", type=int, default=40,
                        help="maximum training epochs (early stopping applies)")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(log_file=LOG_FILE, debug=args.debug)
    viz_style.apply_style()
    common.ensure_directories()

    logger.info("#" * 72)
    logger.info("Capstone Part 3, Task 3 — deep learning and explainability")
    logger.info("#" * 72)

    try:
        import mlflow
        common.setup_mlflow(mlflow)
        common.set_experiment(mlflow, EXPERIMENT)

        df = common.load_features()
        df = common.add_proxy_risk_label(df)

        lstm_results = train_lstm(df, args.epochs, mlflow)
        shap_results = {} if args.skip_shap else run_shap(df)

        results = {"lstm": lstm_results, "shap": shap_results}
        out_path = common.OUTPUT_DIR / "deep_learning_results.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Results written to %s", out_path)

    except (FileNotFoundError, PermissionError, OSError):
        logger.error("Deep learning stage aborted — file access problem", exc_info=True)
        return 3
    except (ValueError, KeyError, TypeError):
        logger.error("Deep learning stage aborted — data or modelling problem",
                     exc_info=True)
        return 5
    except ImportError:
        logger.error("Deep learning stage aborted — missing library", exc_info=True)
        return 6

    logger.info("Deep learning stage finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
