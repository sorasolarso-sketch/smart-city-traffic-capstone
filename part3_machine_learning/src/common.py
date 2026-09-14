"""
Shared utilities for Part 3 — feature sets, splitting, and the proxy label.

Everything in this module exists to keep one decision in one place, because
the same decisions have to hold across six separate scripts:

* which columns may be used as features (leakage control)
* how the train/test split is made (chronological, not random)
* how the proxy accident-risk label is constructed
* where MLflow writes its runs
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PART3_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PART3_DIR.parent
FEATURES_CSV = REPO_ROOT / "data" / "processed" / "traffic_features.csv"
MODEL_DIR = PART3_DIR / "models"
FIGURE_DIR = PART3_DIR / "figures"
OUTPUT_DIR = PART3_DIR / "outputs"
MLRUNS_DIR = PART3_DIR / "mlruns"

RANDOM_STATE = 42
TEST_FRACTION = 0.2

# Weather categories the Part 3 brief treats as severe for the proxy label.
SEVERE_WEATHER = ["Rain", "Snow", "Thunderstorm", "Squall", "Drizzle"]


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# Time features, including the cyclical encodings the brief requires.
TIME_FEATURES = [
    "hour", "day_of_week", "is_weekend", "month", "day_of_year",
    "is_morning_rush", "is_evening_rush", "is_rush_hour",
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
]

# Weather encodings and derived indicators.
WEATHER_FEATURES = [
    "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_severity", "total_precipitation", "adverse_conditions_score",
    "is_severe_weather", "is_low_visibility", "is_precipitating",
    "is_raining", "is_snowing", "is_freezing", "is_extreme_cold",
    "is_hot", "is_overcast",
]

HOLIDAY_FEATURES = ["is_holiday"]

# Columns that must NEVER be used as predictors.
#
# Each of these is derived from traffic_volume, so including any of them
# would leak the answer. This matters twice over for the classification
# task: the proxy risk label is itself a deterministic function of
# congestion_category and the weather columns, so feeding the model
# congestion_category would let it reconstruct the label exactly and score
# a meaningless 100%.
LEAKY_COLUMNS = [
    "traffic_volume", "congestion_category", "congestion_level",
    "is_congested", "high_risk",
]

# 'year' is deliberately excluded. The split below is chronological, so the
# test period contains a year the model never saw in training. A tree model
# cannot extrapolate beyond the range of a feature it was trained on, and a
# linear model would extrapolate a spurious trend, so including 'year'
# would degrade honest forward-looking performance.
EXCLUDED_BY_DESIGN = ["year", "week_of_year", "quarter", "temp_celsius"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Assemble the common feature set used by every Part 3 model."""
    # One-hot weather columns produced by Part 2 (weather_Clear, weather_Rain,
    # ...). The scaled copies (weather_severity_zscore / _minmax) also start
    # with "weather_" and must be excluded: they duplicate weather_severity.
    weather_dummies = sorted(
        c for c in df.columns
        if c.startswith("weather_")
        and c not in {"weather_main", "weather_description", "weather_severity"}
        and not c.endswith(("_zscore", "_minmax"))
    )
    features = TIME_FEATURES + WEATHER_FEATURES + HOLIDAY_FEATURES + weather_dummies

    present = [c for c in features if c in df.columns]
    absent = [c for c in features if c not in df.columns]
    if absent:
        logger.warning(
            "%s expected feature(s) not present in the dataset and will be "
            "skipped: %s", len(absent), absent,
        )

    leaked = [c for c in present if c in LEAKY_COLUMNS]
    if leaked:
        logger.error(
            "Leakage guard triggered — target-derived column(s) found in the "
            "feature set: %s. Removing them.", leaked,
        )
        present = [c for c in present if c not in LEAKY_COLUMNS]

    logger.info("Feature set assembled: %s features", len(present))
    logger.debug("Features: %s", present)
    return present


# ---------------------------------------------------------------------------
# Data loading and target construction
# ---------------------------------------------------------------------------
def load_features(path: Path = FEATURES_CSV) -> pd.DataFrame:
    """Load the Part 2 feature table."""
    try:
        df = pd.read_csv(path, parse_dates=["date_time"], keep_default_na=False)
    except FileNotFoundError:
        logger.error(
            "Feature file not found at %s. Run part2_python/pipeline.py and "
            "part2_python/feature_engineering.py first.", path, exc_info=True,
        )
        raise
    except (OSError, ValueError, pd.errors.ParserError):
        logger.error("Could not read the feature file at %s", path, exc_info=True)
        raise

    df = df.sort_values("date_time").reset_index(drop=True)
    logger.info("Feature data loaded — %s rows, %s columns, %s to %s",
                f"{len(df):,}", df.shape[1],
                df["date_time"].min().date(), df["date_time"].max().date())
    return df


def add_proxy_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """Create the proxy accident-risk label defined in the Part 3 brief.

    No accident dataset was supplied with this capstone. The brief therefore
    specifies a proxy: an hour is labelled high risk when High or Severe
    congestion coincides with severe or low-visibility weather.

    This is a construct, not a measurement. It contains no information about
    any actual collision, and a model trained on it predicts the definition
    rather than the phenomenon. The limitation is analysed at length in the
    bias and fairness report; it is restated here so that nobody reading
    only the code mistakes the output for an accident forecast.
    """
    out = df.copy()

    if "congestion_category" not in out.columns:
        q1, q2, q3 = out["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
        logger.debug("Quartile thresholds: Q1=%.1f Q2=%.1f Q3=%.1f", q1, q2, q3)

        def bucket(v: float) -> str:
            if v <= q1:
                return "Low"
            if v <= q2:
                return "Medium"
            if v <= q3:
                return "High"
            return "Severe"

        out["congestion_category"] = out["traffic_volume"].apply(bucket)

    high_congestion = out["congestion_category"].isin(["High", "Severe"])
    risky_weather = (
        out["weather_main"].isin(SEVERE_WEATHER) | (out["is_low_visibility"] == 1)
    )
    out["high_risk"] = (high_congestion & risky_weather).astype(int)

    positives = int(out["high_risk"].sum())
    logger.info(
        "Proxy accident-risk label created — %s high-risk hour(s) of %s "
        "(%.2f%% positive class)", f"{positives:,}", f"{len(out):,}",
        100 * out["high_risk"].mean(),
    )
    logger.debug(
        "Label components — high/severe congestion: %s hours (%.1f%%), "
        "risky weather: %s hours (%.1f%%)",
        f"{int(high_congestion.sum()):,}", 100 * high_congestion.mean(),
        f"{int(risky_weather.sum()):,}", 100 * risky_weather.mean(),
    )
    if positives < 100:
        logger.warning(
            "Only %s positive example(s); classification metrics will be "
            "unstable", positives,
        )
    return out


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def chronological_split(
    df: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: the earliest rows train, the latest rows test.

    A random split would place the 15:00 and 17:00 readings from the same
    afternoon on opposite sides of the split. Because traffic is strongly
    autocorrelated from one hour to the next, the model could then
    effectively read the answer off its neighbours, and reported accuracy
    would be far higher than anything achievable in deployment, where the
    future is genuinely unseen. Every headline metric in Part 3 uses this
    chronological split; the random split is reported alongside only to
    quantify how large that optimism is.
    """
    cutoff_index = int(len(df) * (1 - test_fraction))
    train = df.iloc[:cutoff_index].copy()
    test = df.iloc[cutoff_index:].copy()

    logger.info(
        "Chronological split — train %s rows (%s to %s), test %s rows (%s to %s)",
        f"{len(train):,}", train["date_time"].min().date(), train["date_time"].max().date(),
        f"{len(test):,}", test["date_time"].min().date(), test["date_time"].max().date(),
    )
    return train, test


def summarise_target_balance(train: pd.DataFrame, test: pd.DataFrame,
                             target: str) -> None:
    """Warn when a chronological split leaves the classes badly unbalanced."""
    train_rate = train[target].mean()
    test_rate = test[target].mean()
    logger.info("Target '%s' positive rate — train %.2f%%, test %.2f%%",
                target, 100 * train_rate, 100 * test_rate)
    if test_rate == 0 or train_rate == 0:
        logger.error("Target '%s' has no positive examples in one split half", target)
    elif abs(train_rate - test_rate) > 0.05:
        logger.warning(
            "Positive rate differs by %.1f percentage points between train and "
            "test. The chronological split has produced a genuine distribution "
            "shift, which is realistic but will depress test metrics.",
            100 * abs(train_rate - test_rate),
        )


def ensure_directories() -> None:
    for directory in (MODEL_DIR, FIGURE_DIR, OUTPUT_DIR, MLRUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


MLFLOW_DB = PART3_DIR / "mlflow.db"


def mlflow_tracking_uri() -> str:
    """SQLite-backed MLflow store, kept inside the repository.

    Recent MLflow releases put the plain filesystem store into maintenance
    mode and refuse to open it. A local SQLite database is the supported
    equivalent, needs no server, and is what ``mlflow ui`` expects:

        mlflow ui --backend-store-uri sqlite:///part3_machine_learning/mlflow.db
    """
    MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{MLFLOW_DB.as_posix()}"


def setup_mlflow(mlflow_module) -> str:
    """Point MLflow at the project's tracking store and return the URI."""
    uri = mlflow_tracking_uri()
    mlflow_module.set_tracking_uri(uri)
    logger.info("MLflow tracking URI: %s", uri)
    return uri


def set_experiment(mlflow_module, name: str) -> str:
    """Select an experiment, creating it with a repo-local artifact store."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    experiment = mlflow_module.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow_module.create_experiment(
            name, artifact_location=MLRUNS_DIR.as_uri()
        )
        logger.info("Created MLflow experiment '%s' (id %s)", name, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.debug("Using existing MLflow experiment '%s' (id %s)",
                     name, experiment_id)
    mlflow_module.set_experiment(name)
    return experiment_id
