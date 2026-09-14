"""
Capstone Part 2, Task 2 — Feature engineering with NumPy and Pandas.

Turns the cleaned dataset into an ML-ready feature table:

* time features, including cyclical encodings
* weather encodings and derived weather indicators
* scaled versions of the continuous variables
* a data-driven congestion target

Usage
-----
    python part2_python/feature_engineering.py
    python part2_python/feature_engineering.py --debug   # show thresholds
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logging_config import configure_logging, log_stage_banner  # noqa: E402

logger = logging.getLogger(__name__)

PART2_DIR = Path(__file__).resolve().parent
REPO_ROOT = PART2_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "traffic_clean.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "traffic_features.csv"
METADATA_PATH = REPO_ROOT / "data" / "processed" / "feature_metadata.json"

# Weather categories treated as severe for the derived risk indicators.
SEVERE_WEATHER = ["Rain", "Snow", "Thunderstorm", "Squall", "Drizzle"]

# Weather categories that materially reduce visibility.
LOW_VISIBILITY_WEATHER = ["Fog", "Mist", "Haze", "Smoke", "Snow", "Squall"]

# Ordered severity scale, 0 (benign) to 10 (most hazardous).
WEATHER_SEVERITY_SCORE = {
    "Clear": 0, "Clouds": 1, "Mist": 3, "Haze": 3, "Smoke": 4, "Fog": 5,
    "Drizzle": 4, "Rain": 6, "Snow": 8, "Squall": 9, "Thunderstorm": 9,
}

RUSH_HOURS_MORNING = (6, 9)
RUSH_HOURS_EVENING = (15, 18)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_clean_data(path: Path) -> pd.DataFrame:
    """Load the output of :mod:`pipeline`."""
    try:
        df = pd.read_csv(path, parse_dates=["date_time"], keep_default_na=False)
    except FileNotFoundError:
        logger.error(
            "Cleaned data not found at %s — run pipeline.py first", path, exc_info=True
        )
        raise
    except (OSError, ValueError, pd.errors.ParserError):
        logger.error("Could not read cleaned data from %s", path, exc_info=True)
        raise

    logger.info("Cleaned data loaded from %s — %s rows, %s columns",
                path.name, f"{len(df):,}", df.shape[1])
    return df


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar and clock features derived from the timestamp."""
    out = df.copy()
    ts = out["date_time"]

    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek           # Monday = 0
    out["day_name"] = ts.dt.day_name()
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["month"] = ts.dt.month
    out["year"] = ts.dt.year
    out["day_of_year"] = ts.dt.dayofyear
    out["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    out["quarter"] = ts.dt.quarter

    # Rush-hour flags, derived from the hourly profile observed in Part 1.
    out["is_morning_rush"] = out["hour"].between(*RUSH_HOURS_MORNING).astype(int)
    out["is_evening_rush"] = out["hour"].between(*RUSH_HOURS_EVENING).astype(int)
    out["is_rush_hour"] = (out["is_morning_rush"] | out["is_evening_rush"]).astype(int)

    # Coarse part-of-day band, useful for association rule mining in Part 3.
    conditions = [
        out["hour"].between(0, 5),
        out["hour"].between(6, 9),
        out["hour"].between(10, 14),
        out["hour"].between(15, 18),
        out["hour"].between(19, 23),
    ]
    labels = ["Night", "Morning Peak", "Midday", "Evening Peak", "Evening"]
    out["part_of_day"] = np.select(conditions, labels, default="Unknown")

    # Season, using the meteorological definition for the northern hemisphere.
    out["season"] = np.select(
        [
            out["month"].isin([12, 1, 2]),
            out["month"].isin([3, 4, 5]),
            out["month"].isin([6, 7, 8]),
            out["month"].isin([9, 10, 11]),
        ],
        ["Winter", "Spring", "Summer", "Autumn"],
        default="Unknown",
    )

    logger.info(
        "Time features added: hour, day_of_week, is_weekend, month, year, "
        "day_of_year, week_of_year, quarter, rush-hour flags, part_of_day, season"
    )
    logger.debug("Weekend hours: %s of %s (%.1f%%)",
                 f"{int(out['is_weekend'].sum()):,}", f"{len(out):,}",
                 100 * out["is_weekend"].mean())
    logger.debug("Rush hours: %s (%.1f%%)", f"{int(out['is_rush_hour'].sum()):,}",
                 100 * out["is_rush_hour"].mean())
    logger.debug("Part-of-day distribution: %s", out["part_of_day"].value_counts().to_dict())
    return out


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sine/cosine encodings for the periodic time variables.

    A model given ``hour`` as a plain integer sees 23 and 0 as maximally
    distant, when in fact they are adjacent. Projecting each period onto a
    circle restores that adjacency: for any period ``T``,

        x_sin = sin(2 * pi * value / T)
        x_cos = cos(2 * pi * value / T)

    Both components are required — either one alone is ambiguous, because
    a sine value maps to two distinct points on the circle.
    """
    out = df.copy()

    for column, period in [("hour", 24), ("day_of_week", 7), ("month", 12),
                           ("day_of_year", 365.25)]:
        radians = 2 * np.pi * out[column] / period
        out[f"{column}_sin"] = np.sin(radians)
        out[f"{column}_cos"] = np.cos(radians)
        logger.debug(
            "Cyclical encoding for '%s' (period %s): sin range [%.3f, %.3f], "
            "cos range [%.3f, %.3f]", column, period,
            out[f"{column}_sin"].min(), out[f"{column}_sin"].max(),
            out[f"{column}_cos"].min(), out[f"{column}_cos"].max(),
        )

    # Verification: sin^2 + cos^2 must equal 1 for every row on the unit circle.
    unit_check = float((out["hour_sin"] ** 2 + out["hour_cos"] ** 2).round(9).nunique())
    if unit_check != 1.0:
        logger.warning(
            "Cyclical hour encoding does not lie exactly on the unit circle; "
            "%s distinct radii found", unit_check,
        )
    else:
        logger.debug("Cyclical encoding verified: sin^2 + cos^2 = 1 for all rows")

    logger.info(
        "Cyclical encodings added for hour (period 24), day_of_week (7), "
        "month (12) and day_of_year (365.25) — 8 new columns"
    )
    return out


# ---------------------------------------------------------------------------
# Weather features
# ---------------------------------------------------------------------------
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encoded and derived weather variables."""
    out = df.copy()

    out["temp_celsius"] = (out["temp"] - 273.15).round(2)

    # Ordinal severity, retaining the natural ordering that one-hot loses.
    out["weather_severity"] = out["weather_main"].map(WEATHER_SEVERITY_SCORE)
    unmapped = int(out["weather_severity"].isna().sum())
    if unmapped > 0:
        logger.warning(
            "%s row(s) carry a weather category absent from the severity "
            "scale and were scored 0: %s", unmapped,
            out.loc[out["weather_severity"].isna(), "weather_main"].unique().tolist(),
        )
        out["weather_severity"] = out["weather_severity"].fillna(0)
    out["weather_severity"] = out["weather_severity"].astype(int)

    # Binary indicators used directly by the Part 3 proxy risk label.
    out["is_severe_weather"] = out["weather_main"].isin(SEVERE_WEATHER).astype(int)
    out["is_low_visibility"] = out["weather_main"].isin(LOW_VISIBILITY_WEATHER).astype(int)
    out["is_precipitating"] = ((out["rain_1h"] > 0) | (out["snow_1h"] > 0)).astype(int)
    out["is_raining"] = (out["rain_1h"] > 0).astype(int)
    out["is_snowing"] = (out["snow_1h"] > 0).astype(int)
    out["is_freezing"] = (out["temp"] < 273.15).astype(int)
    out["is_extreme_cold"] = (out["temp"] < 258.15).astype(int)   # below -15 C
    out["is_hot"] = (out["temp"] > 300.0).astype(int)             # above ~27 C
    out["is_overcast"] = (out["clouds_all"] >= 75).astype(int)
    out["total_precipitation"] = out["rain_1h"] + out["snow_1h"]

    # A compact composite: how adverse is driving in this hour overall?
    out["adverse_conditions_score"] = (
        out["weather_severity"]
        + 2 * out["is_low_visibility"]
        + 2 * out["is_freezing"]
        + 1 * out["is_precipitating"]
    )

    # One-hot encoding of the weather category for linear models.
    dummies = pd.get_dummies(out["weather_main"], prefix="weather", dtype=int)
    out = pd.concat([out, dummies], axis=1)

    logger.info(
        "Weather features added: temp_celsius, weather_severity (0-10 ordinal), "
        "6 binary condition indicators, total_precipitation, "
        "adverse_conditions_score, and %s one-hot weather columns",
        dummies.shape[1],
    )
    logger.debug("One-hot weather columns: %s", list(dummies.columns))
    logger.debug(
        "Indicator prevalence — severe %.1f%%, low visibility %.1f%%, "
        "precipitating %.1f%%, freezing %.1f%%",
        100 * out["is_severe_weather"].mean(), 100 * out["is_low_visibility"].mean(),
        100 * out["is_precipitating"].mean(), 100 * out["is_freezing"].mean(),
    )
    return out


# ---------------------------------------------------------------------------
# Scaled numerical features
# ---------------------------------------------------------------------------
def add_scaled_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardised and min-max scaled versions of the continuous variables.

    Both scalings are produced because the downstream models want different
    things: distance-based methods such as K-means and any regularised
    linear model need z-scores, while a neural network trains more stably on
    inputs bounded to [0, 1].

    The target, ``traffic_volume``, is deliberately left unscaled. Scaling
    it here and then fitting on the full dataset would leak information from
    the test period into training.
    """
    out = df.copy()
    params: dict[str, dict[str, float]] = {}

    for column in ["temp", "clouds_all", "total_precipitation", "weather_severity"]:
        values = out[column].astype(float)

        mean, std = float(values.mean()), float(values.std(ddof=0))
        if std == 0:
            logger.warning(
                "Column '%s' has zero variance; standardised values set to 0",
                column,
            )
            out[f"{column}_zscore"] = 0.0
        else:
            out[f"{column}_zscore"] = (values - mean) / std

        lo, hi = float(values.min()), float(values.max())
        if hi == lo:
            logger.warning(
                "Column '%s' is constant; min-max values set to 0", column
            )
            out[f"{column}_minmax"] = 0.0
        else:
            out[f"{column}_minmax"] = (values - lo) / (hi - lo)

        params[column] = {"mean": mean, "std": std, "min": lo, "max": hi}
        logger.debug(
            "Scaling '%s': mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            column, mean, std, lo, hi,
        )

    logger.info(
        "Scaled features added: z-score and min-max versions of %s continuous "
        "variables (%s new columns). traffic_volume left unscaled as the target.",
        len(params), 2 * len(params),
    )
    return out, params


# ---------------------------------------------------------------------------
# Target variables
# ---------------------------------------------------------------------------
def add_congestion_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Data-driven congestion category, plus the fixed-threshold flag.

    Two targets are produced, and they are not interchangeable.

    ``congestion_category`` follows the Part 3 brief exactly: the quartiles
    of ``traffic_volume`` split the data into Low / Medium / High / Severe,
    each holding roughly 25% of hours. Because the cut-points come from the
    data rather than from a fixed number, the definition travels to another
    corridor without needing to be re-tuned, and the classes stay balanced —
    which matters, because a classifier trained on a badly skewed target
    learns to predict the majority class and little else.

    ``is_congested`` keeps the Part 1 definition of more than 5,500
    vehicles per hour, so that the probability analysis in Part 1 and the
    models in Part 3 can be compared against a common fixed benchmark.
    """
    out = df.copy()

    q1, q2, q3 = out["traffic_volume"].quantile([0.25, 0.50, 0.75]).values
    logger.debug(
        "Quartile thresholds for congestion_category — Q1 (25th) = %.1f, "
        "Q2 (median) = %.1f, Q3 (75th) = %.1f vehicles/hour", q1, q2, q3,
    )

    def bucket(volume: float) -> str:
        if volume <= q1:
            return "Low"
        if volume <= q2:
            return "Medium"
        if volume <= q3:
            return "High"
        return "Severe"

    out["congestion_category"] = out["traffic_volume"].apply(bucket)
    out["congestion_level"] = out["congestion_category"].map(
        {"Low": 0, "Medium": 1, "High": 2, "Severe": 3}
    ).astype(int)

    # Part 1's fixed threshold, retained for comparability.
    out["is_congested"] = (out["traffic_volume"] > 5500).astype(int)

    distribution = out["congestion_category"].value_counts().to_dict()
    logger.info(
        "Congestion target created from data-driven quartiles — "
        "Low <= %.0f, Medium <= %.0f, High <= %.0f, Severe > %.0f vehicles/hour",
        q1, q2, q3, q3,
    )
    logger.info("Congestion category distribution: %s", distribution)
    logger.info(
        "Fixed-threshold flag 'is_congested' (> 5,500 vehicles/hour): "
        "%s hour(s), %.2f%% of the dataset",
        f"{int(out['is_congested'].sum()):,}", 100 * out["is_congested"].mean(),
    )

    thresholds = {
        "q1_25th_percentile": float(q1),
        "q2_median": float(q2),
        "q3_75th_percentile": float(q3),
        "fixed_congestion_threshold": 5500,
        "distribution": {str(k): int(v) for k, v in distribution.items()},
    }
    return out, thresholds


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the full feature-engineering chain."""
    log_stage_banner(logger, "Feature engineering")
    shape_before = df.shape
    logger.info(
        "Dataset shape BEFORE feature engineering: %s rows x %s columns",
        f"{shape_before[0]:,}", shape_before[1],
    )

    out = add_time_features(df)
    out = add_cyclical_features(out)
    out = add_weather_features(out)
    out, scaling_params = add_scaled_features(out)
    out, thresholds = add_congestion_target(out)

    shape_after = out.shape
    logger.info(
        "Dataset shape AFTER feature engineering: %s rows x %s columns",
        f"{shape_after[0]:,}", shape_after[1],
    )
    logger.info(
        "Feature engineering added %s new columns; row count unchanged at %s",
        shape_after[1] - shape_before[1], f"{shape_after[0]:,}",
    )

    if shape_after[0] != shape_before[0]:
        logger.warning(
            "Row count changed during feature engineering: %s -> %s. Feature "
            "engineering must not drop rows.", shape_before[0], shape_after[0],
        )

    nulls = out.isna().sum()
    if int(nulls.sum()) > 0:
        logger.warning(
            "Feature engineering produced %s null value(s) in: %s",
            int(nulls.sum()), nulls[nulls > 0].to_dict(),
        )
    else:
        logger.info("No null values introduced by feature engineering")

    metadata = {
        "rows": int(shape_after[0]),
        "columns_before": int(shape_before[1]),
        "columns_after": int(shape_after[1]),
        "columns_added": int(shape_after[1] - shape_before[1]),
        "congestion_thresholds": thresholds,
        "scaling_parameters": scaling_params,
        "severe_weather_categories": SEVERE_WEATHER,
        "low_visibility_categories": LOW_VISIBILITY_WEATHER,
        "weather_severity_scale": WEATHER_SEVERITY_SCORE,
        "feature_names": list(out.columns),
    }
    return out, metadata


def save_features(df: pd.DataFrame, metadata: dict,
                  out_path: Path, meta_path: Path) -> None:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except OSError:
        logger.error("Could not write feature outputs", exc_info=True)
        raise
    logger.info(
        "Feature table saved to %s — %s rows, %s columns, %.2f MB",
        out_path, f"{len(df):,}", df.shape[1],
        out_path.stat().st_size / 1024**2,
    )
    logger.info("Feature metadata saved to %s", meta_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineer ML-ready features from the cleaned traffic data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--debug", action="store_true",
                        help="show DEBUG messages (thresholds, scaling parameters)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(log_file=PART2_DIR / "logs" / "pipeline.log", debug=args.debug)

    logger.info("#" * 72)
    logger.info("Capstone Part 2 — feature engineering starting")
    logger.info("#" * 72)

    try:
        clean = load_clean_data(args.input)
        features, metadata = engineer_features(clean)
        save_features(features, metadata, args.output, METADATA_PATH)
    except (FileNotFoundError, PermissionError, OSError):
        logger.error("Feature engineering aborted — file access problem", exc_info=True)
        return 3
    except (ValueError, KeyError, TypeError):
        logger.error("Feature engineering aborted — data problem", exc_info=True)
        return 5

    logger.info("Feature engineering finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
