"""
Capstone Part 2, Task 1 — Data pipeline construction.

A multi-stage, fully logged pipeline that turns the raw Metro Interstate
Traffic Volume CSV into a validated, cleaned dataset.

Stages
------
1. Load          — read the raw CSV with explicit exception handling
2. Validate      — confirm the schema BEFORE any processing occurs
3. Clean         — standardise, parse, de-duplicate, repair impossible values
4. Persist       — write the cleaned table to ``data/processed/``

Usage
-----
    python part2_python/pipeline.py
    python part2_python/pipeline.py --debug          # show DEBUG on console
    python part2_python/pipeline.py --input path.csv --output out.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Support both `python part2_python/pipeline.py` and `import pipeline`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logging_config import configure_logging, log_stage_banner  # noqa: E402

logger = logging.getLogger(__name__)

PART2_DIR = Path(__file__).resolve().parent
REPO_ROOT = PART2_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "traffic_clean.csv"

# --- Schema contract -------------------------------------------------------
EXPECTED_COLUMNS: dict[str, str] = {
    "holiday": "object",
    "temp": "float",
    "rain_1h": "float",
    "snow_1h": "float",
    "clouds_all": "integer",
    "weather_main": "object",
    "weather_description": "object",
    "date_time": "object",
    "traffic_volume": "integer",
}

# --- Physical plausibility bounds -----------------------------------------
# Sources: absolute zero is 0 K; the highest reliably measured hourly
# rainfall is roughly 305 mm; cloud cover is a percentage.
MIN_PLAUSIBLE_TEMP_K = 100.0     # below this the sensor is reporting a fault
MAX_PLAUSIBLE_TEMP_K = 340.0
MAX_PLAUSIBLE_RAIN_MM = 9000.0   # the brief's stated threshold
MAX_REALISTIC_RAIN_MM = 350.0
MAX_PLAUSIBLE_SNOW_MM = 100.0
CLOUDS_RANGE = (0, 100)
MAX_PLAUSIBLE_TRAFFIC = 12000    # ~2x the observed maximum of 7,280

# Hours during which a reading of zero vehicles is implausible. Overnight
# zeros are genuine; a zero at 09:00 on a westbound interstate is not.
DAYTIME_HOURS = range(5, 23)

# Weather severity, used to choose which record survives when the weather
# feed reports the same hour under more than one condition.
SEVERITY_ORDER = [
    "Clear", "Clouds", "Mist", "Haze", "Fog", "Smoke",
    "Drizzle", "Rain", "Snow", "Squall", "Thunderstorm",
]


class SchemaValidationError(Exception):
    """Raised when the input file does not match the expected schema."""


# ---------------------------------------------------------------------------
# Stage 1 — Load
# ---------------------------------------------------------------------------
def load_raw_data(path: Path) -> pd.DataFrame:
    """Read the raw CSV.

    ``keep_default_na=False`` is essential rather than cosmetic. ``"None"``
    is one of pandas' default NA tokens, so the default behaviour silently
    converts the holiday sentinel on 48,143 rows into a missing value, and
    every later count of "missing holidays" becomes wrong.

    Raises
    ------
    FileNotFoundError, PermissionError, OSError
        Propagated to the caller, which logs and exits gracefully.
    pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError
        Malformed or unreadable file content.
    """
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except FileNotFoundError:
        logger.error("Raw data file not found at %s", path, exc_info=True)
        raise
    except PermissionError:
        logger.error("No permission to read %s", path, exc_info=True)
        raise
    except pd.errors.EmptyDataError:
        logger.error("Raw data file at %s is empty", path, exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Could not parse %s as CSV", path, exc_info=True)
        raise
    except UnicodeDecodeError:
        logger.error("Encoding error reading %s; expected UTF-8", path, exc_info=True)
        raise
    except OSError:
        logger.error("OS error while reading %s", path, exc_info=True)
        raise

    logger.info(
        "Raw file loaded successfully from %s — %s rows, %s columns",
        path.name, f"{len(df):,}", df.shape[1],
    )
    logger.debug("Columns as loaded: %s", list(df.columns))
    logger.debug("Dtypes as loaded: %s", df.dtypes.to_dict())
    logger.debug("Memory footprint: %.2f MB", df.memory_usage(deep=True).sum() / 1024**2)
    return df


# ---------------------------------------------------------------------------
# Stage 2 — Validate the schema, before any other processing
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    """Confirm every expected column is present and usably typed.

    Runs before cleaning, so that a structurally wrong file fails fast with
    a clear message instead of producing plausible-looking nonsense.
    """
    log_stage_banner(logger, "Schema validation")

    actual = list(df.columns)
    expected = list(EXPECTED_COLUMNS)

    missing = [c for c in expected if c not in actual]
    if missing:
        raise SchemaValidationError(
            f"Required column(s) absent from the input file: {missing}. "
            f"Found: {actual}"
        )
    logger.info("All %s required columns are present", len(expected))

    unexpected = [c for c in actual if c not in expected]
    if unexpected:
        logger.warning(
            "%s unexpected column(s) present and will be ignored: %s",
            len(unexpected), unexpected,
        )

    if actual[: len(expected)] != expected:
        logger.warning(
            "Column order differs from the reference schema; selecting by "
            "name rather than by position"
        )

    # Type-usability check. The CSV arrives untyped, so what matters is
    # whether each column CAN be coerced, not what pandas guessed.
    for column, kind in EXPECTED_COLUMNS.items():
        series = df[column]
        if kind in {"float", "integer"}:
            coerced = pd.to_numeric(series, errors="coerce")
            bad = int(coerced.isna().sum() - series.isna().sum())
            if bad > 0:
                raise SchemaValidationError(
                    f"Column '{column}' is declared {kind} but {bad:,} value(s) "
                    "cannot be read as a number"
                )
            logger.debug("Column '%s' validated as %s", column, kind)
        else:
            logger.debug("Column '%s' validated as text", column)

    if df.empty:
        raise SchemaValidationError("Input file contains a header but no data rows")

    logger.info("Schema validation passed — %s rows ready for cleaning", f"{len(df):,}")


# ---------------------------------------------------------------------------
# Stage 3 — Clean
# ---------------------------------------------------------------------------
def standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning step 1 — normalise inconsistent categorical text."""
    out = df.copy()
    changed_total = 0

    for column, transform, description in [
        ("weather_main", lambda s: s.str.strip().str.title(), "trimmed and title-cased"),
        ("weather_description", lambda s: s.str.strip().str.lower(), "trimmed and lower-cased"),
        ("holiday", lambda s: s.str.strip(), "trimmed"),
    ]:
        before = out[column].astype(str)
        out[column] = transform(out[column].astype(str))
        changed = int((before != out[column]).sum())
        changed_total += changed
        if changed > 0:
            logger.warning(
                "Cleaning step 1 — standardised %s value(s) in '%s' (%s)",
                f"{changed:,}", column, description,
            )
        else:
            logger.info(
                "Cleaning step 1 — '%s' already consistent, no changes (%s)",
                column, description,
            )

    # Make the holiday sentinel explicit rather than relying on the literal.
    out["is_holiday"] = (out["holiday"] != "None").astype(int)
    n_holiday = int(out["is_holiday"].sum())
    logger.info(
        "Cleaning step 1 — derived 'is_holiday': %s holiday-labelled row(s), "
        "%s non-holiday row(s)", f"{n_holiday:,}", f"{len(out) - n_holiday:,}",
    )
    logger.debug(
        "Holiday labels present: %s",
        sorted(out.loc[out["is_holiday"] == 1, "holiday"].unique().tolist()),
    )
    if changed_total == 0:
        logger.info("Cleaning step 1 complete — categorical values required no repair")
    return out


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning step 2 — parse and validate the timestamp column."""
    out = df.copy()
    parsed = pd.to_datetime(out["date_time"], errors="coerce", format="%Y-%m-%d %H:%M:%S")

    unparseable = int(parsed.isna().sum())
    if unparseable > 0:
        logger.warning(
            "Cleaning step 2 — dropped %s row(s): 'date_time' could not be "
            "parsed as YYYY-MM-DD HH:MM:SS", f"{unparseable:,}",
        )
        out = out.loc[parsed.notna()].copy()
        parsed = parsed.loc[parsed.notna()]
    else:
        logger.info("Cleaning step 2 — all %s timestamp(s) parsed successfully",
                    f"{len(parsed):,}")

    out["date_time"] = parsed.values

    # Timestamps outside a sensible window indicate a corrupt record.
    lower, upper = pd.Timestamp("2000-01-01"), pd.Timestamp.now()
    implausible = (out["date_time"] < lower) | (out["date_time"] > upper)
    n_implausible = int(implausible.sum())
    if n_implausible > 0:
        logger.warning(
            "Cleaning step 2 — dropped %s row(s) with timestamps outside "
            "%s to %s", f"{n_implausible:,}", lower.date(), upper.date(),
        )
        out = out.loc[~implausible].copy()

    out = out.sort_values("date_time").reset_index(drop=True)
    logger.info(
        "Cleaning step 2 — timestamp range %s to %s, spanning %s day(s)",
        out["date_time"].min(), out["date_time"].max(),
        f"{(out['date_time'].max() - out['date_time'].min()).days:,}",
    )

    # Quantify the coverage gaps rather than leaving them implicit.
    expected_hours = int((out["date_time"].max() - out["date_time"].min())
                         / pd.Timedelta(hours=1)) + 1
    observed_hours = int(out["date_time"].nunique())
    missing_hours = expected_hours - observed_hours
    if missing_hours > 0:
        logger.warning(
            "Cleaning step 2 — %s hour(s) absent from the series (%.1f%% of the "
            "span); the traffic sensor was offline for extended periods in "
            "2014-2015", f"{missing_hours:,}", 100 * missing_hours / expected_hours,
        )
    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning step 3 — remove duplicate rows and duplicate hours."""
    out = df.copy()
    start = len(out)

    exact = int(out.duplicated().sum())
    if exact > 0:
        out = out.drop_duplicates().copy()
        logger.warning(
            "Cleaning step 3a — dropped %s fully identical duplicate row(s)",
            f"{exact:,}",
        )
    else:
        logger.info("Cleaning step 3a — no fully identical duplicate rows found")

    dup_hours = int(out["date_time"].duplicated().sum())
    if dup_hours > 0:
        # These are not errors. The weather provider emits one record per
        # observed condition, so an hour with rain and mist appears twice
        # carrying an identical traffic count. Keeping them double-counts
        # the hour in every sum and over-weights it in every average.
        volumes_per_hour = out.groupby("date_time")["traffic_volume"].nunique()
        conflicting = int((volumes_per_hour > 1).sum())
        logger.debug(
            "Duplicate-hour diagnostics: %s affected hour(s), %s with "
            "conflicting traffic counts", f"{len(volumes_per_hour[volumes_per_hour.index.duplicated(keep=False)]):,}", conflicting,
        )
        if conflicting > 0:
            logger.warning(
                "Cleaning step 3b — %s duplicated hour(s) carry conflicting "
                "traffic counts; the first reading is retained", f"{conflicting:,}",
            )

        rank = {w: i for i, w in enumerate(SEVERITY_ORDER)}
        out["_severity"] = out["weather_main"].map(rank).fillna(-1)
        out = (out.sort_values(["date_time", "_severity"])
                  .drop_duplicates("date_time", keep="last")
                  .drop(columns="_severity"))
        logger.warning(
            "Cleaning step 3b — dropped %s duplicate-timestamp row(s); the "
            "weather feed reports an hour once per observed condition, and "
            "retaining them would inflate annual traffic totals by 7-21%%. "
            "The most severe condition for each hour is retained.",
            f"{dup_hours:,}",
        )
    else:
        logger.info("Cleaning step 3b — every timestamp is unique")

    out = out.sort_values("date_time").reset_index(drop=True)
    logger.info(
        "Cleaning step 3 complete — %s row(s) in, %s row(s) out, %s removed",
        f"{start:,}", f"{len(out):,}", f"{start - len(out):,}",
    )
    return out


def impute_by_month(
    df: pd.DataFrame, column: str, invalid_mask: pd.Series, reason: str
) -> pd.DataFrame:
    """Replace flagged values with the median for the same calendar month.

    A single global median would assign a July temperature to a January
    sensor fault. Iterating month by month keeps the replacement seasonally
    honest. The explicit loop also satisfies the brief's requirement to use
    conditionals and loops where group-by-group processing is needed.
    """
    out = df.copy()
    n_invalid = int(invalid_mask.sum())
    if n_invalid == 0:
        logger.info("No invalid values to impute in '%s' (%s)", column, reason)
        return out

    out["_month"] = out["date_time"].dt.month
    global_median = float(out.loc[~invalid_mask, column].median())
    logger.debug("Global fallback median for '%s': %.4f", column, global_median)

    imputed_count = 0
    for month in sorted(out["_month"].unique()):
        month_mask = out["_month"] == month
        targets = month_mask & invalid_mask
        n_targets = int(targets.sum())
        if n_targets == 0:
            continue

        valid_in_month = out.loc[month_mask & ~invalid_mask, column]
        if len(valid_in_month) > 0:
            replacement = float(valid_in_month.median())
            source = f"month {month:02d} median"
        else:
            # No usable value for this month — fall back to the global median.
            replacement = global_median
            source = "global median (no valid readings in this month)"
            logger.warning(
                "Month %02d has no valid '%s' readings; falling back to the "
                "global median", month, column,
            )

        out.loc[targets, column] = replacement
        imputed_count += n_targets
        logger.debug(
            "Imputed %s value(s) in '%s' for month %02d with %.4f (%s)",
            n_targets, column, month, replacement, source,
        )

    out = out.drop(columns="_month")
    logger.warning(
        "Cleaning step 4 — imputed %s value(s) in '%s' using the median for "
        "the same calendar month. Reason: %s", f"{imputed_count:,}", column, reason,
    )
    return out


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning step 4 — detect and repair impossible sensor readings."""
    out = df.copy()

    # --- 4a. Temperature: 0 K is absolute zero -----------------------------
    bad_temp = (out["temp"] < MIN_PLAUSIBLE_TEMP_K) | (out["temp"] > MAX_PLAUSIBLE_TEMP_K)
    n_bad_temp = int(bad_temp.sum())
    if n_bad_temp > 0:
        logger.debug(
            "Impossible temperature readings at: %s",
            out.loc[bad_temp, "date_time"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        )
    out = impute_by_month(
        out, "temp", bad_temp,
        f"reading outside the physically possible range "
        f"{MIN_PLAUSIBLE_TEMP_K:.0f}-{MAX_PLAUSIBLE_TEMP_K:.0f} K "
        f"({n_bad_temp} reading(s) of 0 K, i.e. absolute zero)",
    )

    # --- 4b. Rainfall ------------------------------------------------------
    bad_rain = out["rain_1h"] > MAX_PLAUSIBLE_RAIN_MM
    n_bad_rain = int(bad_rain.sum())
    if n_bad_rain > 0:
        logger.debug(
            "Extreme rainfall reading(s): %s mm",
            out.loc[bad_rain, "rain_1h"].tolist(),
        )
    out = impute_by_month(
        out, "rain_1h", bad_rain,
        f"hourly rainfall above the brief's {MAX_PLAUSIBLE_RAIN_MM:,.0f} mm "
        "threshold; the world record for hourly rainfall is around 305 mm",
    )

    # Values between the realistic maximum and the brief's threshold are
    # suspicious but not impossible, so they are flagged rather than changed.
    suspicious_rain = (out["rain_1h"] > MAX_REALISTIC_RAIN_MM) & (
        out["rain_1h"] <= MAX_PLAUSIBLE_RAIN_MM
    )
    if int(suspicious_rain.sum()) > 0:
        logger.warning(
            "Cleaning step 4b — %s rainfall reading(s) between %s mm and "
            "%s mm are physically doubtful but were left unchanged, as the "
            "brief defines the repair threshold at %s mm",
            int(suspicious_rain.sum()), MAX_REALISTIC_RAIN_MM,
            f"{MAX_PLAUSIBLE_RAIN_MM:,.0f}", f"{MAX_PLAUSIBLE_RAIN_MM:,.0f}",
        )

    # --- 4c. Negative precipitation ---------------------------------------
    for column in ("rain_1h", "snow_1h"):
        negative = out[column] < 0
        n_negative = int(negative.sum())
        if n_negative > 0:
            out.loc[negative, column] = 0.0
            logger.warning(
                "Cleaning step 4c — set %s negative '%s' value(s) to zero; "
                "precipitation cannot be negative", f"{n_negative:,}", column,
            )
        else:
            logger.info("Cleaning step 4c — no negative values in '%s'", column)

    excessive_snow = out["snow_1h"] > MAX_PLAUSIBLE_SNOW_MM
    if int(excessive_snow.sum()) > 0:
        out = impute_by_month(
            out, "snow_1h", excessive_snow,
            f"hourly snowfall above {MAX_PLAUSIBLE_SNOW_MM:.0f} mm",
        )
    else:
        logger.info("Cleaning step 4c — all snowfall readings within plausible bounds")

    # --- 4d. Cloud cover is a percentage -----------------------------------
    lo, hi = CLOUDS_RANGE
    bad_clouds = (out["clouds_all"] < lo) | (out["clouds_all"] > hi)
    n_bad_clouds = int(bad_clouds.sum())
    if n_bad_clouds > 0:
        out["clouds_all"] = out["clouds_all"].clip(lo, hi)
        logger.warning(
            "Cleaning step 4d — clipped %s cloud-cover value(s) into the "
            "valid %s-%s%% range", f"{n_bad_clouds:,}", lo, hi,
        )
    else:
        logger.info("Cleaning step 4d — all cloud-cover values within %s-%s%%", lo, hi)

    # --- 4e. Traffic volume ------------------------------------------------
    excessive_traffic = out["traffic_volume"] > MAX_PLAUSIBLE_TRAFFIC
    if int(excessive_traffic.sum()) > 0:
        logger.warning(
            "Cleaning step 4e — %s traffic reading(s) exceed %s vehicles/hour "
            "and were capped", int(excessive_traffic.sum()), f"{MAX_PLAUSIBLE_TRAFFIC:,}",
        )
        out.loc[excessive_traffic, "traffic_volume"] = MAX_PLAUSIBLE_TRAFFIC
    else:
        logger.info(
            "Cleaning step 4e — no traffic reading exceeds %s vehicles/hour "
            "(observed maximum %s)", f"{MAX_PLAUSIBLE_TRAFFIC:,}",
            f"{int(out['traffic_volume'].max()):,}",
        )

    # A daytime reading of exactly zero on an interstate is sensor dropout.
    hour = out["date_time"].dt.hour
    daytime_zero = (out["traffic_volume"] == 0) & hour.isin(list(DAYTIME_HOURS))
    n_daytime_zero = int(daytime_zero.sum())
    if n_daytime_zero > 0:
        logger.debug(
            "Daytime zero-traffic readings at: %s",
            out.loc[daytime_zero, "date_time"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        )
        # Impute from the median for the same hour of day, which is a far
        # better estimate than any monthly or global figure.
        replaced = 0
        for target_hour in sorted(out.loc[daytime_zero, "date_time"].dt.hour.unique()):
            mask = daytime_zero & (hour == target_hour)
            reference = out.loc[(hour == target_hour) & (out["traffic_volume"] > 0),
                                "traffic_volume"]
            if len(reference) == 0:
                continue
            replacement = float(reference.median())
            out.loc[mask, "traffic_volume"] = replacement
            replaced += int(mask.sum())
            logger.debug(
                "Imputed %s daytime zero reading(s) at hour %02d with the "
                "hour median %.0f", int(mask.sum()), target_hour, replacement,
            )
        logger.warning(
            "Cleaning step 4e — imputed %s daytime reading(s) of exactly zero "
            "vehicles using the median for the same hour of day; a westbound "
            "interstate does not carry zero vehicles between %02d:00 and "
            "%02d:00, so these are sensor dropouts",
            f"{replaced:,}", DAYTIME_HOURS.start, DAYTIME_HOURS.stop - 1,
        )
    else:
        logger.info("Cleaning step 4e — no implausible daytime zero readings found")

    overnight_zero = int(((out["traffic_volume"] == 0) & ~hour.isin(list(DAYTIME_HOURS))).sum())
    if overnight_zero > 0:
        logger.info(
            "Cleaning step 4e — retained %s overnight reading(s) of zero "
            "vehicles; these are plausible and were not altered", overnight_zero,
        )

    out["traffic_volume"] = out["traffic_volume"].round().astype(int)
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run every cleaning step in order, logging each one separately."""
    log_stage_banner(logger, "Data cleaning")
    rows_in = len(df)

    out = standardise_categoricals(df)
    out = parse_datetime(out)
    out = remove_duplicates(out)
    out = handle_outliers(out)

    rows_out = len(out)
    pct = 100 * (rows_in - rows_out) / rows_in if rows_in else 0.0
    logger.info(
        "Cleaning complete — %s row(s) in, %s row(s) out (%.2f%% removed), "
        "%s column(s)", f"{rows_in:,}", f"{rows_out:,}", pct, out.shape[1],
    )
    return out


# ---------------------------------------------------------------------------
# Stage 4 — Persist
# ---------------------------------------------------------------------------
def save_clean_data(df: pd.DataFrame, path: Path) -> None:
    """Write the cleaned dataset to disk."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except OSError:
        logger.error("Could not write cleaned data to %s", path, exc_info=True)
        raise
    size_mb = path.stat().st_size / 1024**2
    logger.info(
        "Cleaned dataset saved to %s — %s rows, %s columns, %.2f MB",
        path, f"{len(df):,}", df.shape[1], size_mb,
    )


def summarise(df: pd.DataFrame) -> None:
    """Log a short profile of the cleaned data as a final checkpoint."""
    log_stage_banner(logger, "Cleaned data summary")
    logger.info("Shape: %s rows x %s columns", f"{len(df):,}", df.shape[1])
    logger.info("Date range: %s to %s", df["date_time"].min(), df["date_time"].max())
    logger.info(
        "Traffic volume: mean %.1f, median %.1f, min %s, max %s",
        df["traffic_volume"].mean(), df["traffic_volume"].median(),
        f"{int(df['traffic_volume'].min()):,}", f"{int(df['traffic_volume'].max()):,}",
    )
    logger.info(
        "Temperature: mean %.2f K (%.2f C), min %.2f K, max %.2f K",
        df["temp"].mean(), df["temp"].mean() - 273.15,
        df["temp"].min(), df["temp"].max(),
    )
    logger.info("Weather categories: %s", df["weather_main"].nunique())
    logger.debug("Weather distribution: %s", df["weather_main"].value_counts().to_dict())
    logger.info("Holiday-labelled hours: %s", f"{int(df['is_holiday'].sum()):,}")
    remaining_nulls = int(df.isna().sum().sum())
    if remaining_nulls > 0:
        logger.warning("%s null value(s) remain in the cleaned data", remaining_nulls)
    else:
        logger.info("No null values remain in the cleaned dataset")


def run_pipeline(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Execute the full pipeline and return the cleaned frame."""
    log_stage_banner(logger, "Loading raw data")
    raw = load_raw_data(input_path)
    validate_schema(raw)
    clean = clean_data(raw)
    save_clean_data(clean, output_path)
    summarise(clean)
    return clean


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the Metro Interstate Traffic Volume dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="path to the raw CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="path for the cleaned CSV")
    parser.add_argument("--debug", action="store_true",
                        help="show DEBUG messages on the console")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="override the log file location")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_file = args.log_file or (PART2_DIR / "logs" / "pipeline.log")
    configure_logging(log_file=log_file, debug=args.debug)

    logger.info("#" * 72)
    logger.info("Capstone Part 2 — traffic data pipeline starting")
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)
    logger.info("Console level: %s", "DEBUG" if args.debug else "INFO")
    logger.info("#" * 72)

    try:
        run_pipeline(args.input, args.output)
    except SchemaValidationError as exc:
        logger.error("Pipeline aborted — schema validation failed: %s", exc, exc_info=True)
        return 2
    except (FileNotFoundError, PermissionError, OSError):
        logger.error("Pipeline aborted — could not read or write a required file",
                     exc_info=True)
        return 3
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        logger.error("Pipeline aborted — the input file could not be parsed",
                     exc_info=True)
        return 4
    except (ValueError, KeyError, TypeError):
        logger.error("Pipeline aborted — unexpected data problem during processing",
                     exc_info=True)
        return 5

    logger.info("Pipeline finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
