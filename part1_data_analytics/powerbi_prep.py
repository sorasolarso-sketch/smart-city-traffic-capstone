"""
Capstone Part 1 - Task 4 : Power BI data preparation and dashboard answers.

Power BI Desktop is a Windows application and cannot be executed inside
this project's environment, so this script does two things instead:

  1. Applies exactly the same transformations as the Power Query script in
     `powerbi/power_query_script.m`, and writes the result to
     `powerbi/traffic_powerbi_ready.csv`. Loading that file into Power BI
     reproduces the prepared model directly.

  2. Computes every number the dashboard is required to display, so the
     figures quoted in the report and in the dashboard build guide are
     verified against the data rather than read off a screenshot.

Run from the repository root:

    python part1_data_analytics/powerbi_prep.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PART1_DIR = Path(__file__).resolve().parent
REPO_ROOT = PART1_DIR.parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
PBI_DIR = PART1_DIR / "powerbi"
OUT_DIR = PART1_DIR / "outputs"

SEVERITY_ORDER = [
    "Clear", "Clouds", "Mist", "Haze", "Fog", "Smoke",
    "Drizzle", "Rain", "Snow", "Squall", "Thunderstorm",
]


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def traffic_category(volume: float) -> str:
    """Fixed-threshold category defined in the Part 1 brief.

    Note this is deliberately NOT the same as the quartile-based
    `congestion_category` used in Part 3, which the Part 3 brief defines
    from the data rather than from fixed cut-points.
    """
    if volume < 4500:
        return "Low"
    if volume <= 5500:
        return "Medium"
    return "High"


def profile_raw(df: pd.DataFrame) -> dict:
    """Task 4.1 - data quality assessment of the file as supplied."""
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    blank_strings = {
        c: int((df[c].astype(str).str.strip() == "").sum())
        for c in df.columns if pd.api.types.is_string_dtype(df[c])
    }
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes_as_loaded": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values_per_column": missing,
        "total_missing_values": int(sum(missing.values())),
        "blank_string_values": blank_strings,
        "holiday_sentinel_none_rows": int((df["holiday"] == "None").sum()),
        "holiday_named_rows": int((df["holiday"] != "None").sum()),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "duplicate_timestamp_rows": int(df["date_time"].duplicated().sum()),
        "impossible_temp_zero_kelvin": int((df["temp"] < 100).sum()),
        "impossible_rain_over_9000mm": int((df["rain_1h"] > 9000).sum()),
        "zero_traffic_volume_rows": int((df["traffic_volume"] == 0).sum()),
    }


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Power Query transformation chain."""
    rank = {w: i for i, w in enumerate(SEVERITY_ORDER)}
    out = df.copy()
    out["date_time"] = pd.to_datetime(out["date_time"])

    before = len(out)
    out["_sev"] = out["weather_main"].map(rank).fillna(-1)
    out = out.sort_values(["date_time", "_sev"]).drop_duplicates(
        "date_time", keep="last"
    ).drop(columns="_sev")
    logger.warning(
        "Removed %s duplicate-timestamp rows (weather feed reports multiple "
        "conditions per hour); most severe condition retained", before - len(out)
    )

    # Repair the physically impossible sensor readings before they reach the model.
    bad_temp = out["temp"] < 100
    if bad_temp.any():
        median_temp = out.loc[~bad_temp, "temp"].median()
        out.loc[bad_temp, "temp"] = median_temp
        logger.warning("Replaced %s temperature readings of 0 K with the median "
                       "(%.2f K)", int(bad_temp.sum()), median_temp)

    bad_rain = out["rain_1h"] > 9000
    if bad_rain.any():
        median_rain = out.loc[~bad_rain, "rain_1h"].median()
        out.loc[bad_rain, "rain_1h"] = median_rain
        logger.warning("Replaced %s rainfall readings above 9,000 mm with the "
                       "median (%.2f mm)", int(bad_rain.sum()), median_rain)

    # Derived columns required by Task 4.1
    out["Date"] = out["date_time"].dt.date
    out["Year"] = out["date_time"].dt.year
    out["Month"] = out["date_time"].dt.month
    out["MonthName"] = out["date_time"].dt.strftime("%b")
    out["Hour"] = out["date_time"].dt.hour
    out["DayOfWeek"] = out["date_time"].dt.day_name()
    out["IsWeekend"] = out["date_time"].dt.dayofweek >= 5
    out["TempCelsius"] = (out["temp"] - 273.15).round(2)
    out["TrafficCategory"] = out["traffic_volume"].apply(traffic_category)
    out["IsHoliday"] = out["holiday"] != "None"

    logger.info("Prepared table: %s rows x %s columns", *out.shape)
    return out


def dashboard_answers(df: pd.DataFrame) -> dict:
    """Task 4.2 and 4.3 - every figure the dashboard must display."""
    # --- C. Weather impact -------------------------------------------------
    weather = (
        df.groupby("weather_main")
          .agg(hours=("traffic_volume", "size"),
               avg_traffic=("traffic_volume", "mean"))
          .sort_values("avg_traffic", ascending=False)
    )
    # Categories with a handful of observations cannot support a conclusion.
    reliable = weather[weather["hours"] >= 100]

    highest, lowest = weather.index[0], weather.index[-1]
    r_high, r_low = reliable.index[0], reliable.index[-1]

    # --- B. Hourly pattern, 2017 ------------------------------------------
    y2017 = df[df["Year"] == 2017]
    hourly_2017 = y2017.groupby("Hour")["traffic_volume"].mean()

    # --- A. Daily trends, 2015-2017 ---------------------------------------
    daily = (
        df[df["Year"].isin([2015, 2016, 2017])]
        .groupby(["Year", "Date"])["traffic_volume"].mean()
        .reset_index()
    )
    daily_by_year = {
        str(int(y)): {
            "days": int(len(g)),
            "mean_of_daily_averages": float(g["traffic_volume"].mean()),
            "min_daily_average": float(g["traffic_volume"].min()),
            "max_daily_average": float(g["traffic_volume"].max()),
            "lowest_day": str(g.loc[g["traffic_volume"].idxmin(), "Date"]),
            "highest_day": str(g.loc[g["traffic_volume"].idxmax(), "Date"]),
        }
        for y, g in daily.groupby("Year")
    }

    # --- D. Temperature vs traffic ----------------------------------------
    bins = list(range(-35, 45, 5))
    temp_band = pd.cut(df["TempCelsius"], bins=bins)
    by_band = df.groupby(temp_band, observed=True).agg(
        hours=("traffic_volume", "size"),
        avg_traffic=("traffic_volume", "mean"),
    )
    reliable_bands = by_band[by_band["hours"] >= 200]
    best_band = reliable_bands["avg_traffic"].idxmax()

    # Outliers: hours whose volume is far from what that hour of day normally sees
    hour_mean = df.groupby("Hour")["traffic_volume"].transform("mean")
    hour_std = df.groupby("Hour")["traffic_volume"].transform("std")
    z = (df["traffic_volume"] - hour_mean) / hour_std
    outliers = df.loc[z.abs() > 4, ["date_time", "Hour", "temp", "TempCelsius",
                                    "weather_main", "traffic_volume"]]

    # --- 4.3 KPI cards -----------------------------------------------------
    kpis = {
        "total_hours_analysed": int(len(df)),
        "average_traffic_volume": float(df["traffic_volume"].mean()),
        "average_temperature_kelvin": float(df["temp"].mean()),
        "average_temperature_celsius": float(df["TempCelsius"].mean()),
        "total_vehicles_recorded": int(df["traffic_volume"].sum()),
        "pct_hours_high_category": float((df["TrafficCategory"] == "High").mean()),
    }

    return {
        "A_daily_trends_2015_2017": daily_by_year,
        "B_hourly_2017": {
            "hours_of_data": int(len(y2017)),
            "by_hour": {str(int(h)): float(v) for h, v in hourly_2017.items()},
            "peak_hour": int(hourly_2017.idxmax()),
            "peak_hour_average": float(hourly_2017.max()),
            "quietest_hour": int(hourly_2017.idxmin()),
            "quietest_hour_average": float(hourly_2017.min()),
            "morning_peak_hour": int(hourly_2017.loc[5:10].idxmax()),
            "morning_peak_average": float(hourly_2017.loc[5:10].max()),
            "evening_peak_hour": int(hourly_2017.loc[14:20].idxmax()),
            "evening_peak_average": float(hourly_2017.loc[14:20].max()),
        },
        "C_weather_impact": {
            "table": {
                str(k): {"hours": int(v["hours"]), "avg_traffic": float(v["avg_traffic"])}
                for k, v in weather.to_dict("index").items()
            },
            "highest_avg_weather": str(highest),
            "highest_avg_value": float(weather.loc[highest, "avg_traffic"]),
            "highest_avg_hours": int(weather.loc[highest, "hours"]),
            "lowest_avg_weather": str(lowest),
            "lowest_avg_value": float(weather.loc[lowest, "avg_traffic"]),
            "lowest_avg_hours": int(weather.loc[lowest, "hours"]),
            "difference_highest_minus_lowest": float(
                weather.loc[highest, "avg_traffic"] - weather.loc[lowest, "avg_traffic"]
            ),
            "reliable_only_min_100_hours": {
                "highest_weather": str(r_high),
                "highest_value": float(reliable.loc[r_high, "avg_traffic"]),
                "lowest_weather": str(r_low),
                "lowest_value": float(reliable.loc[r_low, "avg_traffic"]),
                "difference": float(
                    reliable.loc[r_high, "avg_traffic"] - reliable.loc[r_low, "avg_traffic"]
                ),
            },
        },
        "D_temperature_vs_traffic": {
            "pearson_r_celsius_vs_volume": float(
                df["TempCelsius"].corr(df["traffic_volume"])
            ),
            "avg_traffic_by_temp_band": {
                str(k): {"hours": int(v["hours"]), "avg_traffic": float(v["avg_traffic"])}
                for k, v in by_band.to_dict("index").items()
            },
            "highest_traffic_temp_band": str(best_band),
            "highest_traffic_temp_band_value": float(
                reliable_bands.loc[best_band, "avg_traffic"]
            ),
            "outlier_count_z_gt_4_within_hour": int(len(outliers)),
            "outlier_examples": outliers.head(12).assign(
                date_time=lambda d: d["date_time"].astype(str)
            ).to_dict("records"),
            "zero_volume_hours": int((df["traffic_volume"] == 0).sum()),
        },
        "KPI_cards": kpis,
        "slicer_domains": {
            "hour_range": [0, 23],
            "weather_conditions": sorted(df["weather_main"].unique().tolist()),
            "traffic_categories": ["Low", "Medium", "High"],
        },
        "traffic_category_distribution": {
            k: int(v) for k, v in df["TrafficCategory"].value_counts().items()
        },
    }


def main() -> int:
    configure_logging()
    PBI_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        raw = pd.read_csv(RAW_CSV, keep_default_na=False)
    except (OSError, ValueError):
        logger.error("Could not read the raw CSV", exc_info=True)
        return 1

    logger.info("Loaded raw file: %s rows x %s columns", *raw.shape)
    profile = profile_raw(raw)
    prepared = prepare(raw)

    out_csv = PBI_DIR / "traffic_powerbi_ready.csv"
    prepared.to_csv(out_csv, index=False)
    logger.info("Wrote Power BI ready file to %s", out_csv)

    results = {
        "task_4_1_data_quality": profile,
        "prepared_table": {
            "rows": int(len(prepared)),
            "columns": int(prepared.shape[1]),
            "column_names": list(prepared.columns),
        },
        "task_4_2_and_4_3": dashboard_answers(prepared),
    }
    (OUT_DIR / "powerbi_answers.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    c = results["task_4_2_and_4_3"]["C_weather_impact"]
    logger.info("Weather impact: highest=%s (%.0f), lowest=%s (%.0f), difference=%.0f",
                c["highest_avg_weather"], c["highest_avg_value"],
                c["lowest_avg_weather"], c["lowest_avg_value"],
                c["difference_highest_minus_lowest"])
    logger.info("Dashboard answers written to %s", OUT_DIR / "powerbi_answers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
