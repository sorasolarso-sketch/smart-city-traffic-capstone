"""
Capstone Part 1 - Tasks 2 and 3 : descriptive statistics, correlation,
probability and conditional probability.

Writes `outputs/statistics_probability_results.md` (human-readable) and
`outputs/statistics_probability_results.json` (machine-readable, reused by
the report builder).

Run from the repository root:

    python part1_data_analytics/statistics_probability.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PART1_DIR = Path(__file__).resolve().parent
REPO_ROOT = PART1_DIR.parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
OUT_DIR = PART1_DIR / "outputs"

CONGESTION_THRESHOLD = 5500   # vehicles per hour, as defined in the brief
HIGH_TEMP_THRESHOLD = 292.0   # Kelvin, as defined in the brief

# Weather severity order, used only when collapsing an hour that the
# weather feed reported under more than one category. The most severe
# condition observed in the hour is the one retained.
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


def load_raw() -> pd.DataFrame:
    """Load the raw CSV.

    `keep_default_na=False` is essential: 'None' is one of pandas' default
    NA tokens, so the default behaviour would silently turn every
    non-holiday row's holiday value into a missing value.
    """
    df = pd.read_csv(RAW_CSV, keep_default_na=False)
    df["date_time"] = pd.to_datetime(df["date_time"])
    logger.info("Loaded %s rows x %s columns", *df.shape)
    return df


def collapse_to_hours(df: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp, retaining the most severe weather category."""
    rank = {w: i for i, w in enumerate(SEVERITY_ORDER)}
    tmp = df.copy()
    tmp["_sev"] = tmp["weather_main"].map(rank).fillna(-1)
    tmp = tmp.sort_values(["date_time", "_sev"])
    hourly = tmp.drop_duplicates("date_time", keep="last").drop(columns="_sev")
    logger.info(
        "Collapsed %s rows to %s distinct hours (%s duplicate-timestamp rows removed)",
        len(df), len(hourly), len(df) - len(hourly),
    )
    return hourly


# ---------------------------------------------------------------------------
# Task 2.1 - descriptive statistics
# ---------------------------------------------------------------------------
def describe_traffic(series: pd.Series) -> dict:
    values = series.to_numpy(dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        # ddof=1: the dataset is treated as a sample of the corridor's
        # traffic process, not as the entire population of all hours.
        "std_dev_sample": float(np.std(values, ddof=1)),
        "variance_sample": float(np.var(values, ddof=1)),
        "std_dev_population": float(np.std(values, ddof=0)),
        "variance_population": float(np.var(values, ddof=0)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "coefficient_of_variation": float(np.std(values, ddof=1) / np.mean(values)),
        "skewness": float(stats.skew(values)),
        "kurtosis_excess": float(stats.kurtosis(values)),
    }


# ---------------------------------------------------------------------------
# Task 2.2 - correlation
# ---------------------------------------------------------------------------
def correlate(df: pd.DataFrame, label: str) -> dict:
    x = df["temp"].to_numpy(dtype=float)
    y = df["traffic_volume"].to_numpy(dtype=float)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    return {
        "label": label,
        "n": int(len(df)),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "r_squared": float(pearson_r ** 2),
        "spearman_rho": float(spearman_r),
        "spearman_p_value": float(spearman_p),
        "direction": "positive" if pearson_r > 0 else "negative",
        "strength": classify_strength(abs(pearson_r)),
    }


def classify_strength(abs_r: float) -> str:
    if abs_r < 0.10:
        return "negligible"
    if abs_r < 0.30:
        return "weak"
    if abs_r < 0.50:
        return "moderate"
    if abs_r < 0.70:
        return "strong"
    return "very strong"


# ---------------------------------------------------------------------------
# Task 3 - probability and conditional probability
# ---------------------------------------------------------------------------
def probability_analysis(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    congested = df["traffic_volume"] > CONGESTION_THRESHOLD
    clear = df["weather_main"] == "Clear"
    cloudy = df["weather_main"] == "Clouds"
    high_temp = df["temp"] > HIGH_TEMP_THRESHOLD

    p_cong = congested.mean()
    p_clear = clear.mean()
    p_both = (congested & clear).mean()

    # Conditional probabilities
    p_clear_given_cong = (congested & clear).sum() / congested.sum()
    p_high_temp_given_cong = (congested & high_temp).sum() / congested.sum()
    p_cong_given_clear = (congested & clear).sum() / clear.sum()
    p_cong_given_cloudy = (congested & cloudy).sum() / cloudy.sum()

    # Independence test: P(A n B) == P(A) x P(B) ?
    expected_if_independent = p_cong * p_clear
    difference = p_both - expected_if_independent
    ratio = p_both / expected_if_independent if expected_if_independent else float("nan")

    # Chi-square test of independence on the 2x2 congestion/clear table
    contingency = pd.crosstab(congested, clear)
    chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)
    # Phi coefficient, the effect size for a 2x2 table
    phi = float(np.sqrt(chi2 / n))

    # Odds ratio: congestion in clear versus cloudy weather
    a = int((congested & clear).sum())      # congested, clear
    b = int((~congested & clear).sum())     # not congested, clear
    c = int((congested & cloudy).sum())     # congested, cloudy
    d = int((~congested & cloudy).sum())    # not congested, cloudy
    odds_clear = a / b
    odds_cloudy = c / d
    odds_ratio = odds_clear / odds_cloudy
    # 95% CI via the Woolf (log) method
    se_log_or = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    ci_low = float(np.exp(np.log(odds_ratio) - 1.96 * se_log_or))
    ci_high = float(np.exp(np.log(odds_ratio) + 1.96 * se_log_or))

    # Congestion rate by weather category, for interpretation
    by_weather = (
        df.assign(congested=congested)
          .groupby("weather_main")
          .agg(hours=("congested", "size"),
               congested_hours=("congested", "sum"),
               congestion_rate=("congested", "mean"),
               avg_volume=("traffic_volume", "mean"))
          .sort_values("congestion_rate", ascending=False)
    )

    return {
        "label": label,
        "n": int(n),
        "congestion_threshold": CONGESTION_THRESHOLD,
        "high_temp_threshold_kelvin": HIGH_TEMP_THRESHOLD,
        "counts": {
            "congested_hours": int(congested.sum()),
            "clear_hours": int(clear.sum()),
            "cloudy_hours": int(cloudy.sum()),
            "congested_and_clear": a,
            "clear_not_congested": b,
            "congested_and_cloudy": c,
            "cloudy_not_congested": d,
        },
        "p_congestion": float(p_cong),
        "p_clear_weather": float(p_clear),
        "p_congestion_and_clear": float(p_both),
        "p_clear_given_congestion": float(p_clear_given_cong),
        "p_high_temp_given_congestion": float(p_high_temp_given_cong),
        "p_congestion_given_clear": float(p_cong_given_clear),
        "p_congestion_given_cloudy": float(p_cong_given_cloudy),
        "independence": {
            "p_a_and_b_observed": float(p_both),
            "p_a_times_p_b_expected": float(expected_if_independent),
            "absolute_difference": float(difference),
            "observed_over_expected": float(ratio),
            "independent": bool(abs(difference) < 0.001),
            "chi_square": float(chi2),
            "chi_square_p_value": float(chi_p),
            "degrees_of_freedom": int(dof),
            "phi_coefficient": phi,
        },
        "odds_ratio_clear_vs_cloudy": {
            "odds_congestion_clear": float(odds_clear),
            "odds_congestion_cloudy": float(odds_cloudy),
            "odds_ratio": float(odds_ratio),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        },
        "congestion_rate_by_weather": {
            str(k): {
                "hours": int(v["hours"]),
                "congested_hours": int(v["congested_hours"]),
                "congestion_rate": float(v["congestion_rate"]),
                "avg_volume": float(v["avg_volume"]),
            }
            for k, v in by_weather.to_dict("index").items()
        },
    }


def congestion_by_hour(df: pd.DataFrame) -> dict:
    """Congestion rate by hour of day - the confounder behind the weather result."""
    tmp = df.assign(
        hour=df["date_time"].dt.hour,
        congested=df["traffic_volume"] > CONGESTION_THRESHOLD,
        clear=df["weather_main"] == "Clear",
    )
    by_hour = tmp.groupby("hour").agg(
        hours=("congested", "size"),
        congestion_rate=("congested", "mean"),
        clear_rate=("clear", "mean"),
        avg_volume=("traffic_volume", "mean"),
    )
    return {
        str(int(k)): {
            "hours": int(v["hours"]),
            "congestion_rate": float(v["congestion_rate"]),
            "clear_rate": float(v["clear_rate"]),
            "avg_volume": float(v["avg_volume"]),
        }
        for k, v in by_hour.to_dict("index").items()
    }


def hour_controlled_odds(df: pd.DataFrame) -> dict:
    """Congestion rate in clear vs cloudy weather, WITHIN each hour of day.

    If the clear-weather advantage is really a time-of-day artefact, it
    should shrink towards 1.0 once the hour is held constant.
    """
    tmp = df.assign(
        hour=df["date_time"].dt.hour,
        congested=df["traffic_volume"] > CONGESTION_THRESHOLD,
    )
    rows = []
    for hour, grp in tmp.groupby("hour"):
        clear = grp[grp["weather_main"] == "Clear"]["congested"]
        cloudy = grp[grp["weather_main"] == "Clouds"]["congested"]
        if len(clear) < 30 or len(cloudy) < 30:
            continue
        rows.append({
            "hour": int(hour),
            "n_clear": int(len(clear)),
            "n_cloudy": int(len(cloudy)),
            "rate_clear": float(clear.mean()),
            "rate_cloudy": float(cloudy.mean()),
            "rate_difference": float(clear.mean() - cloudy.mean()),
        })
    diffs = [r["rate_difference"] for r in rows]
    return {
        "per_hour": rows,
        "mean_rate_difference_within_hour": float(np.mean(diffs)) if diffs else float("nan"),
        "hours_compared": len(rows),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_pct(p: float) -> str:
    return f"{p:.4f} ({p * 100:.2f}%)"


def write_markdown(results: dict, path: Path) -> None:
    stats_raw = results["task2_1_descriptive"]["all_rows"]
    stats_hourly = results["task2_1_descriptive"]["hourly_deduplicated"]
    corr_raw = results["task2_2_correlation"]["all_rows_as_given"]
    corr_clean = results["task2_2_correlation"]["hourly_valid_temperature"]
    prob = results["task3_probability"]["all_rows"]
    prob_h = results["task3_probability"]["hourly_deduplicated"]

    lines: list[str] = []
    a = lines.append

    a("# Part 1 - Statistics and Probability Results\n")
    a("Generated by `part1_data_analytics/statistics_probability.py`.\n")
    a("## How the dataset is counted\n")
    a(f"- Raw CSV rows: **{results['dataset']['raw_rows']:,}**")
    a(f"- Distinct hourly timestamps: **{results['dataset']['distinct_hours']:,}**")
    a(f"- Duplicate-timestamp rows: **{results['dataset']['duplicate_rows']:,}** "
      "(the weather feed reports an hour more than once when several conditions "
      "are observed; the traffic count on those rows is identical)")
    a(f"- Impossible temperature readings (0 K): **{results['dataset']['zero_kelvin_rows']}**")
    a(f"- Impossible rainfall readings (> 9,000 mm): **{results['dataset']['extreme_rain_rows']}**\n")
    a("Every figure below is reported on two bases: **all rows as supplied** "
      "(the literal reading of the brief) and **de-duplicated hours** "
      "(one row per timestamp, most severe weather retained). Agreement "
      "between the two is what makes the conclusions safe to act on.\n")

    a("\n## Task 2.1 - Descriptive statistics for traffic volume\n")
    a("| Statistic | All rows | De-duplicated hours |")
    a("| --- | ---: | ---: |")
    for key, name in [
        ("count", "Count"), ("mean", "Mean"), ("median", "Median"),
        ("std_dev_sample", "Standard deviation (sample, ddof=1)"),
        ("variance_sample", "Variance (sample, ddof=1)"),
        ("std_dev_population", "Standard deviation (population)"),
        ("variance_population", "Variance (population)"),
        ("minimum", "Minimum"), ("maximum", "Maximum"), ("range", "Range"),
        ("q1", "25th percentile"), ("q3", "75th percentile"), ("iqr", "Interquartile range"),
        ("coefficient_of_variation", "Coefficient of variation"),
        ("skewness", "Skewness"), ("kurtosis_excess", "Excess kurtosis"),
    ]:
        a(f"| {name} | {stats_raw[key]:,.4f} | {stats_hourly[key]:,.4f} |")

    a("\n## Task 2.2 - Correlation between temperature and traffic volume\n")
    a("| Measure | All rows (temp as given) | De-duplicated hours, 0 K removed |")
    a("| --- | ---: | ---: |")
    a(f"| n | {corr_raw['n']:,} | {corr_clean['n']:,} |")
    a(f"| Pearson r | {corr_raw['pearson_r']:.4f} | {corr_clean['pearson_r']:.4f} |")
    a(f"| r-squared | {corr_raw['r_squared']:.4f} | {corr_clean['r_squared']:.4f} |")
    a(f"| p-value | {corr_raw['pearson_p_value']:.3e} | {corr_clean['pearson_p_value']:.3e} |")
    a(f"| Spearman rho | {corr_raw['spearman_rho']:.4f} | {corr_clean['spearman_rho']:.4f} |")
    a(f"| Direction | {corr_raw['direction']} | {corr_clean['direction']} |")
    a(f"| Strength | {corr_raw['strength']} | {corr_clean['strength']} |")

    a("\n## Task 3 - Probability and congestion\n")
    a(f"Congestion is defined as traffic volume > {CONGESTION_THRESHOLD:,} vehicles per hour. "
      f"High temperature is defined as > {HIGH_TEMP_THRESHOLD:.0f} K "
      f"({HIGH_TEMP_THRESHOLD - 273.15:.2f} C).\n")
    a("| Probability | All rows | De-duplicated hours |")
    a("| --- | ---: | ---: |")
    a(f"| P(Congestion) | {fmt_pct(prob['p_congestion'])} | {fmt_pct(prob_h['p_congestion'])} |")
    a(f"| P(Clear Weather) | {fmt_pct(prob['p_clear_weather'])} | {fmt_pct(prob_h['p_clear_weather'])} |")
    a(f"| P(Congestion AND Clear) | {fmt_pct(prob['p_congestion_and_clear'])} | {fmt_pct(prob_h['p_congestion_and_clear'])} |")
    a(f"| P(Clear \\| Congestion) | {fmt_pct(prob['p_clear_given_congestion'])} | {fmt_pct(prob_h['p_clear_given_congestion'])} |")
    a(f"| P(High Temp \\| Congestion) | {fmt_pct(prob['p_high_temp_given_congestion'])} | {fmt_pct(prob_h['p_high_temp_given_congestion'])} |")
    a(f"| P(Congestion \\| Clear) | {fmt_pct(prob['p_congestion_given_clear'])} | {fmt_pct(prob_h['p_congestion_given_clear'])} |")
    a(f"| P(Congestion \\| Cloudy) | {fmt_pct(prob['p_congestion_given_cloudy'])} | {fmt_pct(prob_h['p_congestion_given_cloudy'])} |")

    ind = prob["independence"]
    a("\n### Independence test: is congestion independent of clear weather?\n")
    a("Two events A and B are independent when P(A n B) = P(A) x P(B).\n")
    a(f"- Observed P(Congestion n Clear) = **{ind['p_a_and_b_observed']:.6f}**")
    a(f"- Expected under independence, P(Congestion) x P(Clear) = "
      f"{prob['p_congestion']:.6f} x {prob['p_clear_weather']:.6f} = "
      f"**{ind['p_a_times_p_b_expected']:.6f}**")
    a(f"- Absolute difference = **{ind['absolute_difference']:.6f}**")
    a(f"- Observed / expected = **{ind['observed_over_expected']:.4f}**")
    a(f"- Chi-square = {ind['chi_square']:.2f}, df = {ind['degrees_of_freedom']}, "
      f"p = {ind['chi_square_p_value']:.3e}, phi = {ind['phi_coefficient']:.4f}")
    verdict = "NOT independent" if not ind["independent"] else "independent"
    gap = ind["observed_over_expected"] - 1
    sense = "above" if gap > 0 else "below"
    a(f"\n**Conclusion: the two events are {verdict}.** The observed joint "
      f"probability sits {abs(gap):.1%} {sense} the independence expectation, so "
      f"congestion and clear weather co-occur {'more' if gap > 0 else 'less'} often "
      "than chance alone would produce. The chi-square test rejects independence "
      f"(p = {ind['chi_square_p_value']:.2e}), but the phi coefficient of "
      f"{ind['phi_coefficient']:.4f} shows the association is very weak: with roughly "
      "48,000 observations even a trivial departure from independence is "
      "statistically significant. Statistical significance here is a statement about "
      "sample size, not about practical importance.\n")

    orr = prob["odds_ratio_clear_vs_cloudy"]
    a("\n### Odds ratio: congestion in clear versus cloudy weather\n")
    c = prob["counts"]
    a("| | Congested | Not congested |")
    a("| --- | ---: | ---: |")
    a(f"| Clear | {c['congested_and_clear']:,} | {c['clear_not_congested']:,} |")
    a(f"| Cloudy | {c['congested_and_cloudy']:,} | {c['cloudy_not_congested']:,} |")
    a("")
    a(f"- Odds of congestion in clear weather = {c['congested_and_clear']:,} / "
      f"{c['clear_not_congested']:,} = **{orr['odds_congestion_clear']:.4f}**")
    a(f"- Odds of congestion in cloudy weather = {c['congested_and_cloudy']:,} / "
      f"{c['cloudy_not_congested']:,} = **{orr['odds_congestion_cloudy']:.4f}**")
    a(f"- **Odds ratio = {orr['odds_ratio']:.4f}** "
      f"(95% CI {orr['ci95_low']:.4f} to {orr['ci95_high']:.4f})")

    a("\n### Congestion rate by weather category\n")
    a("| Weather | Hours | Congested hours | Congestion rate | Average volume |")
    a("| --- | ---: | ---: | ---: | ---: |")
    for w, v in prob["congestion_rate_by_weather"].items():
        a(f"| {w} | {v['hours']:,} | {v['congested_hours']:,} | "
          f"{v['congestion_rate'] * 100:.2f}% | {v['avg_volume']:,.0f} |")

    hc = results["task3_probability"]["hour_controlled"]
    a("\n### Controlling for hour of day\n")
    a("The odds ratio above compares clear and cloudy hours without regard to "
      "WHEN they occurred. Holding hour of day constant and re-comparing:\n")
    a("| Hour | n clear | n cloudy | Congestion rate, clear | Congestion rate, cloudy | Difference |")
    a("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in hc["per_hour"]:
        a(f"| {r['hour']:02d} | {r['n_clear']:,} | {r['n_cloudy']:,} | "
          f"{r['rate_clear'] * 100:.2f}% | {r['rate_cloudy'] * 100:.2f}% | "
          f"{r['rate_difference'] * 100:+.2f} pp |")
    a(f"\nMean within-hour difference across the {hc['hours_compared']} hours compared: "
      f"**{hc['mean_rate_difference_within_hour'] * 100:+.2f} percentage points**.\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def main() -> int:
    configure_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        raw = load_raw()
    except (OSError, ValueError):
        logger.error("Could not load the raw CSV", exc_info=True)
        return 1

    hourly = collapse_to_hours(raw)
    hourly_valid_temp = hourly[hourly["temp"] > 100]

    results = {
        "dataset": {
            "raw_rows": int(len(raw)),
            "distinct_hours": int(raw["date_time"].nunique()),
            "duplicate_rows": int(len(raw) - raw["date_time"].nunique()),
            "zero_kelvin_rows": int((raw["temp"] < 100).sum()),
            "extreme_rain_rows": int((raw["rain_1h"] > 9000).sum()),
            "date_min": str(raw["date_time"].min()),
            "date_max": str(raw["date_time"].max()),
        },
        "task2_1_descriptive": {
            "all_rows": describe_traffic(raw["traffic_volume"]),
            "hourly_deduplicated": describe_traffic(hourly["traffic_volume"]),
        },
        "task2_2_correlation": {
            "all_rows_as_given": correlate(raw, "all rows, temperature as given"),
            "hourly_valid_temperature": correlate(
                hourly_valid_temp, "de-duplicated hours, 0 K readings removed"
            ),
        },
        "task3_probability": {
            "all_rows": probability_analysis(raw, "all rows"),
            "hourly_deduplicated": probability_analysis(hourly, "de-duplicated hours"),
            "hour_controlled": hour_controlled_odds(hourly),
            "congestion_by_hour": congestion_by_hour(hourly),
        },
    }

    try:
        (OUT_DIR / "statistics_probability_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        write_markdown(results, OUT_DIR / "statistics_probability_results.md")
    except OSError:
        logger.error("Could not write results", exc_info=True)
        return 1

    p = results["task3_probability"]["all_rows"]
    logger.info("P(Congestion)=%.4f  P(Clear)=%.4f  P(both)=%.4f  OR=%.3f",
                p["p_congestion"], p["p_clear_weather"], p["p_congestion_and_clear"],
                p["odds_ratio_clear_vs_cloudy"]["odds_ratio"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
