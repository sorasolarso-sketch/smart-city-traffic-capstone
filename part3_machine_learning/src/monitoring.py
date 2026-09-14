"""
Capstone Part 3, Task 6.4 and 6.5 — Model monitoring and alerting simulation.

Simulates a deployed model being monitored month by month across the
held-out period, checking two things:

* Feature distribution drift — Population Stability Index (PSI) and the
  Kolmogorov-Smirnov statistic for every numeric input, comparing each
  monitoring window against the training reference.
* Prediction error drift — the regressor's MAE in each window compared
  with its MAE on the training-period validation set.

Each window receives a status of PASS or ALERT, and the results are
written to a JSON report, a markdown dashboard, and a figure. A final
section injects synthetic drift to demonstrate that the alarm actually
fires when it should.

Usage
-----
    python part3_machine_learning/src/monitoring.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent.parent / "part2_python"))

import common  # noqa: E402
from logging_config import configure_logging, log_stage_banner  # noqa: E402
import viz_style  # noqa: E402

logger = logging.getLogger(__name__)

LOG_FILE = common.PART3_DIR / "logs" / "part3.log"
REGRESSOR_PATH = common.MODEL_DIR / "regressor_hist_gradient_boosting.joblib"

# Thresholds. PSI conventions from credit-risk model governance: < 0.10 no
# shift, 0.10-0.25 moderate, > 0.25 significant. The error threshold is a
# policy choice: a 40% rise in MAE over the validation baseline is enough
# to make a timing recommendation unreliable at peak hours.
PSI_WARN = 0.10
PSI_ALERT = 0.25
ERROR_ALERT_RATIO = 1.40
MIN_WINDOW_HOURS = 300

MONITORED_FEATURES = ["temp", "clouds_all", "rain_1h", "weather_severity",
                      "hour", "is_weekend", "adverse_conditions_score"]


# ---------------------------------------------------------------------------
# Drift statistics
# ---------------------------------------------------------------------------
def population_stability_index(reference: np.ndarray, current: np.ndarray,
                               bins: int = 10) -> float:
    """PSI between two samples, using quantile bins from the reference.

    PSI = sum over bins of (p_current - p_reference) * ln(p_current / p_reference)
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if np.unique(reference).size <= bins:
        # Discrete feature: one bin per observed value.
        values = np.union1d(np.unique(reference), np.unique(current))
        ref_counts = np.array([(reference == v).sum() for v in values], dtype=float)
        cur_counts = np.array([(current == v).sum() for v in values], dtype=float)
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
        cur_counts = np.histogram(current, bins=edges)[0].astype(float)

    eps = 1e-6
    ref_pct = np.clip(ref_counts / ref_counts.sum(), eps, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def calibrate_psi_thresholds(train: pd.DataFrame) -> dict:
    """Learn how much month-to-month PSI is normal for each feature.

    The textbook PSI cut-offs (0.10 / 0.25) were devised for credit
    scorecards, where inputs are slow-moving. Weather is not slow-moving:
    a single month of cloud cover compared with the pooled multi-year
    distribution for that calendar month routinely produces a PSI above
    0.25 through ordinary variability, and an alarm that fires every month
    is an alarm nobody reads.

    So each feature's alert threshold is set empirically: every month in
    the training period is scored against the same calendar month in the
    OTHER training years, and the threshold is the 95th percentile of that
    distribution, never lower than the conventional 0.25. Drift is then
    defined as "more change than this feature has ever shown between one
    year and the next", which is what an operator actually wants to know.
    """
    months = train["date_time"].dt.month
    years = train["date_time"].dt.year
    thresholds, history = {}, {}

    for feature in MONITORED_FEATURES:
        if feature not in train.columns:
            continue
        samples = []
        for (year, month), current in train.groupby([years, months]):
            if len(current) < MIN_WINDOW_HOURS:
                continue
            reference = train[(months == month) & (years != year)]
            if len(reference) < MIN_WINDOW_HOURS:
                continue
            samples.append(population_stability_index(reference[feature],
                                                      current[feature]))
        if samples:
            p95 = float(np.percentile(samples, 95))
            thresholds[feature] = max(PSI_ALERT, p95)
            history[feature] = {"n_months": len(samples),
                                "median_psi": float(np.median(samples)),
                                "p95_psi": p95}
        else:
            thresholds[feature] = PSI_ALERT
            history[feature] = {"n_months": 0}
        logger.debug("PSI calibration for '%s': %s", feature, history[feature])

    logger.info("PSI alert thresholds calibrated from %s training months: %s",
                history[MONITORED_FEATURES[0]]["n_months"],
                {k: round(v, 3) for k, v in thresholds.items()})
    return {"thresholds": thresholds, "history": history}


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  thresholds: dict | None = None) -> dict:
    thresholds = thresholds or {}
    report = {}
    for feature in MONITORED_FEATURES:
        if feature not in reference.columns:
            continue
        psi = population_stability_index(reference[feature], current[feature])
        ks_stat, ks_p = stats.ks_2samp(reference[feature], current[feature])
        alert_at = thresholds.get(feature, PSI_ALERT)
        warn_at = max(PSI_WARN, 0.6 * alert_at)
        status = "ALERT" if psi > alert_at else ("WARN" if psi > warn_at else "PASS")
        report[feature] = {
            "psi": round(psi, 4),
            "alert_threshold": round(alert_at, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": float(ks_p),
            "reference_mean": float(reference[feature].mean()),
            "current_mean": float(current[feature].mean()),
            "status": status,
        }
    return report


# ---------------------------------------------------------------------------
# Monitoring run
# ---------------------------------------------------------------------------
def evaluate_window(model, features: list[str], window: pd.DataFrame,
                    reference: pd.DataFrame, baseline_mae: float,
                    label: str, psi_thresholds: dict | None = None) -> dict:
    preds = model.predict(window[features])
    mae = float(mean_absolute_error(window["traffic_volume"], preds))
    error_ratio = mae / baseline_mae
    bias = float(np.mean(preds - window["traffic_volume"]))

    drift = feature_drift(reference, window, psi_thresholds)
    drifted = [f for f, r in drift.items() if r["status"] == "ALERT"]
    warned = [f for f, r in drift.items() if r["status"] == "WARN"]

    reasons = []
    if error_ratio > ERROR_ALERT_RATIO:
        reasons.append(f"MAE {mae:.0f} is {error_ratio:.2f}x the baseline {baseline_mae:.0f}")
    if drifted:
        reasons.append("significant feature drift in: " + ", ".join(drifted))
    status = "ALERT" if reasons else "PASS"

    if status == "ALERT":
        logger.warning("Monitoring window %s — ALERT: %s", label, "; ".join(reasons))
    else:
        logger.info("Monitoring window %s — PASS (MAE %.0f, ratio %.2f, max PSI %.3f)",
                    label, mae, error_ratio,
                    max((r["psi"] for r in drift.values()), default=0.0))
    if warned and status == "PASS":
        logger.info("  moderate drift (warn) in: %s", ", ".join(warned))

    return {
        "window": label,
        "hours": int(len(window)),
        "mae": round(mae, 2),
        "baseline_mae": round(baseline_mae, 2),
        "error_ratio": round(error_ratio, 4),
        "mean_bias": round(bias, 2),
        "status": status,
        "reasons": reasons,
        "feature_drift": drift,
        "features_alert": drifted,
        "features_warn": warned,
    }


def build_error_baselines(model, train: pd.DataFrame, features: list[str]) -> dict:
    """Establish a season-aware, out-of-sample error baseline.

    Traffic error is strongly seasonal — winter months are harder to
    predict than summer ones — so a single baseline number taken from a
    summer validation slice would flag every winter month as degraded, and
    a baseline taken in-sample would be optimistic for a boosted model.

    The fix is to hold out the final 365 days of the training period,
    refit an identical model on everything before it, and record that
    model's MAE for each calendar month. The production model is then
    judged against the baseline for the SAME calendar month.
    """
    from sklearn.base import clone

    cutoff = train["date_time"].max() - pd.Timedelta(days=365)
    fit_part = train[train["date_time"] <= cutoff]
    val_part = train[train["date_time"] > cutoff]

    baseline_model = clone(model)
    baseline_model.fit(fit_part[features], fit_part["traffic_volume"])
    preds = baseline_model.predict(val_part[features])
    errors = np.abs(preds - val_part["traffic_volume"].to_numpy())

    overall = float(errors.mean())
    by_month = (pd.Series(errors, index=val_part["date_time"].dt.month.values)
                  .groupby(level=0).mean())
    logger.info(
        "Error baseline from a season-spanning validation year (%s to %s, "
        "%s hours): overall MAE %.1f; monthly range %.1f (month %02d) to %.1f "
        "(month %02d)", val_part["date_time"].min().date(),
        val_part["date_time"].max().date(), f"{len(val_part):,}", overall,
        by_month.min(), int(by_month.idxmin()), by_month.max(), int(by_month.idxmax()),
    )
    for month, mae in by_month.items():
        logger.debug("Baseline MAE for calendar month %02d: %.1f", int(month), mae)

    return {"overall": overall,
            "by_month": {int(m): float(v) for m, v in by_month.items()}}


def run_monitoring() -> dict:
    log_stage_banner(logger, "Task 6.4 / 6.5 — Monitoring and alerting simulation")

    model = joblib.load(REGRESSOR_PATH)
    features = list(model.feature_names_in_)
    logger.info("Loaded regressor from %s (%s features)", REGRESSOR_PATH.name,
                len(features))

    df = common.load_features()
    train, test = common.chronological_split(df)

    baselines = build_error_baselines(model, train, features)
    calibration = calibrate_psi_thresholds(train)
    psi_thresholds = calibration["thresholds"]
    logger.info("Alert policy — error ratio > %.2f against the same-calendar-month "
                "baseline, or any feature PSI > %.2f against the same-calendar-month "
                "training reference", ERROR_ALERT_RATIO, PSI_ALERT)

    # ---- Monitor each calendar month of the held-out period -------------
    # Each month is compared with the training rows from the SAME calendar
    # month. Comparing January against a five-year all-season reference
    # would flag temperature drift every winter, which is seasonality, not
    # drift, and would render the alarm meaningless.
    test = test.assign(period=test["date_time"].dt.to_period("M").astype(str))
    train_month = train["date_time"].dt.month
    windows = []
    for period, window in test.groupby("period", sort=True):
        if len(window) < MIN_WINDOW_HOURS:
            logger.warning("Window %s skipped — only %s hours (< %s minimum)",
                           period, len(window), MIN_WINDOW_HOURS)
            continue
        month = int(window["date_time"].dt.month.iloc[0])
        reference = train[train_month == month]
        baseline_mae = baselines["by_month"].get(month, baselines["overall"])
        windows.append(evaluate_window(model, features, window, reference,
                                       baseline_mae, period, psi_thresholds))

    alerts = [w for w in windows if w["status"] == "ALERT"]
    logger.info("Real-data monitoring complete — %s window(s), %s ALERT, %s PASS",
                len(windows), len(alerts), len(windows) - len(alerts))

    # ---- Synthetic drift: prove the alarm fires ----------------------------
    log_stage_banner(logger, "Synthetic drift scenarios")
    scenarios = []
    last = test[test["period"] == windows[-1]["window"]].copy() if windows else test.tail(720).copy()

    sensor_fault = last.copy()
    sensor_fault["temp"] = sensor_fault["temp"] + 12.0       # +12 K bias
    sensor_fault["is_freezing"] = (sensor_fault["temp"] < 273.15).astype(int)
    sensor_fault["is_hot"] = (sensor_fault["temp"] > 300).astype(int)
    scenarios.append(("synthetic: temperature sensor +12 K bias", sensor_fault))

    demand_shift = last.copy()
    demand_shift["traffic_volume"] = (demand_shift["traffic_volume"] * 1.35).round()
    scenarios.append(("synthetic: demand +35% (new development opens)", demand_shift))

    schedule_shift = last.copy()
    schedule_shift["traffic_volume"] = np.roll(schedule_shift["traffic_volume"].values, 2)
    scenarios.append(("synthetic: peak shifts 2 hours later", schedule_shift))

    last_month = int(last["date_time"].dt.month.iloc[0])
    last_reference = train[train_month == last_month]
    last_baseline = baselines["by_month"].get(last_month, baselines["overall"])
    synthetic = []
    for label, frame in scenarios:
        synthetic.append(evaluate_window(model, features, frame, last_reference,
                                         last_baseline, label, psi_thresholds))
    fired = sum(1 for s in synthetic if s["status"] == "ALERT")
    if fired == len(synthetic):
        logger.info("All %s synthetic drift scenarios raised an ALERT — the "
                    "detector responds to both feature drift and error drift", fired)
    else:
        logger.warning("Only %s of %s synthetic scenarios raised an ALERT; "
                       "thresholds may be too lenient", fired, len(synthetic))

    overall = "ALERT" if alerts else "PASS"
    summary = {
        "overall_status": overall,
        "policy": {
            "error_alert_ratio": ERROR_ALERT_RATIO,
            "psi_warn": PSI_WARN, "psi_alert": PSI_ALERT,
            "min_window_hours": MIN_WINDOW_HOURS,
            "monitored_features": MONITORED_FEATURES,
            "psi_alert_thresholds_calibrated": {k: round(v, 4) for k, v in psi_thresholds.items()},
            "psi_calibration_history": calibration["history"],
        },
        "baseline_mae": round(baselines["overall"], 2),
        "baseline_mae_by_calendar_month": {str(k): round(v, 2)
                                           for k, v in baselines["by_month"].items()},
        "model": REGRESSOR_PATH.name,
        "windows_evaluated": len(windows),
        "windows_alert": len(alerts),
        "windows_pass": len(windows) - len(alerts),
        "real_data_windows": windows,
        "synthetic_scenarios": synthetic,
    }
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_dashboard(summary: dict, path: Path) -> None:
    lines = [
        "# Model Monitoring Dashboard\n",
        f"**Overall status: {'🔴 ALERT — requires investigation' if summary['overall_status'] == 'ALERT' else '🟢 PASS — normal'}**\n",
        f"Model under watch: `{summary['model']}`  ",
        f"Baseline MAE (season-spanning validation year, compared month-for-month): **{summary['baseline_mae']:.1f} vehicles/hour overall**  ",
        f"Alert policy: error ratio > {summary['policy']['error_alert_ratio']}, "
        f"or any feature's PSI above its calibrated threshold (never below "
        f"{summary['policy']['psi_alert']})\n",
        "## Monthly windows on held-out data\n",
        "| Window | Hours | MAE | Ratio to baseline | Mean bias | Worst drift: PSI ÷ threshold (feature) | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for w in summary["real_data_windows"]:
        worst = max(w["feature_drift"].items(),
                    key=lambda kv: kv[1]["psi"] / kv[1]["alert_threshold"])
        icon = "🔴 ALERT" if w["status"] == "ALERT" else "🟢 PASS"
        lines.append(
            f"| {w['window']} | {w['hours']:,} | {w['mae']:.0f} | {w['error_ratio']:.2f} | "
            f"{w['mean_bias']:+.0f} | {worst[1]['psi'] / worst[1]['alert_threshold']:.2f} ({worst[0]}) | {icon} |"
        )

    lines += ["\n## Synthetic drift scenarios (alarm test)\n",
              "| Scenario | MAE | Ratio | Drifted features | Status |",
              "| --- | ---: | ---: | --- | --- |"]
    for s in summary["synthetic_scenarios"]:
        icon = "🔴 ALERT" if s["status"] == "ALERT" else "🟢 PASS"
        lines.append(
            f"| {s['window'].replace('synthetic: ', '')} | {s['mae']:.0f} | "
            f"{s['error_ratio']:.2f} | {', '.join(s['features_alert']) or '—'} | {icon} |"
        )

    lines += ["\n## How to read this\n",
              "- **PSI** measures how far a feature's distribution has moved from the "
              "training rows for the same calendar month. Because weather varies a lot "
              "from one year's March to the next, each feature's alert threshold is "
              "calibrated to the 95th percentile of the month-to-month PSI observed "
              "within the training years (never below the conventional 0.25). A "
              "drift ratio above 1.0 therefore means *more change than this feature "
              "has ever shown between years*.",
              "- **Ratio to baseline** compares this month's prediction error with the "
              "error the model achieved on its own validation slice. A ratio above 1.40 "
              "means predictions have degraded enough that peak-hour recommendations can "
              "no longer be trusted.",
              "- A window is **ALERT** if either check fails. The response is "
              "investigation, not automatic retraining: a drift alert on `temp` in a "
              "heatwave is expected; the same alert in January is a broken sensor.",
              f"\n_Generated by `part3_machine_learning/src/monitoring.py`._"]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Monitoring dashboard written to %s", path)


def plot_monitoring(summary: dict) -> None:
    windows = summary["real_data_windows"]
    if not windows:
        return
    labels = [w["window"] for w in windows]
    ratios = [w["error_ratio"] for w in windows]
    # Drift is plotted relative to each feature's calibrated threshold, so a
    # value of 1.0 means "at the alert line" regardless of the feature.
    drift_ratio = [max(r["psi"] / r["alert_threshold"] for r in w["feature_drift"].values())
                   for w in windows]
    colors = [viz_style.STATUS["critical"] if w["status"] == "ALERT"
              else viz_style.SERIES[0] for w in windows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].bar(labels, ratios, color=colors, width=0.7)
    axes[0].axhline(ERROR_ALERT_RATIO, color=viz_style.STATUS["critical"],
                    linestyle="--", linewidth=1.4, label=f"alert at {ERROR_ALERT_RATIO}x")
    axes[0].axhline(1.0, color=viz_style.TEXT_MUTED, linestyle=":", linewidth=1.2,
                    label="baseline")
    axes[0].set_title("Prediction error drift")
    axes[0].set_ylabel("MAE ÷ same-month baseline MAE")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(loc="upper right")

    axes[1].bar(labels, drift_ratio, color=colors, width=0.7)
    axes[1].axhline(1.0, color=viz_style.STATUS["critical"], linestyle="--",
                    linewidth=1.4, label="alert threshold (calibrated per feature)")
    axes[1].axhline(0.6, color=viz_style.STATUS["warning"], linestyle="--",
                    linewidth=1.2, label="warn")
    axes[1].set_title("Feature distribution drift (worst feature)")
    axes[1].set_ylabel("PSI ÷ calibrated alert threshold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(loc="upper right")

    n_alert = summary["windows_alert"]
    fig.suptitle(
        f"Monitoring across the held-out period — {n_alert} of "
        f"{summary['windows_evaluated']} months raised an ALERT",
        x=0.005, ha="left", fontsize=13, fontweight="bold", y=1.12,
    )
    path = common.FIGURE_DIR / "17_monitoring_dashboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate model monitoring and alerting.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(log_file=LOG_FILE, debug=args.debug)
    viz_style.apply_style()
    common.ensure_directories()

    logger.info("#" * 72)
    logger.info("Capstone Part 3, Task 6 — monitoring and alerting")
    logger.info("#" * 72)

    try:
        summary = run_monitoring()
        (common.OUTPUT_DIR / "monitoring_report.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Monitoring report written to %s",
                    common.OUTPUT_DIR / "monitoring_report.json")
        write_dashboard(summary, common.OUTPUT_DIR / "MONITORING_DASHBOARD.md")
        plot_monitoring(summary)
    except FileNotFoundError:
        logger.error("Monitoring aborted — run supervised.py first to create the model",
                     exc_info=True)
        return 3
    except (OSError, ValueError, KeyError, TypeError):
        logger.error("Monitoring aborted — unexpected problem", exc_info=True)
        return 5

    logger.info("Monitoring stage finished — overall status %s", summary["overall_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
