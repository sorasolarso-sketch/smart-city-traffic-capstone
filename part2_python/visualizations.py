"""
Capstone Part 2, Task 3 — Traffic pattern visualisations.

Produces six Matplotlib figures, each saved to ``figures/`` and each
accompanied by a written interpretation in ``figures/INTERPRETATIONS.md``.

Usage
-----
    python part2_python/visualizations.py
    python part2_python/visualizations.py --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend; no display server required

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logging_config import configure_logging, log_stage_banner  # noqa: E402
import viz_style  # noqa: E402

logger = logging.getLogger(__name__)

PART2_DIR = Path(__file__).resolve().parent
REPO_ROOT = PART2_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "traffic_features.csv"
FIGURE_DIR = PART2_DIR / "figures"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

# Every figure appends an entry here, which is written out at the end.
INTERPRETATIONS: list[tuple[str, str, str]] = []


def thousands(x, _pos) -> str:
    return f"{x:,.0f}"


def save_figure(fig, name: str, title: str, interpretation: str) -> Path:
    """Save a figure and record its interpretation.

    Logs an INFO message including the file path on success, which is the
    confirmation trail the brief requires.
    """
    path = FIGURE_DIR / name
    try:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    except OSError:
        logger.error("Failed to save figure to %s", path, exc_info=True)
        raise
    finally:
        plt.close(fig)

    size_kb = path.stat().st_size / 1024
    logger.info("Figure saved: %s (%.0f KB)", path, size_kb)
    INTERPRETATIONS.append((name, title, interpretation))
    return path


# ---------------------------------------------------------------------------
# Figure 1 — Hourly demand, weekday versus weekend
# ---------------------------------------------------------------------------
def fig_hourly_weekday_weekend(df: pd.DataFrame) -> None:
    weekday = df[df["is_weekend"] == 0].groupby("hour")["traffic_volume"].mean()
    weekend = df[df["is_weekend"] == 1].groupby("hour")["traffic_volume"].mean()

    fig, ax = plt.subplots(figsize=viz_style.FIGSIZE_WIDE)
    ax.plot(weekday.index, weekday.values, color=viz_style.SERIES[0],
            label="Weekday", marker="o", markersize=4)
    ax.plot(weekend.index, weekend.values, color=viz_style.SERIES[1],
            label="Weekend", marker="o", markersize=4)

    # Direct labels on the two peaks rather than a number on every point.
    wd_peak = int(weekday.idxmax())
    we_peak = int(weekend.idxmax())
    ax.annotate(f"Weekday peak  {weekday.max():,.0f} at {wd_peak:02d}:00",
                xy=(wd_peak, weekday.max()), xytext=(0, 15),
                textcoords="offset points", ha="center",
                color=viz_style.TEXT_SECONDARY, fontsize=9)
    ax.annotate(f"Weekend peak  {weekend.max():,.0f} at {we_peak:02d}:00",
                xy=(we_peak, weekend.max()), xytext=(10, -26),
                textcoords="offset points", ha="left",
                color=viz_style.TEXT_SECONDARY, fontsize=9)

    ax.set_title("Weekend traffic has one broad afternoon peak, not two commuter peaks")
    viz_style.add_subtitle(ax, "Average westbound I-94 volume by hour of day, 2012–2018")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average vehicles per hour")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_ylim(0, weekday.max() * 1.20)
    ax.legend(loc="upper left")
    viz_style.add_source_note(fig, "Source: Metro Interstate Traffic Volume, cleaned (40,575 hours)")

    ratio = weekday.max() / weekend.max()
    wd_morning_hour = int(weekday.loc[5:10].idxmax())
    wd_morning_value = weekday.loc[5:10].max()
    wd_trough_hour = int(weekday.loc[9:13].idxmin())
    wd_trough_value = weekday.loc[9:13].min()
    interpretation = (
        f"The two curves have fundamentally different shapes. Weekdays show the "
        f"classic twin-peak commuter profile: volume climbs steeply from 04:00 to a "
        f"morning peak of {wd_morning_value:,.0f} vehicles/hour at "
        f"{wd_morning_hour:02d}:00, falls back to {wd_trough_value:,.0f} at "
        f"{wd_trough_hour:02d}:00, then builds again to the true daily maximum of "
        f"{weekday.max():,.0f} at {wd_peak:02d}:00. The afternoon peak is the larger "
        f"of the two, by {100 * (weekday.max() / wd_morning_value - 1):.0f}%, which is "
        f"worth stating plainly because traffic policy conventionally treats the "
        f"morning commute as the binding constraint. On this corridor it is not. "
        f"Weekends have no morning peak whatsoever: volume rises gradually to a single "
        f"broad plateau of {weekend.max():,.0f} vehicles/hour around {we_peak:02d}:00, "
        f"{ratio:.2f}x below the weekday maximum, and decays slowly through the "
        f"evening. The practical consequence is that weekday and weekend demand cannot "
        f"be served by one timing policy — any model that ignores day type will be "
        f"systematically wrong on two days in seven. This is why 'is_weekend' and the "
        f"cyclical hour encoding are both in the feature set."
    )
    save_figure(fig, "01_hourly_weekday_vs_weekend.png",
                "Hourly demand: weekday versus weekend", interpretation)


# ---------------------------------------------------------------------------
# Figure 2 — Traffic volume distribution with the quartile cut-points
# ---------------------------------------------------------------------------
def fig_distribution(df: pd.DataFrame) -> None:
    volumes = df["traffic_volume"]
    q1, q2, q3 = volumes.quantile([0.25, 0.50, 0.75])

    fig, ax = plt.subplots(figsize=viz_style.FIGSIZE_WIDE)
    ax.hist(volumes, bins=60, color=viz_style.SERIES[0], edgecolor=viz_style.SURFACE,
            linewidth=0.6)

    for value, label in [(q1, "Q1"), (q2, "Median"), (q3, "Q3")]:
        ax.axvline(value, color=viz_style.TEXT_SECONDARY, linestyle="--", linewidth=1.2)
        ax.text(value, ax.get_ylim()[1] * 0.95, f" {label} {value:,.0f}",
                color=viz_style.TEXT_SECONDARY, fontsize=9, va="top")

    ax.set_title("Traffic volume is bimodal, not bell-shaped")
    viz_style.add_subtitle(
        ax, "Distribution of hourly volume, with the quartile cut-points used for the congestion target"
    )
    ax.set_xlabel("Vehicles per hour")
    ax.set_ylabel("Number of hours")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    viz_style.add_source_note(fig, "Source: Metro Interstate Traffic Volume, cleaned (40,575 hours)")

    low_mode = int(volumes[volumes < 1500].mode().iloc[0]) if (volumes < 1500).any() else 0
    interpretation = (
        f"The distribution has two distinct masses rather than a single central "
        f"tendency: a tall spike of near-empty overnight hours below roughly 1,000 "
        f"vehicles, and a broad daytime mass between about 4,000 and 6,000. Almost "
        f"nothing sits in between. This matters more than it first appears. The mean "
        f"({volumes.mean():,.0f}) and the median ({volumes.median():,.0f}) both fall in "
        f"the sparse valley between the two modes, so neither describes a typical hour "
        f"— there is no typical hour on this corridor, only typical nights and typical "
        f"days. Reporting 'average traffic' to a stakeholder without this caveat is "
        f"actively misleading. It also explains the negative excess kurtosis of -1.31 "
        f"found in Part 1: the distribution is flatter than normal because its mass "
        f"sits at the two ends. The dashed lines show the quartile cut-points "
        f"(Q1 {q1:,.0f}, median {q2:,.0f}, Q3 {q3:,.0f}) that define the four-level "
        f"congestion target used in Part 3; splitting on quartiles rather than fixed "
        f"thresholds guarantees four balanced classes despite this awkward shape."
    )
    save_figure(fig, "02_traffic_volume_distribution.png",
                "Distribution of hourly traffic volume", interpretation)


# ---------------------------------------------------------------------------
# Figure 3 — Hour x day-of-week heatmap
# ---------------------------------------------------------------------------
def fig_heatmap(df: pd.DataFrame) -> None:
    pivot = (df.pivot_table(index="day_name", columns="hour",
                            values="traffic_volume", aggfunc="mean")
               .reindex(DAY_ORDER))

    fig, ax = plt.subplots(figsize=(12, 4.6))
    mesh = ax.imshow(pivot.values, aspect="auto", cmap=viz_style.SEQUENTIAL_CMAP,
                     interpolation="nearest")

    peak_day = pivot.max(axis=1).idxmax()
    peak_hour = int(pivot.loc[peak_day].idxmax())
    peak_value = pivot.loc[peak_day, peak_hour]

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_yticks(range(len(DAY_ORDER)))
    ax.set_yticklabels(DAY_ORDER, fontsize=9)
    ax.set_title(
        f"The busiest hour of the week is {peak_day} at {peak_hour:02d}:00"
    )
    viz_style.add_subtitle(ax, "Average vehicles per hour, by day of week and hour of day")
    ax.set_xlabel("Hour of day")
    ax.grid(False)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.03)
    cbar.set_label("Average vehicles per hour", color=viz_style.TEXT_SECONDARY, fontsize=9)
    cbar.ax.tick_params(labelsize=8, colors=viz_style.TEXT_MUTED)
    cbar.outline.set_visible(False)

    # Ring the single busiest cell rather than labelling every cell.
    ax.add_patch(plt.Rectangle(
        (peak_hour - 0.5, DAY_ORDER.index(peak_day) - 0.5), 1, 1,
        fill=False, edgecolor=viz_style.SERIES[1], linewidth=2.5,
    ))
    viz_style.add_source_note(fig, "Source: Metro Interstate Traffic Volume, cleaned (40,575 hours)")

    sat_peak = pivot.loc["Saturday"].max()
    interpretation = (
        f"The heatmap separates two effects the hourly average conflates. Reading "
        f"across, Monday to Friday share an almost identical signature — a firm "
        f"morning block at 06:00–08:00 and a wider, darker afternoon block from "
        f"14:00 to 18:00. Reading down, the weekend rows lose the morning block "
        f"entirely and shift their mass into the middle of the day. The single "
        f"busiest cell of the week is {peak_day} at {peak_hour:02d}:00, averaging "
        f"{peak_value:,.0f} vehicles/hour, which is {peak_value / sat_peak:.2f}x the "
        f"Saturday maximum of {sat_peak:,.0f}. {peak_day} afternoon is therefore the "
        f"correct target for any intervention aimed at the worst hour of the week, "
        f"rather than the Monday morning commute that traffic policy conventionally "
        f"focuses on. Note also the faint vertical band at 05:00 across all seven "
        f"rows: the corridor wakes up at the same time regardless of day type, but "
        f"only fills up on weekdays."
    )
    save_figure(fig, "03_hour_day_heatmap.png",
                "Traffic intensity by hour and day of week", interpretation)


# ---------------------------------------------------------------------------
# Figure 4 — Weather condition versus traffic, with exposure shown
# ---------------------------------------------------------------------------
def fig_weather(df: pd.DataFrame) -> None:
    grouped = (df.groupby("weather_main")
                 .agg(avg=("traffic_volume", "mean"), hours=("traffic_volume", "size"))
                 .sort_values("avg"))
    reliable = grouped["hours"] >= 100

    fig, ax = plt.subplots(figsize=viz_style.FIGSIZE_WIDE)
    colors = [viz_style.SERIES[0] if ok else viz_style.TEXT_MUTED for ok in reliable]
    bars = ax.barh(grouped.index, grouped["avg"], color=colors, height=0.68)

    for bar, (name, row) in zip(bars, grouped.iterrows()):
        ax.text(bar.get_width() + 55, bar.get_y() + bar.get_height() / 2,
                f"{row['avg']:,.0f}   n={int(row['hours']):,}",
                va="center", fontsize=8.5, color=viz_style.TEXT_SECONDARY)

    ax.set_title("Weather ranking is driven by when conditions occur, not by driver behaviour")
    viz_style.add_subtitle(
        ax, "Average traffic by weather condition; grey bars rest on fewer than 100 hours"
    )
    ax.set_xlabel("Average vehicles per hour")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.set_xlim(0, grouped["avg"].max() * 1.28)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=viz_style.SERIES[0],
               markersize=9, label="100+ hours observed"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=viz_style.TEXT_MUTED,
               markersize=9, label="Fewer than 100 hours — not interpretable"),
    ]
    ax.legend(handles=legend, loc="lower right")
    viz_style.add_source_note(fig, "Source: Metro Interstate Traffic Volume, cleaned (40,575 hours)")

    top = grouped[reliable].index[-1]
    bottom = grouped[reliable].index[0]
    gap = grouped.loc[top, "avg"] - grouped.loc[bottom, "avg"]
    interpretation = (
        f"Among conditions with enough observations to interpret, {top} carries the "
        f"highest average traffic at {grouped.loc[top, 'avg']:,.0f} vehicles/hour and "
        f"{bottom} the lowest at {grouped.loc[bottom, 'avg']:,.0f}, a gap of "
        f"{gap:,.0f}. It is tempting to read this as drivers responding to weather. "
        f"That reading is wrong, and the chart is deliberately labelled to resist it. "
        f"Cloud is the default daytime condition in Minneapolis while clear skies are "
        f"disproportionately nocturnal, so this ranking largely encodes what time of "
        f"day each condition tends to occur. Part 1 demonstrates this directly: the "
        f"raw odds ratio makes congestion look 26% less likely in clear weather, yet "
        f"holding hour of day constant shrinks the clear-versus-cloudy difference to "
        f"+0.03 percentage points. The greyed bars carry the second lesson — Squall "
        f"rests on a single observed hour, so its apparent position at the bottom of "
        f"any ranking is an artefact of sample size, not a finding."
    )
    save_figure(fig, "04_weather_impact.png",
                "Average traffic by weather condition", interpretation)


# ---------------------------------------------------------------------------
# Figure 5 — Temperature versus traffic, faceted by part of day
# ---------------------------------------------------------------------------
def fig_temp_scatter(df: pd.DataFrame) -> None:
    order = ["Night", "Morning Peak", "Midday", "Evening Peak", "Evening"]
    present = [p for p in order if p in set(df["part_of_day"])]

    fig, axes = plt.subplots(1, len(present), figsize=(15, 3.9), sharey=True, sharex=True)
    if len(present) == 1:
        axes = [axes]

    sample = df.sample(n=min(9000, len(df)), random_state=42)
    overall_r = df["temp_celsius"].corr(df["traffic_volume"])

    for ax, band in zip(axes, present):
        subset = sample[sample["part_of_day"] == band]
        ax.scatter(subset["temp_celsius"], subset["traffic_volume"],
                   s=5, alpha=0.22, color=viz_style.SERIES[0], linewidths=0)
        band_r = df.loc[df["part_of_day"] == band, "temp_celsius"].corr(
            df.loc[df["part_of_day"] == band, "traffic_volume"])
        ax.set_title(f"{band}\nr = {band_r:+.3f}", fontsize=10.5, pad=8)
        ax.set_xlabel("Temperature (°C)")
        ax.grid(axis="both")
    axes[0].set_ylabel("Vehicles per hour")
    axes[0].yaxis.set_major_formatter(FuncFormatter(thousands))

    fig.suptitle(
        "Splitting by time of day dissolves the temperature–traffic relationship",
        x=0.005, ha="left", fontsize=13, fontweight="bold", y=1.10,
    )
    fig.text(0.005, 1.02,
             f"Overall Pearson r = {overall_r:+.3f}; each panel shows the same relationship within one time band",
             ha="left", fontsize=9.5, color=viz_style.TEXT_SECONDARY)
    viz_style.add_source_note(fig, "Source: Metro Interstate Traffic Volume, cleaned; 9,000-point sample per panel")

    interpretation = (
        f"Pooled across all hours, temperature and traffic correlate at "
        f"r = {overall_r:+.3f} — weak, but positive and highly significant given "
        f"40,575 observations. Faceting by time of day shows why that number should "
        f"not be trusted as a behavioural finding. Each panel is a horizontal cloud: "
        f"within any given time band, knowing the temperature tells you very little "
        f"about how many vehicles are on the road. The pooled correlation arises "
        f"almost entirely from the fact that warm hours and busy hours are both "
        f"daytime summer hours. This is a textbook illustration of why correlation "
        f"does not imply causation, and a concrete warning for the models in Part 3: "
        f"temperature will appear in the feature set, but any importance it receives "
        f"should be read as a proxy for season and daylight rather than as evidence "
        f"that drivers respond to the thermometer."
    )
    save_figure(fig, "05_temperature_vs_traffic_by_daypart.png",
                "Temperature versus traffic, by part of day", interpretation)


# ---------------------------------------------------------------------------
# Figure 6 — Congestion over time, with the data gaps made visible
# ---------------------------------------------------------------------------
def fig_congestion_over_time(df: pd.DataFrame) -> None:
    monthly = (df.set_index("date_time")
                 .resample("MS")
                 .agg(congestion_rate=("is_congested", "mean"),
                      hours=("is_congested", "size")))
    # An incomplete month cannot support a rate; show it as a gap instead.
    incomplete = monthly["hours"] < 400
    plotted = monthly["congestion_rate"].mask(incomplete) * 100

    fig, ax = plt.subplots(figsize=viz_style.FIGSIZE_WIDE)
    ax.plot(plotted.index, plotted.values, color=viz_style.SERIES[0], linewidth=2)

    for idx in monthly.index[incomplete]:
        ax.axvspan(idx, idx + pd.offsets.MonthEnd(1), color=viz_style.GRID, zorder=0)

    ax.set_title("Congestion is stable over six years; the gaps are sensor outages, not quiet months")
    viz_style.add_subtitle(
        ax, "Share of hours above 5,500 vehicles, by month. Shaded bands have under 400 hours recorded."
    )
    ax.set_xlabel("")
    ax.set_ylabel("Congested hours (%)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, max(plotted.max() * 1.25, 1))

    legend = [
        Line2D([0], [0], color=viz_style.SERIES[0], linewidth=2, label="Monthly congestion rate"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=viz_style.GRID,
               markersize=11, label="Insufficient data (< 400 hours)"),
    ]
    ax.legend(handles=legend, loc="lower right")
    viz_style.add_source_note(fig, "Source: Metro Interstate Traffic Volume, cleaned (40,575 hours)")

    valid = monthly.loc[~incomplete, "congestion_rate"] * 100
    interpretation = (
        f"Across the months with adequate coverage the congestion rate sits in a "
        f"narrow band between {valid.min():.1f}% and {valid.max():.1f}% of hours, with "
        f"no visible trend over six years: this corridor's congestion problem is "
        f"stable and structural rather than growing. The shaded bands are the reason "
        f"this figure exists in its current form. {int(incomplete.sum())} of "
        f"{len(monthly)} months have fewer than 400 recorded hours, and an earlier "
        f"draft that plotted them as ordinary points produced a dramatic-looking "
        f"collapse through 2014 and 2015 that was entirely an artefact of sensor "
        f"downtime. Masking them and shading the gap is the honest presentation. Any "
        f"dashboard built for the mobility team should adopt the same convention, "
        f"because a stakeholder who sees an unmarked dip will reasonably conclude "
        f"that demand fell."
    )
    save_figure(fig, "06_congestion_rate_over_time.png",
                "Monthly congestion rate, 2012–2018", interpretation)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def write_interpretations(path: Path) -> None:
    lines = [
        "# Part 2, Task 3 — Figure interpretations\n",
        "Generated by `part2_python/visualizations.py`. Each figure below is "
        "saved in this directory.\n",
    ]
    for i, (filename, title, text) in enumerate(INTERPRETATIONS, start=1):
        lines.append(f"\n## Figure {i} — {title}\n")
        lines.append(f"![{title}]({filename})\n")
        lines.append(f"**File:** `{filename}`\n")
        lines.append(f"\n{text}\n")
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        logger.error("Could not write interpretations to %s", path, exc_info=True)
        raise
    logger.info("Figure interpretations written to %s", path)


def load_features(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["date_time"], keep_default_na=False)
    except FileNotFoundError:
        logger.error("Feature file not found at %s — run feature_engineering.py first",
                     path, exc_info=True)
        raise
    except (OSError, ValueError, pd.errors.ParserError):
        logger.error("Could not read features from %s", path, exc_info=True)
        raise
    logger.info("Feature data loaded from %s — %s rows, %s columns",
                path.name, f"{len(df):,}", df.shape[1])
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Part 2 figures.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(log_file=PART2_DIR / "logs" / "pipeline.log", debug=args.debug)
    viz_style.apply_style()

    logger.info("#" * 72)
    logger.info("Capstone Part 2 — visualisation stage starting")
    logger.info("#" * 72)
    log_stage_banner(logger, "Visualisations")

    try:
        df = load_features(args.input)
        fig_hourly_weekday_weekend(df)
        fig_distribution(df)
        fig_heatmap(df)
        fig_weather(df)
        fig_temp_scatter(df)
        fig_congestion_over_time(df)
        write_interpretations(FIGURE_DIR / "INTERPRETATIONS.md")
    except (FileNotFoundError, PermissionError, OSError):
        logger.error("Visualisation stage aborted — file access problem", exc_info=True)
        return 3
    except (ValueError, KeyError, TypeError, IndexError):
        logger.error("Visualisation stage aborted — data problem", exc_info=True)
        return 5

    logger.info("All %s figures generated successfully in %s",
                len(INTERPRETATIONS), FIGURE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
