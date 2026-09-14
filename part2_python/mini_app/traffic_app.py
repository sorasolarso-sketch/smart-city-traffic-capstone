"""
Capstone Part 2, Task 4 — Mini traffic analytics application.

A command-line tool for querying the processed traffic dataset.

Commands
--------
    summary                     overview of the processed dataset
    query      DATE [--hour H]  traffic for a specific date, or date and hour
    peak       [--top N]        the busiest periods
    compare                     weekday versus weekend traffic
    recommend  [--day-type]     recommended travel windows
    weather    [--condition C]  traffic broken down by weather condition

Examples
--------
    python part2_python/mini_app/traffic_app.py summary
    python part2_python/mini_app/traffic_app.py query 2017-08-31 --hour 17
    python part2_python/mini_app/traffic_app.py peak --top 5 --day-type weekday
    python part2_python/mini_app/traffic_app.py compare
    python part2_python/mini_app/traffic_app.py recommend --day-type weekday
    python part2_python/mini_app/traffic_app.py weather --condition Snow

A note on output
----------------
``print()`` appears throughout this module and only in this module. That is
deliberate and consistent with the brief: the printed lines ARE the product
of the tool, the answer the user asked for. Everything describing internal
progress — which command ran, with which arguments, and any failure — goes
to the logger instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PART2_DIR = APP_DIR.parent
REPO_ROOT = PART2_DIR.parent
sys.path.insert(0, str(PART2_DIR))
from logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "traffic_features.csv"
APP_LOG = PART2_DIR / "logs" / "app.log"

RULE = "-" * 68
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


class UserInputError(Exception):
    """Raised when the user supplies an argument the tool cannot act on."""


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["date_time"], keep_default_na=False)
    except FileNotFoundError:
        logger.error(
            "Processed dataset not found at %s. Run pipeline.py and "
            "feature_engineering.py first.", path,
        )
        raise UserInputError(
            f"Processed dataset not found at {path}.\n"
            "Run these two commands first:\n"
            "    python part2_python/pipeline.py\n"
            "    python part2_python/feature_engineering.py"
        ) from None
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.error("Could not read the dataset at %s: %s", path, exc)
        raise UserInputError(f"Could not read the dataset at {path}.") from None

    logger.info("Dataset loaded: %s rows, %s columns", f"{len(df):,}", df.shape[1])
    return df


def parse_user_date(text: str) -> datetime:
    """Parse a user-supplied date, accepting several common formats.

    Raises :class:`UserInputError` with a readable message rather than
    letting a parsing exception reach the user as a traceback.
    """
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    logger.error("Invalid date supplied by user: %r", text)
    raise UserInputError(
        f"'{text}' is not a date this tool recognises.\n"
        "Accepted formats: YYYY-MM-DD (preferred), DD/MM/YYYY, DD-MM-YYYY, YYYY/MM/DD.\n"
        "Example: 2017-08-31"
    )


def validate_hour(hour: int | None) -> int | None:
    if hour is None:
        return None
    if not 0 <= hour <= 23:
        logger.error("Invalid hour supplied by user: %s", hour)
        raise UserInputError(f"Hour must be between 0 and 23; received {hour}.")
    return hour


def filter_day_type(df: pd.DataFrame, day_type: str) -> pd.DataFrame:
    if day_type == "weekday":
        return df[df["is_weekend"] == 0]
    if day_type == "weekend":
        return df[df["is_weekend"] == 1]
    return df


def describe_level(volume: float, df: pd.DataFrame) -> str:
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75])
    if volume <= q1:
        return "Low"
    if volume <= q2:
        return "Medium"
    if volume <= q3:
        return "High"
    return "Severe"


# ---------------------------------------------------------------------------
# Command: summary
# ---------------------------------------------------------------------------
def cmd_summary(df: pd.DataFrame, args: argparse.Namespace) -> int:
    print()
    print("TRAFFIC DATASET SUMMARY")
    print(RULE)
    print(f"  Hours of data          {len(df):,}")
    print(f"  Date range             {df['date_time'].min():%Y-%m-%d} to "
          f"{df['date_time'].max():%Y-%m-%d}")
    print(f"  Average volume         {df['traffic_volume'].mean():,.0f} vehicles/hour")
    print(f"  Median volume          {df['traffic_volume'].median():,.0f} vehicles/hour")
    print(f"  Busiest hour recorded  {df['traffic_volume'].max():,} vehicles "
          f"({df.loc[df['traffic_volume'].idxmax(), 'date_time']:%Y-%m-%d %H:%M})")
    print(f"  Average temperature    {df['temp_celsius'].mean():.1f} °C")
    print(f"  Congested hours        {int(df['is_congested'].sum()):,} "
          f"({100 * df['is_congested'].mean():.1f}% above 5,500 vehicles/hour)")
    print()
    print("  Congestion categories")
    for level in ["Low", "Medium", "High", "Severe"]:
        count = int((df["congestion_category"] == level).sum())
        share = 100 * count / len(df)
        bar = "#" * int(share / 2)
        print(f"    {level:<8} {count:>7,}  {share:5.1f}%  {bar}")
    print()
    print("  Weather conditions observed")
    for condition, count in df["weather_main"].value_counts().head(6).items():
        avg = df.loc[df["weather_main"] == condition, "traffic_volume"].mean()
        print(f"    {condition:<14} {count:>7,} hours   avg {avg:,.0f} vehicles/hour")
    print()
    return 0


# ---------------------------------------------------------------------------
# Command: query
# ---------------------------------------------------------------------------
def cmd_query(df: pd.DataFrame, args: argparse.Namespace) -> int:
    target = parse_user_date(args.date)
    hour = validate_hour(args.hour)

    day = df[df["date_time"].dt.date == target.date()]
    if day.empty:
        available_min = df["date_time"].min().date()
        available_max = df["date_time"].max().date()
        logger.error("No records found for %s", target.date())
        raise UserInputError(
            f"No traffic records exist for {target:%Y-%m-%d}.\n"
            f"The dataset covers {available_min} to {available_max}, but it has "
            "substantial gaps: the sensor was offline for long stretches of 2014 "
            "and 2015. Try a date in 2016 or 2017, for example 2017-08-31."
        )

    if hour is not None:
        row = day[day["date_time"].dt.hour == hour]
        if row.empty:
            logger.error("No record for %s hour %02d", target.date(), hour)
            hours_present = sorted(day["date_time"].dt.hour.tolist())
            raise UserInputError(
                f"No record for {target:%Y-%m-%d} at {hour:02d}:00.\n"
                f"Hours available on that date: "
                f"{', '.join(f'{h:02d}' for h in hours_present)}"
            )
        record = row.iloc[0]
        print()
        print(f"TRAFFIC AT {record['date_time']:%Y-%m-%d %H:%M} "
              f"({record['day_name']})")
        print(RULE)
        print(f"  Traffic volume     {int(record['traffic_volume']):,} vehicles/hour")
        print(f"  Congestion level   {record['congestion_category']}")
        print(f"  Weather            {record['weather_main']} "
              f"({record['weather_description']})")
        print(f"  Temperature        {record['temp_celsius']:.1f} °C")
        print(f"  Cloud cover        {int(record['clouds_all'])}%")
        print(f"  Rain (1h)          {record['rain_1h']:.2f} mm")
        if record["is_holiday"]:
            print(f"  Public holiday     {record['holiday']}")
        same_hour = df[df["hour"] == hour]["traffic_volume"].mean()
        delta = record["traffic_volume"] - same_hour
        print()
        print(f"  Typical for {hour:02d}:00   {same_hour:,.0f} vehicles/hour")
        print(f"  Difference         {delta:+,.0f} ({delta / same_hour:+.1%})")
        print()
        return 0

    print()
    print(f"TRAFFIC ON {target:%Y-%m-%d} ({day.iloc[0]['day_name']})")
    print(RULE)
    print(f"  Hours recorded     {len(day)}")
    print(f"  Daily average      {day['traffic_volume'].mean():,.0f} vehicles/hour")
    print(f"  Busiest hour       {int(day['traffic_volume'].max()):,} at "
          f"{day.loc[day['traffic_volume'].idxmax(), 'date_time']:%H:%M}")
    print(f"  Quietest hour      {int(day['traffic_volume'].min()):,} at "
          f"{day.loc[day['traffic_volume'].idxmin(), 'date_time']:%H:%M}")
    print(f"  Temperature range  {day['temp_celsius'].min():.1f} °C to "
          f"{day['temp_celsius'].max():.1f} °C")
    if day["is_holiday"].any():
        print(f"  Public holiday     {day.loc[day['is_holiday'] == 1, 'holiday'].iloc[0]}")
    print()
    print("  Hour by hour")
    for _, row in day.sort_values("date_time").iterrows():
        bar = "#" * int(row["traffic_volume"] / 180)
        print(f"    {row['date_time']:%H:%M}  {int(row['traffic_volume']):>5,}  "
              f"{row['weather_main']:<13} {bar}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Command: peak
# ---------------------------------------------------------------------------
def cmd_peak(df: pd.DataFrame, args: argparse.Namespace) -> int:
    if args.top < 1:
        logger.error("Invalid --top value: %s", args.top)
        raise UserInputError(f"--top must be 1 or greater; received {args.top}.")

    subset = filter_day_type(df, args.day_type)
    if subset.empty:
        raise UserInputError(f"No records match day type '{args.day_type}'.")

    by_hour = (subset.groupby("hour")["traffic_volume"]
                     .agg(["mean", "size"])
                     .sort_values("mean", ascending=False))
    top = by_hour.head(args.top)

    print()
    print(f"HIGHEST-TRAFFIC HOURS — {args.day_type}")
    print(RULE)
    print(f"  {'Rank':<6}{'Hour':<8}{'Avg vehicles/hour':<22}{'Sample'}")
    for rank, (hour, row) in enumerate(top.iterrows(), start=1):
        print(f"  {rank:<6}{int(hour):02d}:00   {row['mean']:>10,.0f}"
              f"{'':<12}{int(row['size']):,} hours")

    print()
    print(f"  Busiest single hours on record ({args.day_type})")
    worst = subset.nlargest(args.top, "traffic_volume")
    for rank, (_, row) in enumerate(worst.iterrows(), start=1):
        print(f"  {rank:<6}{row['date_time']:%Y-%m-%d %H:%M} ({row['day_name'][:3]})  "
              f"{int(row['traffic_volume']):>6,} vehicles   {row['weather_main']}")

    print()
    print("  Busiest hour of each weekday")
    for day in DAY_ORDER:
        day_data = subset[subset["day_name"] == day]
        if day_data.empty:
            continue
        hourly = day_data.groupby("hour")["traffic_volume"].mean()
        print(f"    {day:<11} {int(hourly.idxmax()):02d}:00   "
              f"{hourly.max():>6,.0f} vehicles/hour")
    print()
    return 0


# ---------------------------------------------------------------------------
# Command: compare
# ---------------------------------------------------------------------------
def cmd_compare(df: pd.DataFrame, args: argparse.Namespace) -> int:
    weekday = df[df["is_weekend"] == 0]
    weekend = df[df["is_weekend"] == 1]

    print()
    print("WEEKDAY VERSUS WEEKEND TRAFFIC")
    print(RULE)
    print(f"  {'Measure':<26}{'Weekday':>14}{'Weekend':>14}{'Difference':>14}")

    rows = [
        ("Hours recorded", len(weekday), len(weekend), "{:,.0f}"),
        ("Average volume", weekday["traffic_volume"].mean(),
         weekend["traffic_volume"].mean(), "{:,.0f}"),
        ("Median volume", weekday["traffic_volume"].median(),
         weekend["traffic_volume"].median(), "{:,.0f}"),
        ("Peak hour average", weekday.groupby("hour")["traffic_volume"].mean().max(),
         weekend.groupby("hour")["traffic_volume"].mean().max(), "{:,.0f}"),
        ("Congested hours (%)", 100 * weekday["is_congested"].mean(),
         100 * weekend["is_congested"].mean(), "{:.1f}"),
    ]
    for label, wd_value, we_value, fmt in rows:
        diff = wd_value - we_value
        print(f"  {label:<26}{fmt.format(wd_value):>14}{fmt.format(we_value):>14}"
              f"{fmt.format(diff):>14}")

    wd_peak_hour = int(weekday.groupby("hour")["traffic_volume"].mean().idxmax())
    we_peak_hour = int(weekend.groupby("hour")["traffic_volume"].mean().idxmax())
    print()
    print(f"  Weekday peak occurs at {wd_peak_hour:02d}:00, "
          f"weekend peak at {we_peak_hour:02d}:00")

    print()
    print("  Average volume by hour")
    print(f"  {'Hour':<7}{'Weekday':>10}{'Weekend':>10}   {'Gap':>8}")
    wd_hourly = weekday.groupby("hour")["traffic_volume"].mean()
    we_hourly = weekend.groupby("hour")["traffic_volume"].mean()
    for hour in range(24):
        gap = wd_hourly[hour] - we_hourly[hour]
        marker = "  <-- widest gap" if gap == (wd_hourly - we_hourly).max() else ""
        print(f"  {hour:02d}:00 {wd_hourly[hour]:>10,.0f}{we_hourly[hour]:>10,.0f}"
              f"   {gap:>+8,.0f}{marker}")
    print()
    return 0


# ---------------------------------------------------------------------------
# Command: recommend
# ---------------------------------------------------------------------------
def cmd_recommend(df: pd.DataFrame, args: argparse.Namespace) -> int:
    earliest = validate_hour(args.earliest)
    latest = validate_hour(args.latest)
    if earliest is not None and latest is not None and earliest > latest:
        logger.error("Invalid window: earliest %s after latest %s", earliest, latest)
        raise UserInputError(
            f"--earliest ({earliest:02d}:00) cannot be later than "
            f"--latest ({latest:02d}:00)."
        )

    subset = filter_day_type(df, args.day_type)
    window = subset
    if earliest is not None:
        window = window[window["hour"] >= earliest]
    if latest is not None:
        window = window[window["hour"] <= latest]

    if window.empty:
        raise UserInputError(
            "No hours fall inside the requested window. Widen --earliest/--latest."
        )

    hourly = window.groupby("hour")["traffic_volume"].mean().sort_values()
    overall = subset["traffic_volume"].mean()

    print()
    print(f"RECOMMENDED TRAVEL WINDOWS — {args.day_type}")
    if earliest is not None or latest is not None:
        lo = f"{earliest:02d}:00" if earliest is not None else "00:00"
        hi = f"{latest:02d}:00" if latest is not None else "23:00"
        print(f"Restricted to departures between {lo} and {hi}")
    print(RULE)

    best = hourly.head(3)
    for rank, (hour, volume) in enumerate(best.items(), start=1):
        # Expressed as a signed change against the average, so that this
        # list and the "hours to avoid" list below read on the same scale.
        change = 100 * (volume / overall - 1)
        print(f"  {rank}. {int(hour):02d}:00–{int(hour) + 1:02d}:00   "
              f"{volume:>6,.0f} vehicles/hour   {change:+.0f}% vs average")

    best_hour = int(best.index[0])
    print()
    print("  Recommendation")
    print(f"    For a {args.day_type} journey, consider travelling between "
          f"{best_hour:02d}:00 and {best_hour + 1:02d}:00, when historical traffic "
          f"volumes")
    print(f"    average {best.iloc[0]:,.0f} vehicles per hour — "
          f"{100 * (1 - best.iloc[0] / overall):.0f}% below the "
          f"{args.day_type} average of {overall:,.0f}.")

    worst = hourly.tail(3).sort_values(ascending=False)
    print()
    print("  Hours to avoid")
    for hour, volume in worst.items():
        excess = 100 * (volume / overall - 1)
        print(f"    {int(hour):02d}:00–{int(hour) + 1:02d}:00   "
              f"{volume:>6,.0f} vehicles/hour   {excess:+.0f}% vs average")

    # Daytime-only view, since 03:00 is rarely a practical suggestion.
    daytime = hourly[(hourly.index >= 7) & (hourly.index <= 19)]
    if not daytime.empty:
        quietest_daytime = int(daytime.idxmin())
        print()
        print("  Practical daytime alternative (07:00–19:00)")
        print(f"    {quietest_daytime:02d}:00–{quietest_daytime + 1:02d}:00 is the "
              f"quietest hour within normal waking travel times, at "
              f"{daytime.min():,.0f} vehicles/hour.")

    if args.day_type != "weekend":
        print()
        print("  Note: weekend patterns differ substantially. Run with "
              "--day-type weekend for weekend advice.")
    print()
    return 0


# ---------------------------------------------------------------------------
# Command: weather
# ---------------------------------------------------------------------------
def cmd_weather(df: pd.DataFrame, args: argparse.Namespace) -> int:
    available = sorted(df["weather_main"].unique())

    if args.condition:
        match = [c for c in available if c.lower() == args.condition.lower()]
        if not match:
            logger.error("Unknown weather condition requested: %r", args.condition)
            raise UserInputError(
                f"'{args.condition}' is not a weather condition in this dataset.\n"
                f"Available conditions: {', '.join(available)}"
            )
        condition = match[0]
        subset = df[df["weather_main"] == condition]

        print()
        print(f"TRAFFIC DURING {condition.upper()}")
        print(RULE)
        print(f"  Hours observed     {len(subset):,} "
              f"({100 * len(subset) / len(df):.1f}% of all hours)")
        print(f"  Average volume     {subset['traffic_volume'].mean():,.0f} vehicles/hour")
        print(f"  Overall average    {df['traffic_volume'].mean():,.0f} vehicles/hour")
        print(f"  Difference         "
              f"{subset['traffic_volume'].mean() - df['traffic_volume'].mean():+,.0f}")
        print(f"  Congested hours    {100 * subset['is_congested'].mean():.1f}% "
              f"(all conditions: {100 * df['is_congested'].mean():.1f}%)")
        print(f"  Average temp       {subset['temp_celsius'].mean():.1f} °C")

        if len(subset) < 100:
            print()
            print(f"  WARNING: only {len(subset)} hour(s) of data. These figures are "
                  "not reliable")
            print("           and should not be used to draw conclusions.")

        print()
        print("  When this condition occurs")
        by_hour = subset.groupby("hour").size()
        share = (by_hour / df.groupby("hour").size() * 100).round(1)
        busiest = share.nlargest(3)
        print(f"    Most frequent at: "
              + ", ".join(f"{int(h):02d}:00 ({v:.0f}% of those hours)"
                          for h, v in busiest.items()))
        print()
        print("  Reading this fairly: the average above mixes together the time of day")
        print("  each condition tends to occur. A condition common overnight will show")
        print("  low traffic for reasons that have nothing to do with the weather.")
        print()
        return 0

    print()
    print("TRAFFIC BY WEATHER CONDITION")
    print(RULE)
    print(f"  {'Condition':<15}{'Hours':>9}{'Avg volume':>13}{'Congested':>12}")
    table = (df.groupby("weather_main")
               .agg(hours=("traffic_volume", "size"),
                    avg=("traffic_volume", "mean"),
                    congested=("is_congested", "mean"))
               .sort_values("avg", ascending=False))
    for condition, row in table.iterrows():
        flag = "  (low n)" if row["hours"] < 100 else ""
        print(f"  {condition:<15}{int(row['hours']):>9,}{row['avg']:>13,.0f}"
              f"{100 * row['congested']:>11.1f}%{flag}")
    print()
    print("  Conditions marked (low n) rest on fewer than 100 observed hours and")
    print("  should not be compared against the others.")
    print()
    return 0


COMMANDS = {
    "summary": cmd_summary,
    "query": cmd_query,
    "peak": cmd_peak,
    "compare": cmd_compare,
    "recommend": cmd_recommend,
    "weather": cmd_weather,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic_app",
        description="Query the processed Metro Interstate traffic dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  traffic_app.py summary\n"
            "  traffic_app.py query 2017-08-31 --hour 17\n"
            "  traffic_app.py peak --top 5 --day-type weekday\n"
            "  traffic_app.py compare\n"
            "  traffic_app.py recommend --day-type weekday --earliest 7 --latest 19\n"
            "  traffic_app.py weather --condition Snow\n"
        ),
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="path to the processed feature CSV")
    parser.add_argument("--debug", action="store_true",
                        help="show DEBUG messages on the console")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser("summary", help="overview of the processed dataset")

    p_query = sub.add_parser("query", help="traffic for a specific date or hour")
    p_query.add_argument("date", help="date, e.g. 2017-08-31")
    p_query.add_argument("--hour", type=int, default=None,
                         help="hour of day, 0-23; omit for a full-day view")

    p_peak = sub.add_parser("peak", help="identify the busiest periods")
    p_peak.add_argument("--top", type=int, default=5, help="how many to list")
    p_peak.add_argument("--day-type", choices=["weekday", "weekend", "all"],
                        default="all")

    sub.add_parser("compare", help="compare weekday and weekend traffic")

    p_rec = sub.add_parser("recommend", help="recommend travel windows")
    p_rec.add_argument("--day-type", choices=["weekday", "weekend", "all"],
                       default="weekday")
    p_rec.add_argument("--earliest", type=int, default=None,
                       help="earliest acceptable departure hour, 0-23")
    p_rec.add_argument("--latest", type=int, default=None,
                       help="latest acceptable departure hour, 0-23")

    p_weather = sub.add_parser("weather", help="traffic by weather condition")
    p_weather.add_argument("--condition", default=None,
                           help="a single condition, e.g. Snow, Rain, Clear")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(log_file=APP_LOG, debug=args.debug)

    if not args.command:
        parser.print_help()
        return 0

    # Log which command was invoked and with which arguments, as required.
    invoked = {k: v for k, v in vars(args).items() if k not in {"command", "debug"}}
    logger.info("Command invoked: '%s' with arguments %s", args.command, invoked)

    try:
        df = load_data(args.data)
        exit_code = COMMANDS[args.command](df, args)
    except UserInputError as exc:
        # A readable message for the user; no traceback reaches the console.
        logger.error("Command '%s' failed: %s", args.command,
                     str(exc).replace("\n", " "))
        print(f"\nError: {exc}\n", file=sys.stderr)
        return 1
    except KeyError:
        logger.error("Command '%s' failed — expected column missing from the "
                     "dataset; re-run feature_engineering.py", args.command,
                     exc_info=True)
        print("\nError: the dataset is missing a required column. Re-run:\n"
              "    python part2_python/feature_engineering.py\n", file=sys.stderr)
        return 2
    except (ValueError, TypeError, IndexError):
        logger.error("Command '%s' failed with an unexpected data error",
                     args.command, exc_info=True)
        print("\nError: the command could not be completed. See the log at "
              f"{APP_LOG} for details.\n", file=sys.stderr)
        return 3

    logger.info("Command '%s' completed successfully (exit code %s)",
                args.command, exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
