"""
Capstone Part 3, Task 5 — Travel-timing recommendation system.

The dataset describes a single corridor, so there is no route choice to
make. The system therefore recommends WHEN to travel rather than which way:

* ``historical`` mode ranks departure hours by what the corridor has
  actually carried under matching conditions (day type, weather).
* ``model`` mode asks the trained gradient-boosting regressor to predict
  each candidate hour for a specific calendar date and assumed weather,
  which lets it account for holidays and season.

Both modes produce the same output: ranked travel windows, the hours to
avoid, and a plain-language recommendation sentence.

Usage
-----
    python part3_machine_learning/src/recommender.py --day-type weekday
    python part3_machine_learning/src/recommender.py --date 2018-07-04 --weather Rain
    python part3_machine_learning/src/recommender.py --date 2018-03-15 \\
        --earliest 7 --latest 19 --duration 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent.parent / "part2_python"))

import common  # noqa: E402
from logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

LOG_FILE = common.PART3_DIR / "logs" / "part3.log"
PROFILE_PATH = common.OUTPUT_DIR / "recommender_profile.json"
REGRESSOR_PATH = common.MODEL_DIR / "regressor_hist_gradient_boosting.joblib"

# US federal holidays present in the data, used to flag a requested date.
FIXED_HOLIDAYS = {(1, 1), (7, 4), (11, 11), (12, 25)}

WEATHER_SEVERITY = {
    "Clear": 0, "Clouds": 1, "Mist": 3, "Haze": 3, "Smoke": 4, "Fog": 5,
    "Drizzle": 4, "Rain": 6, "Snow": 8, "Squall": 9, "Thunderstorm": 9,
}
SEVERE = {"Rain", "Snow", "Thunderstorm", "Squall", "Drizzle"}
LOW_VIS = {"Fog", "Mist", "Haze", "Smoke", "Snow", "Squall"}


class RecommendationError(Exception):
    """Raised for a request the recommender cannot satisfy."""


@dataclass
class TravelWindow:
    start_hour: int
    end_hour: int
    expected_volume: float
    relative_to_average: float          # e.g. -0.42 means 42% below average
    congestion_level: str
    rank: int = 0


@dataclass
class Recommendation:
    mode: str
    day_type: str
    date: str | None
    weather: str | None
    window_hours: int
    candidate_hours: list[int]
    average_volume: float
    best_windows: list[TravelWindow]
    worst_windows: list[TravelWindow]
    hourly_profile: dict[int, float]
    recommendation: str
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hourly_profile"] = {str(k): round(v, 1) for k, v in self.hourly_profile.items()}
        return data


# ---------------------------------------------------------------------------
# Historical profile
# ---------------------------------------------------------------------------
def build_profile(df: pd.DataFrame) -> dict:
    """Precompute the lookup tables the historical mode needs."""
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values

    def table(subset: pd.DataFrame) -> dict[str, float]:
        return {str(int(h)): float(v) for h, v in
                subset.groupby("hour")["traffic_volume"].mean().items()}

    profile = {
        "quartiles": {"q1": float(q1), "q2": float(q2), "q3": float(q3)},
        "overall_mean": float(df["traffic_volume"].mean()),
        "by_day_type": {
            "weekday": table(df[df["is_weekend"] == 0]),
            "weekend": table(df[df["is_weekend"] == 1]),
            "all": table(df),
        },
        "by_day_type_and_weather": {},
        "weather_sample_sizes": {},
    }
    for day_type, mask in [("weekday", df["is_weekend"] == 0),
                           ("weekend", df["is_weekend"] == 1)]:
        profile["by_day_type_and_weather"][day_type] = {}
        profile["weather_sample_sizes"][day_type] = {}
        for condition, group in df[mask].groupby("weather_main"):
            profile["by_day_type_and_weather"][day_type][condition] = table(group)
            profile["weather_sample_sizes"][day_type][condition] = int(len(group))

    logger.info("Recommender profile built from %s hours", f"{len(df):,}")
    return profile


def load_or_build_profile() -> dict:
    if PROFILE_PATH.exists():
        try:
            profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            logger.info("Recommender profile loaded from %s", PROFILE_PATH.name)
            return profile
        except (OSError, json.JSONDecodeError):
            logger.warning("Profile at %s unreadable; rebuilding", PROFILE_PATH)
    df = common.load_features()
    profile = build_profile(df)
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    logger.info("Recommender profile saved to %s", PROFILE_PATH)
    return profile


# ---------------------------------------------------------------------------
# Feature construction for model mode
# ---------------------------------------------------------------------------
def features_for_date(target: date, weather: str, temp_c: float | None,
                      feature_names: list[str]) -> pd.DataFrame:
    """Build one feature row per hour of the requested day."""
    rows = []
    is_weekend = int(target.weekday() >= 5)
    is_holiday = int((target.month, target.day) in FIXED_HOLIDAYS)
    day_of_year = target.timetuple().tm_yday

    # A seasonal default when the caller gives no temperature.
    if temp_c is None:
        temp_c = 8.2 + 16.0 * np.cos(2 * np.pi * (day_of_year - 200) / 365.25)
    temp_k = temp_c + 273.15

    severity = WEATHER_SEVERITY.get(weather, 1)
    is_precip = int(weather in {"Rain", "Drizzle", "Snow", "Thunderstorm", "Squall"})
    rain = 0.8 if weather in {"Rain", "Drizzle", "Thunderstorm"} else 0.0
    snow = 0.1 if weather == "Snow" else 0.0
    clouds = {"Clear": 5, "Clouds": 75, "Mist": 90, "Haze": 60, "Fog": 95,
              "Smoke": 60, "Drizzle": 90, "Rain": 90, "Snow": 90,
              "Squall": 95, "Thunderstorm": 95}.get(weather, 50)

    for hour in range(24):
        row = {
            "hour": hour, "day_of_week": target.weekday(), "is_weekend": is_weekend,
            "month": target.month, "day_of_year": day_of_year,
            "is_morning_rush": int(6 <= hour <= 9),
            "is_evening_rush": int(15 <= hour <= 18),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_week_sin": np.sin(2 * np.pi * target.weekday() / 7),
            "day_of_week_cos": np.cos(2 * np.pi * target.weekday() / 7),
            "month_sin": np.sin(2 * np.pi * target.month / 12),
            "month_cos": np.cos(2 * np.pi * target.month / 12),
            "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "temp": temp_k, "rain_1h": rain, "snow_1h": snow, "clouds_all": clouds,
            "weather_severity": severity, "total_precipitation": rain + snow,
            "is_severe_weather": int(weather in SEVERE),
            "is_low_visibility": int(weather in LOW_VIS),
            "is_precipitating": is_precip, "is_raining": int(rain > 0),
            "is_snowing": int(snow > 0), "is_freezing": int(temp_k < 273.15),
            "is_extreme_cold": int(temp_k < 258.15), "is_hot": int(temp_k > 300),
            "is_overcast": int(clouds >= 75), "is_holiday": is_holiday,
        }
        row["is_rush_hour"] = int(row["is_morning_rush"] or row["is_evening_rush"])
        row["adverse_conditions_score"] = (
            severity + 2 * row["is_low_visibility"] + 2 * row["is_freezing"] + is_precip
        )
        for name in feature_names:
            if name.startswith("weather_") and name not in row:
                row[name] = int(name == f"weather_{weather}")
        rows.append(row)

    frame = pd.DataFrame(rows)
    missing = [c for c in feature_names if c not in frame.columns]
    for c in missing:
        frame[c] = 0
    if missing:
        logger.debug("Feature(s) defaulted to 0 for model input: %s", missing)
    return frame[feature_names]


# ---------------------------------------------------------------------------
# Core recommendation logic
# ---------------------------------------------------------------------------
def classify(volume: float, q: dict) -> str:
    if volume <= q["q1"]:
        return "Low"
    if volume <= q["q2"]:
        return "Medium"
    if volume <= q["q3"]:
        return "High"
    return "Severe"


def recommend(
    day_type: str = "weekday",
    target_date: date | None = None,
    weather: str | None = None,
    earliest: int = 0,
    latest: int = 23,
    duration: int = 1,
    temp_c: float | None = None,
    profile: dict | None = None,
    prefer_model: bool = True,
) -> Recommendation:
    """Produce a travel-timing recommendation.

    Parameters
    ----------
    day_type:  'weekday', 'weekend' or 'all'. Ignored when a date is given.
    target_date: a specific calendar date; enables model mode.
    weather:   an assumed weather condition, e.g. 'Rain'.
    earliest, latest: the acceptable departure hour range, inclusive.
    duration:  journey length in whole hours; windows are scored on the
               mean expected volume across every hour they span.
    temp_c:    assumed temperature; a seasonal default is used if omitted.
    """
    if not 0 <= earliest <= 23 or not 0 <= latest <= 23:
        raise RecommendationError("earliest and latest must be between 0 and 23")
    if earliest > latest:
        raise RecommendationError(
            f"earliest ({earliest:02d}:00) is after latest ({latest:02d}:00)")
    if not 1 <= duration <= 12:
        raise RecommendationError("duration must be between 1 and 12 hours")
    if weather is not None and weather not in WEATHER_SEVERITY:
        raise RecommendationError(
            f"'{weather}' is not a recognised weather condition. Choose from: "
            + ", ".join(sorted(WEATHER_SEVERITY)))

    profile = profile or load_or_build_profile()
    caveats: list[str] = []

    if target_date is not None:
        day_type = "weekend" if target_date.weekday() >= 5 else "weekday"

    # ---- Obtain the hourly expected-volume profile -----------------------
    mode = "historical"
    hourly: dict[int, float] = {}

    if target_date is not None and prefer_model and REGRESSOR_PATH.exists():
        try:
            model = joblib.load(REGRESSOR_PATH)
            feature_names = list(getattr(model, "feature_names_in_", []))
            if not feature_names:
                raise RecommendationError("model has no recorded feature names")
            frame = features_for_date(target_date, weather or "Clouds", temp_c,
                                      feature_names)
            preds = model.predict(frame)
            hourly = {h: float(max(v, 0)) for h, v in enumerate(preds)}
            mode = "model"
            logger.info("Model mode — predicted 24 hours for %s (%s, %s)",
                        target_date, day_type, weather or "Clouds")
            if (target_date.month, target_date.day) in FIXED_HOLIDAYS:
                caveats.append(
                    f"{target_date} is a public holiday; the model has seen only a "
                    "handful of holidays, so treat the figures as indicative.")
        except (OSError, ValueError, KeyError, RecommendationError) as exc:
            logger.warning("Model mode unavailable (%s); falling back to historical",
                           exc)

    if not hourly:
        if weather and day_type in ("weekday", "weekend"):
            table = profile["by_day_type_and_weather"].get(day_type, {}).get(weather)
            n = profile["weather_sample_sizes"].get(day_type, {}).get(weather, 0)
            if table and n >= 200:
                hourly = {int(h): v for h, v in table.items()}
                logger.info("Historical mode — %s, %s (%s hours of history)",
                            day_type, weather, f"{n:,}")
            else:
                caveats.append(
                    f"Only {n} hours of '{weather}' history on {day_type}s; the "
                    "recommendation uses all-weather history instead.")
                logger.warning("Insufficient %s/%s history (%s hours); using "
                               "all-weather profile", day_type, weather, n)
        if not hourly:
            hourly = {int(h): v for h, v in profile["by_day_type"][day_type].items()}
            logger.info("Historical mode — %s, all weather", day_type)

    # Some hours can be absent from a sparse historical slice.
    for h in range(24):
        if h not in hourly:
            hourly[h] = profile["by_day_type"]["all"].get(str(h),
                                                          profile["overall_mean"])
            caveats.append(f"No history for {h:02d}:00 under these conditions; "
                           "the all-conditions average was used.")

    # ---- Score candidate windows -----------------------------------------
    average = float(np.mean([hourly[h] for h in range(24)]))
    candidates = [h for h in range(earliest, latest + 1) if h + duration - 1 <= 23]
    if not candidates:
        raise RecommendationError(
            f"A {duration}-hour journey cannot start between {earliest:02d}:00 and "
            f"{latest:02d}:00 and finish before midnight.")

    windows = []
    for start in candidates:
        span = [hourly[h] for h in range(start, start + duration)]
        expected = float(np.mean(span))
        windows.append(TravelWindow(
            start_hour=start, end_hour=start + duration,
            expected_volume=expected,
            relative_to_average=(expected - average) / average,
            congestion_level=classify(expected, profile["quartiles"]),
        ))
    windows.sort(key=lambda w: w.expected_volume)
    for i, w in enumerate(windows, start=1):
        w.rank = i

    best = windows[:3]
    worst = sorted(windows, key=lambda w: -w.expected_volume)[:3]

    # ---- Plain-language sentence -----------------------------------------
    top = best[0]
    when = (f"a {day_type} journey" if target_date is None
            else f"your journey on {target_date:%A %d %B %Y}")
    weather_clause = f" in {weather.lower()}" if weather else ""
    sentence = (
        f"For {when}{weather_clause}, consider travelling between "
        f"{top.start_hour:02d}:00 and {top.end_hour:02d}:00, when "
        f"{'predicted' if mode == 'model' else 'historical'} traffic volumes average "
        f"{top.expected_volume:,.0f} vehicles per hour — "
        f"{abs(top.relative_to_average):.0%} "
        f"{'below' if top.relative_to_average < 0 else 'above'} the daily average of "
        f"{average:,.0f}. Avoid {worst[0].start_hour:02d}:00–{worst[0].end_hour:02d}:00, "
        f"the busiest window in your range at {worst[0].expected_volume:,.0f}."
    )
    if len(best) > 1:
        sentence += (f" If that does not suit, {best[1].start_hour:02d}:00–"
                     f"{best[1].end_hour:02d}:00 is the next best option.")

    if weather in {"Snow", "Fog", "Thunderstorm", "Squall"}:
        caveats.append(
            f"{weather} lowers volume only slightly but raises collision risk; a "
            "quieter road is not necessarily a safer one.")

    return Recommendation(
        mode=mode, day_type=day_type,
        date=target_date.isoformat() if target_date else None,
        weather=weather, window_hours=duration, candidate_hours=candidates,
        average_volume=average, best_windows=best, worst_windows=worst,
        hourly_profile=hourly, recommendation=sentence, caveats=caveats,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_recommendation(rec: Recommendation) -> None:
    rule = "-" * 68
    print()
    print(f"TRAVEL TIMING RECOMMENDATION  ({rec.mode} mode)")
    print(rule)
    print(f"  Day type        {rec.day_type}")
    if rec.date:
        print(f"  Date            {rec.date}")
    if rec.weather:
        print(f"  Weather         {rec.weather}")
    print(f"  Journey length  {rec.window_hours} hour(s)")
    print(f"  Daily average   {rec.average_volume:,.0f} vehicles/hour")
    print()
    print("  Best windows")
    for w in rec.best_windows:
        print(f"    {w.rank}. {w.start_hour:02d}:00–{w.end_hour:02d}:00   "
              f"{w.expected_volume:>6,.0f} vehicles/hour   "
              f"{w.relative_to_average:+.0%}   {w.congestion_level}")
    print()
    print("  Windows to avoid")
    for w in rec.worst_windows:
        print(f"       {w.start_hour:02d}:00–{w.end_hour:02d}:00   "
              f"{w.expected_volume:>6,.0f} vehicles/hour   "
              f"{w.relative_to_average:+.0%}   {w.congestion_level}")
    print()
    print("  Recommendation")
    for line in _wrap(rec.recommendation, 64):
        print(f"    {line}")
    if rec.caveats:
        print()
        print("  Please note")
        for caveat in rec.caveats:
            for i, line in enumerate(_wrap(caveat, 62)):
                print(f"    {'- ' if i == 0 else '  '}{line}")
    print()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recommend when to travel on I-94 westbound.")
    parser.add_argument("--day-type", choices=["weekday", "weekend", "all"], default="weekday")
    parser.add_argument("--date", help="specific date, YYYY-MM-DD (enables model mode)")
    parser.add_argument("--weather", help="assumed weather, e.g. Clear, Rain, Snow")
    parser.add_argument("--temp", type=float, help="assumed temperature in Celsius")
    parser.add_argument("--earliest", type=int, default=0)
    parser.add_argument("--latest", type=int, default=23)
    parser.add_argument("--duration", type=int, default=1, help="journey hours")
    parser.add_argument("--historical", action="store_true",
                        help="force historical mode even when a date is given")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(log_file=LOG_FILE, debug=args.debug)
    logger.info("Recommender invoked with %s",
                {k: v for k, v in vars(args).items() if k != "debug"})

    target = None
    if args.date:
        try:
            target = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            logger.error("Invalid date supplied: %r", args.date)
            print(f"\nError: '{args.date}' is not a valid date. Use YYYY-MM-DD.\n",
                  file=sys.stderr)
            return 1

    try:
        rec = recommend(
            day_type=args.day_type, target_date=target, weather=args.weather,
            earliest=args.earliest, latest=args.latest, duration=args.duration,
            temp_c=args.temp, prefer_model=not args.historical,
        )
    except RecommendationError as exc:
        logger.error("Recommendation failed: %s", exc)
        print(f"\nError: {exc}\n", file=sys.stderr)
        return 1
    except (FileNotFoundError, OSError):
        logger.error("Recommendation failed — data or model files missing", exc_info=True)
        print("\nError: required data files are missing. Run the Part 2 pipeline "
              "and part3 supervised.py first.\n", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(rec.to_dict(), indent=2))
    else:
        print_recommendation(rec)
    logger.info("Recommendation produced in %s mode: best window %02d:00-%02d:00",
                rec.mode, rec.best_windows[0].start_hour, rec.best_windows[0].end_hour)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
