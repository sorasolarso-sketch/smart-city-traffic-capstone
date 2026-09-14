# Part 2 — Python: A Reproducible Traffic Analytics Pipeline

Four scripts that take the raw CSV to an ML-ready feature table, six
diagnostic figures and a command-line query tool, with every data change
logged with a count and a reason.

## Structure

```
part2_python/
├── logging_config.py         shared logging setup — configures nothing on import
├── pipeline.py               Task 1: load → validate schema → clean → save
├── feature_engineering.py    Task 2: 57 engineered features + congestion target
├── visualizations.py         Task 3: six Matplotlib figures + written interpretations
├── viz_style.py              shared plotting palette and rcParams
├── mini_app/
│   └── traffic_app.py        Task 4: CLI with six commands
├── figures/
│   ├── 01_hourly_weekday_vs_weekend.png
│   ├── 02_traffic_volume_distribution.png
│   ├── 03_hour_day_heatmap.png
│   ├── 04_weather_impact.png
│   ├── 05_temperature_vs_traffic_by_daypart.png
│   ├── 06_congestion_rate_over_time.png
│   └── INTERPRETATIONS.md    one interpretation per figure
├── logs/
│   ├── pipeline.log          sample output from pipeline, features and visualisations
│   └── app.log               sample output from the CLI
├── report/Part2_Python_Methodology_Report.docx
└── README.md
```

## Run

From the repository root, in order:

```bash
python part2_python/pipeline.py              # → data/processed/traffic_clean.csv
python part2_python/feature_engineering.py   # → data/processed/traffic_features.csv
python part2_python/visualizations.py        # → part2_python/figures/
```

Flags common to all three: `--debug` (DEBUG on the console), and `--input` /
`--output` on the first two to point at other files.

### The mini application

```bash
python part2_python/mini_app/traffic_app.py --help
python part2_python/mini_app/traffic_app.py summary
python part2_python/mini_app/traffic_app.py query 2017-08-31            # whole day
python part2_python/mini_app/traffic_app.py query 2017-08-31 --hour 17  # one hour vs typical
python part2_python/mini_app/traffic_app.py peak --top 5 --day-type weekday
python part2_python/mini_app/traffic_app.py compare                      # weekday vs weekend
python part2_python/mini_app/traffic_app.py recommend --day-type weekend --earliest 8 --latest 20
python part2_python/mini_app/traffic_app.py weather                      # all conditions
python part2_python/mini_app/traffic_app.py weather --condition Snow
```

Invalid input (a malformed date, `--hour 25`, an unknown condition, a window
that ends before it starts) produces one ERROR line in the log and a short
readable message on stderr — never a traceback.

## What the pipeline does, step by step

| Step | Action | Rows affected | Logged at |
| --- | --- | ---: | --- |
| Load | `pd.read_csv(..., keep_default_na=False)` — `"None"` is a pandas NA token and must not become NaN | 48,204 × 9 | INFO |
| Validate | All nine expected columns present; numeric columns coercible; non-empty. Raises `SchemaValidationError` before any cleaning | — | INFO / ERROR |
| 1. Standardise | Trim + title-case `weather_main`, lower-case `weather_description`, trim `holiday`; derive `is_holiday` | 1,730 values | WARNING |
| 2. Parse dates | Explicit `%Y-%m-%d %H:%M:%S`; drop unparseable; quantify the 11,976-hour (22.8%) coverage gap | 0 dropped | INFO / WARNING |
| 3a. Exact duplicates | `drop_duplicates()` | 17 | WARNING |
| 3b. Duplicate hours | One row per timestamp, most severe weather kept (the feed emits one row per condition) | 7,612 | WARNING |
| 4a. Temperature | 0 K readings imputed with the **same-calendar-month median**, in an explicit loop over months | 10 | WARNING (DEBUG per month) |
| 4b. Rainfall | > 9,000 mm/h imputed with same-month median | 1 | WARNING |
| 4c–4d. Precipitation, cloud | Negative precipitation → 0; cloud clipped to 0–100 | 0 | INFO |
| 4e. Traffic | Daytime zero readings imputed with the same-hour median; overnight zero retained | 1 (+1 kept) | WARNING / INFO |
| Save | `data/processed/traffic_clean.csv` | 40,575 × 10 | INFO |

## Feature engineering

Adds 57 columns, removes no rows (logged as shape before 40,575 × 10, after
40,575 × 67):

- **Time:** hour, day_of_week, day_name, is_weekend, month, year, day_of_year,
  week_of_year, quarter, is_morning_rush, is_evening_rush, is_rush_hour,
  part_of_day, season.
- **Cyclical:** sin/cos of hour (24), day_of_week (7), month (12), day_of_year
  (365.25). Unit-circle check logged at DEBUG.
- **Weather:** temp_celsius, weather_severity (0–10 ordinal), is_severe_weather,
  is_low_visibility, is_precipitating, is_raining, is_snowing, is_freezing,
  is_extreme_cold, is_hot, is_overcast, total_precipitation,
  adverse_conditions_score, and 11 one-hot `weather_*` columns.
- **Scaled:** `_zscore` and `_minmax` versions of temp, clouds_all,
  total_precipitation, weather_severity. The target is not scaled.
- **Targets:** `congestion_category` from data-driven quartiles (Low ≤ 1,249,
  Medium ≤ 3,428, High ≤ 4,952, Severe above; 25% each — thresholds logged at
  DEBUG) with `congestion_level` 0–3, plus `is_congested` (> 5,500, Part 1's
  fixed definition). The two are documented in `data/processed/feature_metadata.json`.

## Logging configuration

`logging_config.configure_logging()` is called only inside each script's
`main()`. It attaches a **console handler at INFO** and a **file handler at
DEBUG** to the root logger; library code uses `logging.getLogger(__name__)`
and adds no handlers. Format: `timestamp | LEVEL | module | message`.

- Pipeline, feature engineering and visualisations append to `logs/pipeline.log`.
- The CLI appends to `logs/app.log`.
- `--debug` raises the console to DEBUG; the file always has DEBUG, so
  intermediate values (quartiles, medians, scaler parameters) are on record
  without cluttering normal runs.

| Level | Meaning in this project |
| --- | --- |
| DEBUG | Intermediate calculations not in the final output |
| INFO | Milestones: loaded (rows × cols), step done, shape before/after, figure saved with path, command invoked with args |
| WARNING | Rows dropped / imputed / capped — always with count and reason |
| ERROR | Stage cannot continue; logged with `exc_info=True`, then graceful exit |

`print()` appears only in `mini_app/traffic_app.py`, for the answers.
