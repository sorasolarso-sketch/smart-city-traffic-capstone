const path = require("path");
const { buildDocument } = require("./report_lib");

const ROOT = path.resolve(__dirname, "../..");
const FIG = (f) => path.join(ROOT, "part2_python/figures", f);
const OUT = path.join(ROOT, "part2_python/report/Part2_Python_Methodology_Report.docx");

const blocks = [
  { h1: "1. Objective and design" },
  { p: "Part 2 turns the one-off analysis of Part 1 into a reproducible pipeline: four Python stages that take the raw CSV to an ML-ready feature table, six diagnostic figures, and a command-line application for querying the result. Every stage is a separate script with its own entry point, every data change is logged with a count and a reason, and the whole chain re-runs from the raw file with three commands. The design goal was that a grader — or a colleague six months from now — could read `logs/pipeline.log` and reconstruct exactly what happened to every row without opening the code." },
  { table: {
    header: ["Stage", "Script", "Input → output", "Rows in → out"],
    align: ["left", "left", "left", "right"],
    widths: [1500, 2100, 4200, 2292],
    rows: [
      ["1. Clean", "pipeline.py", "raw CSV → data/processed/traffic_clean.csv", "48,204 → 40,575"],
      ["2. Features", "feature_engineering.py", "clean CSV → traffic_features.csv (67 columns)", "40,575 → 40,575"],
      ["3. Visualise", "visualizations.py", "features → figures/*.png + INTERPRETATIONS.md", "6 figures"],
      ["4. Query", "mini_app/traffic_app.py", "features → answers on the console", "6 commands"],
    ],
  }},

  { h1: "2. Pipeline methodology (Task 1)" },
  { p: "**Loading and schema validation.** The file is read with `keep_default_na=False`, which matters more than it sounds: `\"None\"` is one of pandas' default missing-value tokens, so the default reader silently converts the holiday sentinel on 48,143 rows into NaN and every later holiday count is wrong. Each I/O failure mode (missing file, permission, empty file, parse error, encoding) has its own `except` clause; there is no bare `except`. Schema validation runs before any other processing and checks column presence, coercibility of numeric columns, and that the file is non-empty, raising a `SchemaValidationError` that the entry point logs with `exc_info=True` and converts to exit code 2." },
  { p: "**Cleaning, one logged step at a time.** Step 1 standardises the three text columns (1,730 `weather_description` values were re-cased). Step 2 parses timestamps against an explicit format, drops any that fail, and quantifies the coverage gap — 11,976 hours, 22.8% of the span, are absent because the sensor was offline. Step 3 removes 17 exact duplicates and then 7,612 duplicate-timestamp rows; these are not errors but the weather feed's one-record-per-condition convention, and keeping them inflates every sum by 7–21%, so the most severe condition per hour is retained. Step 4 repairs impossible values: 10 temperatures of 0 K and one rainfall of 9,831 mm are imputed with **the median for the same calendar month**, computed in an explicit loop over months so a January fault is never patched with a July value; one daytime reading of zero vehicles is imputed from the same-hour median, while the single overnight zero is retained as plausible. Every step emits its own WARNING with the row count and the reason." },

  { h1: "3. Feature engineering (Task 2)" },
  { p: "The feature stage adds 57 columns and removes no rows. **Time:** hour, day of week, weekend flag, month, day of year, rush-hour flags, part-of-day band and season. **Cyclical encodings** of hour (period 24), day of week (7), month (12) and day of year (365.25) as sine/cosine pairs, so that 23:00 and 00:00 sit next to each other in feature space; a unit-circle check (sin² + cos² = 1) is logged at DEBUG. **Weather:** an ordinal severity score (0–10), binary indicators for severe weather, low visibility, precipitation, freezing and overcast, a composite adverse-conditions score, and 11 one-hot columns. **Scaling:** z-score and min-max versions of temperature, cloud cover, precipitation and severity; the target is deliberately left unscaled to avoid leaking test-period statistics into training." },
  { p: "**Target.** Two congestion definitions coexist and are never mixed. `congestion_category` follows the Part 3 brief: quartiles of traffic volume (Q1 = 1,249, median = 3,428, Q3 = 4,952) split the data into Low / Medium / High / Severe, each holding 25% of hours — balanced classes by construction, and cut-points that transfer to another corridor without re-tuning. `is_congested` keeps Part 1's fixed 5,500-vehicle threshold (15.05% of hours) so the two parts can be compared. The quartile thresholds are logged at DEBUG only, as the brief requires; the shape before (40,575 × 10) and after (40,575 × 67) is logged at INFO." },

  { h1: "4. What the figures show (Task 3)" },
  { image: FIG("01_hourly_weekday_vs_weekend.png"), width: 470,
    caption: "Figure 1 — Weekday demand has two peaks; the evening one (16:00, 6,241 veh/h) is the larger. Weekends have one broad midday plateau." },
  { p: "Six figures were produced, each with a written interpretation in `figures/INTERPRETATIONS.md`. Three findings carry through to Part 3. First, **the weekday and weekend profiles are different shapes**, not scaled copies: weekdays peak twice (07:00 and 16:00), weekends once (13:00, at 4,408 veh/h, 1.42× lower than the weekday maximum). Any model without a day-type feature is wrong two days in seven. Second, **volume is bimodal** — overnight and daytime masses with a near-empty valley between — which is why the mean (3,291) and median (3,428) describe no actual hour and why quartile-based classes were chosen over fixed thresholds. Third, **the weather ranking is a time-of-day artefact**: Clouds averages 3,617 veh/h and Fog 2,654 because fog forms at 04:00; faceting the temperature scatter by part of day dissolves the r = +0.139 correlation inside every band. The heatmap identifies Wednesday 16:00 as the single busiest cell of the week, and the congestion-over-time chart masks any month with fewer than 400 recorded hours, because an earlier draft that plotted them produced a dramatic — and entirely fictitious — collapse through 2014–2015." },

  { h1: "5. The mini application (Task 4)" },
  { p: "`traffic_app.py` exposes six sub-commands via `argparse`: `summary`, `query DATE [--hour H]`, `peak`, `compare`, `recommend` and `weather`. Every invocation logs the command and its arguments at INFO to `logs/app.log`. Invalid input — a malformed date, an hour of 25, an unknown weather condition, a departure window that ends before it starts — raises a `UserInputError` that is logged as a single ERROR line and printed as a two-line message; no traceback reaches the user. `print()` appears only in this module, only for the answers themselves, which is the one use the brief permits. The recommend command already produces the plain-language sentence the Part 3 recommender formalises (\"For a weekday journey, consider travelling between 19:00 and 21:00…\")." },

  { h1: "6. Logging configuration (Task 5)" },
  { p: "`logging_config.py` is imported by every module but configures nothing on import. Each entry-point script calls `configure_logging()` inside its own `main()`, which attaches two handlers to the root logger: a console handler at INFO and a file handler at DEBUG writing to `part2_python/logs/pipeline.log` (the CLI writes to `logs/app.log`). Library modules obtain their logger with `logging.getLogger(__name__)` and attach no handlers. The formatter is `timestamp | level | module | message`. Because the file handler sits at DEBUG while the console sits at INFO, intermediate values — quartile thresholds, per-month medians, scaler parameters — are preserved in the log file for audit but never clutter a normal console run; `--debug` on any script raises the console to DEBUG. INFO marks milestones (loaded, saved, shape before/after), WARNING marks every change to the data with a count and reason, ERROR is used only with `exc_info=True` immediately before a graceful non-zero exit." },
  { p: "Reproducibility is closed by the repository itself: incremental commits per task, a `.gitignore` that keeps caches out, a `requirements.txt`, and a README that lists the three commands needed to regenerate every output in this report from the raw CSV." },
];

buildDocument({
  title: "Python Pipeline Methodology Report",
  subtitle: "Part 2 — Building a reproducible traffic analytics pipeline",
  meta: [["Student", "Sora"], ["Programme", "NUS/Emeritus Applied Machine Learning and Data Science"],
         ["Capstone", "Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility"]],
  blocks, outFile: OUT, footerText: "Part 2 — Python Methodology Report",
}).then((f) => console.log("written", f)).catch((e) => { console.error(e); process.exit(1); });
