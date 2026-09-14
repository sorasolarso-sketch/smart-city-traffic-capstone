# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

**NUS/Emeritus Applied Machine Learning and Data Science — Capstone Project**
**Student:** Sora

An end-to-end traffic intelligence solution built on 48,204 hourly records of
westbound I-94 traffic near Minneapolis–St Paul (October 2012 – September 2018),
with weather and US federal holiday data. The project moves from SQL and
statistics (Part 1), through a logged, reproducible Python pipeline (Part 2),
to supervised, unsupervised and deep learning models with explainability,
MLflow tracking, a FastAPI deployment mock-up, drift monitoring and a
travel-timing recommendation system (Part 3).

> **Accident data.** No accident dataset was provided with this capstone.
> Following the brief, a **documented proxy label** is used for the
> classification task: an hour is *high risk* when High/Severe congestion
> (upper two quartiles of traffic volume) coincides with severe or
> low-visibility weather. This proxy contains no information about actual
> collisions and is **not** an accident prediction. Every artefact that
> exposes it — code, API, model registry and both reports — says so.

---

## Repository structure

```
smart-city-traffic-capstone/
├── README.md                      ← this file
├── run_all.sh                     ← rebuilds every output from the raw CSV
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/Metro_Interstate_Traffic_Volume.csv
│   └── processed/                 ← traffic_clean.csv, traffic_features.csv, feature_metadata.json
├── docs/
│   ├── Submission_Template.docx   ← completed GitHub submission template
│   └── report_builder/            ← scripts that generate the four Word reports
├── part1_data_analytics/
│   ├── sql/                       ← 01_load_and_verify, 02_annual_traffic_trends, 03_holiday_temperature
│   ├── run_sql_analysis.py        ← builds traffic.db, runs every query, writes outputs/sql_results.md
│   ├── statistics_probability.py  ← Tasks 2 and 3
│   ├── powerbi_prep.py            ← Task 4 data prep and pre-computed dashboard answers
│   ├── powerbi/                   ← power_query_script.m, traffic_powerbi_ready.csv, DASHBOARD_BUILD_GUIDE.md
│   ├── outputs/                   ← traffic.db, sql_results.md, statistics_probability_results.{md,json}, powerbi_answers.json
│   ├── report/Part1_Data_Analytics_Insights_Report.docx
│   └── README.md
├── part2_python/
│   ├── logging_config.py          ← shared logging setup (console INFO + file DEBUG)
│   ├── pipeline.py                ← Task 1: load, validate schema, clean (each step logged)
│   ├── feature_engineering.py     ← Task 2: 57 engineered features + congestion target
│   ├── visualizations.py          ← Task 3: six figures + INTERPRETATIONS.md
│   ├── viz_style.py               ← shared plotting style
│   ├── mini_app/traffic_app.py    ← Task 4: CLI with six commands
│   ├── figures/                   ← generated PNGs and interpretations
│   ├── logs/pipeline.log, app.log ← sample log output
│   ├── report/Part2_Python_Methodology_Report.docx
│   └── README.md
└── part3_machine_learning/
    ├── src/
    │   ├── common.py              ← feature set, leakage guard, chronological split, proxy label, MLflow setup
    │   ├── supervised.py          ← Task 1: classification + regression, MLflow-tracked
    │   ├── unsupervised.py        ← Task 2: K-means + association rules
    │   ├── deep_learning.py       ← Task 3: LSTM + SHAP (on a surrogate tree model)
    │   ├── recommender.py         ← Task 5: travel-timing recommendation system
    │   ├── monitoring.py          ← Task 6.4/6.5: drift monitoring + PASS/ALERT dashboard
    │   └── model_registry.py      ← Task 6.1/6.2: MLflow model registry + version table
    ├── api/app.py, test_api.py    ← Task 6.3: FastAPI deployment mock-up + smoke tests
    ├── models/                    ← trained .joblib and .keras artefacts
    ├── mlflow.db, mlruns/         ← MLflow tracking store and artefacts
    ├── outputs/                   ← results JSON, MODEL_REGISTRY.md, MONITORING_DASHBOARD.md, clustered_hours.csv
    ├── figures/                   ← 11 generated PNGs
    ├── logs/part3.log, api.log
    ├── reports/Final_Capstone_Report.docx, Bias_and_Fairness_Report.docx
    └── README.md
```

## Tools and technologies

| Area | Tools |
| --- | --- |
| Data analytics | SQLite 3, Python 3.11, pandas, NumPy, SciPy, Power BI Desktop (Power Query M, DAX) |
| Pipeline and visualisation | pandas, NumPy, Matplotlib, Python `logging`, `argparse` |
| Machine learning | scikit-learn (logistic regression, random forest, gradient boosting, K-means), mlxtend (Apriori) |
| Deep learning and explainability | TensorFlow/Keras (LSTM), SHAP (TreeExplainer) |
| MLOps | MLflow (tracking + model registry, SQLite backend), FastAPI + Pydantic + Uvicorn, custom PSI/KS drift monitor |
| Reporting | docx (Node.js) for the Word reports |

## Quick start

```bash
git clone https://github.com/<your-username>/smart-city-traffic-capstone.git
cd smart-city-traffic-capstone
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

bash run_all.sh            # everything, ~12 min on a laptop CPU
bash run_all.sh --quick    # everything except the LSTM + SHAP stage, ~2 min
```

`run_all.sh` runs the stages below in order. Each can also be run on its own.

### Part 1 — Data analytics

```bash
python part1_data_analytics/run_sql_analysis.py         # SQLite load, verification, trend + holiday queries
python part1_data_analytics/statistics_probability.py   # descriptive stats, correlation, probability
python part1_data_analytics/powerbi_prep.py             # Power BI ready CSV + verified dashboard answers
```

The Power BI dashboard itself is built in Power BI Desktop (Windows) from
`part1_data_analytics/powerbi/power_query_script.m` following
`part1_data_analytics/powerbi/DASHBOARD_BUILD_GUIDE.md`, which contains every
answer the dashboard must show, pre-computed from the data.

### Part 2 — Python pipeline

```bash
python part2_python/pipeline.py                 # raw CSV → data/processed/traffic_clean.csv
python part2_python/feature_engineering.py      # → data/processed/traffic_features.csv (67 columns)
python part2_python/visualizations.py           # → part2_python/figures/*.png + INTERPRETATIONS.md
```

Add `--debug` to any of these to see DEBUG-level output (quartile thresholds,
per-month medians, scaler parameters) on the console; it is always written to
the log file.

### Part 2 — Mini application

```bash
python part2_python/mini_app/traffic_app.py summary
python part2_python/mini_app/traffic_app.py query 2017-08-31 --hour 17
python part2_python/mini_app/traffic_app.py peak --top 5 --day-type weekday
python part2_python/mini_app/traffic_app.py compare
python part2_python/mini_app/traffic_app.py recommend --day-type weekday --earliest 7 --latest 19
python part2_python/mini_app/traffic_app.py weather --condition Snow
python part2_python/mini_app/traffic_app.py --help
```

### Part 3 — Machine learning

```bash
python part3_machine_learning/src/supervised.py        # classification + regression, tracked in MLflow
python part3_machine_learning/src/unsupervised.py      # K-means + association rules
python part3_machine_learning/src/deep_learning.py     # LSTM + SHAP (add --epochs 5 for a quick run)
python part3_machine_learning/src/monitoring.py        # drift monitoring + PASS/ALERT dashboard
python part3_machine_learning/src/model_registry.py    # register versions, write MODEL_REGISTRY.md
python part3_machine_learning/src/recommender.py --date 2018-07-04 --weather Rain --earliest 6 --latest 20
```

MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///part3_machine_learning/mlflow.db
```

API (deployment simulation):

```bash
uvicorn part3_machine_learning.api.app:app --reload --port 8000
# open http://127.0.0.1:8000/docs
python part3_machine_learning/api/test_api.py          # five in-process smoke tests
```

## Models

| Model | Task | Algorithm | Test metric (chronological hold-out, Oct 2017 – Sep 2018) |
| --- | --- | --- | --- |
| `traffic_volume_regressor` (champion) | hourly volume | HistGradientBoostingRegressor | MAE 225.8 vehicles/h (6.8% of mean), R² 0.964 |
| `traffic_volume_regressor` (challenger) | hourly volume | Linear regression | MAE 645.0, R² 0.809 |
| `traffic_lstm_forecaster` | next-hour volume from 24 h history | LSTM 64→32 | see `outputs/deep_learning_results.json` |
| `traffic_risk_classifier` (champion) | **proxy** risk label | Random Forest, 300 trees | ROC AUC 0.996, F1 0.948 |
| `traffic_risk_classifier` (challenger) | **proxy** risk label | Logistic regression | ROC AUC 0.993, F1 0.896 |

All models share one 44-feature set (time, cyclical encodings, weather
encodings and indicators, holiday flag). Target-derived columns are barred
as leakage in `common.py`. **Every metric uses a chronological split** —
the random-split figure is reported alongside only to quantify its optimism.
The exact current numbers are in `part3_machine_learning/outputs/*.json` and
`part3_machine_learning/outputs/MODEL_REGISTRY.md`.

## Logging configuration

Every module obtains its logger with `logging.getLogger(__name__)` and attaches
no handlers. Handlers are configured **only** in entry-point scripts, via
`configure_logging()` in `part2_python/logging_config.py`, which attaches:

| Handler | Level | Destination |
| --- | --- | --- |
| Console | INFO (DEBUG with `--debug`) | stdout |
| File | DEBUG | `part2_python/logs/pipeline.log` (Part 2 pipeline), `part2_python/logs/app.log` (CLI), `part3_machine_learning/logs/part3.log` (Part 3), `part3_machine_learning/logs/api.log` (API) |

Format: `timestamp | LEVEL | module | message`.

| Level | Used for |
| --- | --- |
| DEBUG | Intermediate values useful only for troubleshooting: quartile thresholds, per-month medians, scaler parameters, cluster centres, per-epoch loss. Never shown on a normal console run. |
| INFO | Expected milestones: file loaded (with rows × columns), stage completed, shape before/after feature engineering, figure or model saved, CLI command invoked with its arguments. |
| WARNING | Recoverable events that change the data or need attention: rows dropped, values imputed, outliers capped (always with count and reason), coverage gaps, class imbalance, monitoring ALERTs. |
| ERROR | A failure that stops a stage; always logged with `exc_info=True` immediately before a graceful non-zero exit. CLI input errors are logged as one ERROR line and shown to the user as a readable message, never a traceback. |

`print()` is used only in the CLI application and the recommender, only for
the answer the user asked for.

## Reproducing the main results

1. `bash run_all.sh` regenerates every table, figure, model, log and JSON file.
2. `node docs/report_builder/build_part1_report.js` (and `build_part2_report.js`,
   `build_final_report.js`, `build_bias_fairness_report.js`) regenerate the
   four Word reports from the JSON outputs, so the reports never drift from
   the numbers. Requires Node.js with the `docx` package.
3. The LSTM runs with TensorFlow op determinism enabled and a fixed seed, so
   re-runs give identical metrics on the same hardware.

## Assumptions and limitations

- **Single corridor, single direction, single weather station.** Nothing here
  generalises beyond westbound I-94 at this sensor without new data.
- **22.8% of hours are missing**, mostly in 2014–2015. Annual totals are
  therefore reported per hour and seasonally adjusted, and time-series
  visuals mask months with fewer than 400 recorded hours.
- **Duplicate timestamps** (7,629 rows) are the weather feed's
  one-row-per-condition convention, not errors. One row per hour is kept,
  retaining the most severe condition.
- **The risk label is a proxy** (see the note at the top). The classifier's
  high scores reflect the circularity of that label, not predictive insight
  into collisions.
- **Model error is uneven**: roughly 7× higher at the 16:00 peak than at
  03:00, ~29% higher on weekends than weekdays, and higher in winter. See the
  bias and fairness report.
- **The data predates 2020**; any operational use would need retraining on
  current data.
- The Power BI `.pbix` is not committed because Power BI Desktop is
  Windows-only; the complete Power Query script, prepared data and a
  step-by-step build guide with verified answers are provided instead.
