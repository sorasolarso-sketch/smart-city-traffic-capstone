# Part 3 — Machine Learning and AI: An Intelligent Mobility Solution

Supervised, unsupervised and deep learning models on the Part 2 feature
table, with SHAP explainability, MLflow experiment tracking and a model
registry, a FastAPI deployment mock-up, drift monitoring with PASS/ALERT
reporting, and a travel-timing recommendation system.

> **Accident dataset: none was sourced.** A **documented proxy label** is
> used, exactly as the brief specifies: `high_risk = 1` when
> `congestion_category ∈ {High, Severe}` (upper two quartiles of traffic
> volume) **and** the weather is severe (Rain, Snow, Thunderstorm, Squall,
> Drizzle) or low-visibility (Fog, Mist, Haze, Smoke, Snow, Squall). This
> label contains no information about real collisions; the classifier
> trained on it is a *congestion-in-bad-weather* detector and is labelled as
> such in the API, the registry and the reports. See
> `reports/Bias_and_Fairness_Report.docx`, Section 1.2.

## Structure

```
part3_machine_learning/
├── src/
│   ├── common.py            feature set (44 cols), leakage guard, chronological split, proxy label, MLflow helpers
│   ├── supervised.py        Task 1 — logistic regression + random forest; linear regression + gradient boosting
│   ├── unsupervised.py      Task 2 — K-means (k chosen by elbow/silhouette) + Apriori association rules
│   ├── deep_learning.py     Task 3 — LSTM on 24-hour windows + SHAP on a surrogate random forest
│   ├── recommender.py       Task 5 — travel-timing recommendations (historical and model modes)
│   ├── monitoring.py        Task 6.4/6.5 — PSI/KS feature drift + error drift, calibrated thresholds, PASS/ALERT
│   └── model_registry.py    Task 6.1/6.2 — MLflow Model Registry, champion/challenger aliases, MODEL_REGISTRY.md
├── api/
│   ├── app.py               Task 6.3 — FastAPI: /health /models /predict/volume /predict/risk /recommend
│   └── test_api.py          five in-process smoke tests
├── models/                  classifier_*.joblib, regressor_*.joblib, lstm_traffic_forecast.keras
├── mlflow.db  mlruns/       MLflow tracking store (SQLite) and artefacts
├── outputs/
│   ├── supervised_results.json, unsupervised_results.json, deep_learning_results.json
│   ├── monitoring_report.json, MONITORING_DASHBOARD.md
│   ├── MODEL_REGISTRY.md, experiment_summary.json
│   ├── clustered_hours.csv, recommender_profile.json
├── figures/                 07–17: importances, ROC, regression diagnostics, k selection, clusters,
│                            rules, LSTM history and predictions, SHAP summary and bar, monitoring
├── logs/part3.log, api.log
├── reports/
│   ├── Final_Capstone_Report.docx        methodology and findings across every task, all three parts
│   └── Bias_and_Fairness_Report.docx     Task 7: bias, fairness, governance, sustainability
└── README.md
```

## Run

Prerequisite: Part 2 has produced `data/processed/traffic_features.csv`.
From the repository root:

```bash
export MLFLOW_DISABLE_AGENT_HINT=1                    # optional, quietens MLflow
python part3_machine_learning/src/supervised.py       # ~1 min
python part3_machine_learning/src/unsupervised.py     # ~30 s
python part3_machine_learning/src/deep_learning.py    # ~7 min on CPU; --epochs 5 for a quick run; --skip-shap
python part3_machine_learning/src/monitoring.py       # ~20 s (needs the regressor from supervised.py)
python part3_machine_learning/src/model_registry.py   # registers versions, writes outputs/MODEL_REGISTRY.md
```

Or everything, including Parts 1 and 2: `bash run_all.sh` from the root.

### Recommendation system

```bash
python part3_machine_learning/src/recommender.py --day-type weekday --earliest 7 --latest 19 --duration 2
python part3_machine_learning/src/recommender.py --date 2018-07-04 --weather Rain --earliest 6 --latest 20
python part3_machine_learning/src/recommender.py --day-type weekend --json
```

Historical mode ranks hours by matching history (day type, and weather when
≥ 200 hours exist). Giving `--date` switches to model mode: the champion
regressor predicts all 24 hours of that date under the assumed weather, so
holidays and season are respected. Output is ranked windows plus a
plain-language sentence, e.g. *"For a weekday journey, consider travelling
between 19:00 and 21:00, when historical traffic volumes average 3,105
vehicles per hour — 13% below the daily average of 3,572."*

### MLflow

```bash
mlflow ui --backend-store-uri sqlite:///part3_machine_learning/mlflow.db
```

Three experiments (`traffic-accident-risk-classification`,
`traffic-volume-regression`, `traffic-lstm-demand-forecast`), each run
logging hyperparameters (`hp_*`), split details, every metric and the model
artefact. The registry holds `traffic_risk_classifier`,
`traffic_volume_regressor` and `traffic_lstm_forecaster` with `champion` /
`challenger` aliases. `outputs/MODEL_REGISTRY.md` is the human-readable
version table.

### API

```bash
uvicorn part3_machine_learning.api.app:app --reload --port 8000
curl -X POST http://127.0.0.1:8000/predict/volume -H "Content-Type: application/json" \
     -d '{"date":"2018-07-04","hour":17,"weather":"Rain","temp_celsius":24}'
python part3_machine_learning/api/test_api.py
```

Interactive docs at `http://127.0.0.1:8000/docs`. Every `/predict/risk`
response carries the proxy-label disclaimer.

## Method notes

- **Feature set (44 columns):** time features incl. cyclical encodings of
  hour and day of week (and month, day of year), weather encodings and
  derived indicators, holiday flag, one-hot weather. `traffic_volume`,
  `congestion_category`, `congestion_level`, `is_congested` and `high_risk`
  are barred as leakage in `common.get_feature_columns`; `year` is excluded
  because the split is chronological.
- **Split:** first 80% of the timeline trains (to 2017-10-26), last 20%
  tests. The random-split score is computed alongside and reported only to
  quantify its optimism.
- **Baselines:** majority-class accuracy for classification, mean prediction
  for regression, persistence (previous hour) for the LSTM — every model is
  reported against one.
- **LSTM sequences** are built only inside unbroken hourly runs (the sensor
  outages would otherwise be spliced into fake days), then sorted by time
  before the split. Op determinism is enabled for reproducibility.
- **SHAP** is applied to a Random Forest on the same target (MAE within ~5%
  of the LSTM), because a 24 × 10 sequence attribution does not give a
  mobility team anything to act on. The report explains and bounds this.
- **Monitoring** compares each held-out month with the *same calendar month*
  in training, uses a season-spanning out-of-sample error baseline per
  calendar month, and calibrates each feature's PSI alert threshold to the
  95th percentile of month-to-month variation within the training years
  (floor 0.25). Three synthetic drift scenarios verify the alarm fires.

## Results

Current numbers are in `outputs/*.json` and `outputs/MODEL_REGISTRY.md`;
headline figures are summarised in the root README and discussed in
`reports/Final_Capstone_Report.docx`.

## Logging

All scripts use `logging.getLogger(__name__)` and the shared
`configure_logging()` from `part2_python/logging_config.py`: console at INFO,
file at DEBUG (`logs/part3.log`; the API writes `logs/api.log`). WARNING is
used for rows dropped, class imbalance, non-reproducible splits detected,
and every monitoring ALERT; ERROR only with `exc_info=True` before a graceful
exit. `print()` appears only in `recommender.py`, for the recommendation
itself.
