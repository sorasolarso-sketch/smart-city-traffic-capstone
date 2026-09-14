const path = require("path");
const fs = require("fs");
const { buildDocument } = require("./report_lib");

const ROOT = path.resolve(__dirname, "../..");
const P1 = path.join(ROOT, "part1_data_analytics");
const P2 = path.join(ROOT, "part2_python");
const P3 = path.join(ROOT, "part3_machine_learning");
const OUT = path.join(P3, "reports/Final_Capstone_Report.docx");
const J = (f) => JSON.parse(fs.readFileSync(f, "utf8"));
const F2 = (f) => path.join(P2, "figures", f);
const F3 = (f) => path.join(P3, "figures", f);

const sp = J(path.join(P1, "outputs/statistics_probability_results.json"));
const pb = J(path.join(P1, "outputs/powerbi_answers.json"));
const fm = J(path.join(ROOT, "data/processed/feature_metadata.json"));
const sv = J(path.join(P3, "outputs/supervised_results.json"));
const un = J(path.join(P3, "outputs/unsupervised_results.json"));
const dl = J(path.join(P3, "outputs/deep_learning_results.json"));
const mo = J(path.join(P3, "outputs/monitoring_report.json"));

const prob = sp.task3_probability.all_rows;
const rf = sv.classification.models.random_forest.metrics;
const lr = sv.classification.models.logistic_regression.metrics;
const hgb = sv.regression.models.hist_gradient_boosting.metrics;
const lin = sv.regression.models.linear_regression.metrics;
const base = sv.regression.baseline_mean_prediction;
const lstm = dl.lstm.metrics;
const shapTop = Object.entries(dl.shap.top_10);
const km = un.kmeans;
const rules = un.association_rules.top_rules_by_lift;
const f0 = (x) => Number(x).toLocaleString("en-US", { maximumFractionDigits: 0 });
const f1 = (x) => Number(x).toFixed(1), f2 = (x) => Number(x).toFixed(2), f3 = (x) => Number(x).toFixed(3), f4 = (x) => Number(x).toFixed(4);
const pct1 = (x) => `${(100 * x).toFixed(1)}%`;
const hh = (h) => String(h).padStart(2, "0") + ":00";
const byHour = sv.regression.mae_by_hour;
const hours = Object.keys(byHour).map(Number);
const worstHour = hours.reduce((a, b) => (byHour[a] > byHour[b] ? a : b));
const bestHour = hours.reduce((a, b) => (byHour[a] < byHour[b] ? a : b));
const b17 = pb.task_4_2_and_4_3.B_hourly_2017;
const lstmBetter = lstm.mae < hgb.mae;
const lstmGap = Math.abs(hgb.mae - lstm.mae);
const cW = pb.task_4_2_and_4_3.C_weather_impact;

const blocks = [
  { h1: "Executive summary" },
  { p: "This capstone takes 48,204 hourly records of westbound I-94 traffic near Minneapolis–St Paul and carries them from a SQL query to a monitored, versioned, API-served forecasting system with a plain-language recommendation layer. Three parts build on one another: Part 1 establishes what the data says and, as importantly, where it lies; Part 2 makes that analysis reproducible as a logged Python pipeline with a 67-column feature table; Part 3 trains, explains, tracks, deploys and monitors models on top of it." },
  { p: "The headline results are strong and the report is careful about what they mean. The gradient-boosting regressor predicts hourly volume to within **" + f1(hgb.mae) + " vehicles (" + f1(hgb.mae_as_pct_of_mean) + "% of the mean, R² " + f3(hgb.r2) + ")** on a strictly chronological hold-out; the LSTM reaches **" + f1(lstm.mae) + " (R² " + f3(lstm.r2) + ")**" + (lstmBetter ? ", a modest improvement" : " — close, but not better, once its evaluation split was made properly chronological") + ". The proxy accident-risk classifier reaches ROC AUC **" + f4(rf.roc_auc) + "**, a figure that reflects the circularity of a label derived from the model's own inputs rather than any insight into collisions, and it is presented as such throughout. Four findings carry through every part: the evening peak, not the morning one, is the corridor's binding constraint; weather has almost no independent effect on volume once time of day is controlled; the raw feed's duplicate hours and impossible readings would have inverted several conclusions had they not been repaired first; and a model's average error hides a " + f1(byHour[worstHour] / byHour[bestHour]) + "-fold gap between its best and worst hours." },
  { callout: "**On the accident dataset.** None was provided. Following the brief, a proxy label was constructed: an hour is *high risk* when High/Severe congestion coincides with severe or low-visibility weather. Every artefact that touches this label — code, API, registry, README and both reports — states that it is a proxy and not an accident prediction. The bias and fairness report examines the consequences in detail." },

  { h1: "1. The data, and what had to be fixed before it could be trusted" },
  { p: "The dataset spans 2 October 2012 to 30 September 2018 with nine columns: holiday, temperature (Kelvin), rainfall and snowfall (mm/h), cloud cover (%), two weather text fields, a timestamp and the hourly westbound vehicle count. No column contains a null. That turned out to be the least informative fact about its quality." },
  { table: {
    header: ["Issue", "Extent", "Consequence if ignored", "Treatment"],
    align: ["left", "right", "left", "left"],
    widths: [2200, 1300, 3400, 3192],
    rows: [
      ["Duplicate timestamps (weather feed emits one row per condition)", "7,629 rows (15.8%)", "Annual totals inflated 6.9–21.1%; peak hours over-weighted in every average", "Keep one row per hour, retaining the most severe condition"],
      ["Temperature of 0 K", "10 rows", "Mean temperature and correlation distorted; scatter axis compressed", "Impute with same-calendar-month median (explicit loop over months)"],
      ["Rainfall of 9,831 mm/h", "1 row", "Any rainfall visual becomes a single pixel", "Impute with same-month median"],
      ["Zero vehicles mid-morning", "1 row (1 overnight zero retained)", "Spurious outlier in regression", "Impute with same-hour median"],
      ["Holiday label on 00:00 only; NY Day 2015 absent", "61 rows labelled", "Holiday analysis reduces to one midnight reading", "Select holidays by calendar date"],
      ["Sensor outages", "11,976 hours (22.8%) missing, mostly 2014–15", "Raw yearly totals track uptime, not demand; unmarked gaps read as demand collapse", "Seasonally adjusted index; masked months in time-series visuals; contiguous-run detection for the LSTM"],
      ["pandas reads \"None\" as NaN", "48,143 rows", "Every holiday count wrong", "`keep_default_na=False` at load"],
    ],
  }},
  { p: "After treatment the working dataset is **40,575 distinct hours**. Each of these repairs is a separately logged step in `part2_python/pipeline.py`, and the case for each is made in the Part 1 insights report." },

  { h1: "2. Part 1 — Data analytics" },
  { h2: "2.1 SQL trend analysis" },
  { p: "Loaded into SQLite with an explicit typed schema and de-duplicated in a CTE, the yearly comparison shows raw totals swinging by +256% and −39% between successive years, entirely because recorded hours swung from 2,103 to 7,294 to 4,501. On average volume per hour the same changes are +2.6% and −1.2%. A month-adjusted index (each year's months against the pooled norm for those months) gives the defensible picture: demand slightly above norm in 2013 (+34 vehicles/hour), sliding to its low in 2016 (−89), then the largest single-year rise in the series to 2017 (+89). 2017 is the only year with near-complete coverage and is recommended as the planning baseline. Holiday temperatures moved in opposite directions on New Year's Day (+3.0 K, 2016→2017) and Labor Day (−3.7 K) while traffic rose on both, so year-on-year temperature change does not appear relevant to holiday traffic; the holiday effect itself (30–40% below ordinary days) dwarfs it." },
  { h2: "2.2 Statistics, correlation and probability" },
  { p: "Hourly volume: mean " + f1(sp.task2_1_descriptive.all_rows.mean) + ", median " + f0(sp.task2_1_descriptive.all_rows.median) + ", standard deviation " + f1(sp.task2_1_descriptive.all_rows.std_dev_sample) + ", variance " + f0(sp.task2_1_descriptive.all_rows.variance_sample) + ", range 0–7,280. The excess kurtosis of " + f2(sp.task2_1_descriptive.all_rows.kurtosis_excess) + " signals a bimodal distribution — an overnight mass near zero and a daytime mass around 4,000–6,000 — in which mean and median both describe a volume that rarely occurs. Temperature correlates with volume at Pearson r = " + f3(sp.task2_2_correlation.all_rows_as_given.pearson_r) + " (r² = " + f3(sp.task2_2_correlation.all_rows_as_given.r_squared) + "): positive, weak, and a textbook confound — warm hours and busy hours are both daytime summer hours." },
  { p: "With congestion defined as > 5,500 vehicles/hour: P(Congestion) = " + f4(prob.p_congestion) + ", P(Clear) = " + f4(prob.p_clear_weather) + ", P(Congestion ∩ Clear) = " + f4(prob.p_congestion_and_clear) + " against " + f4(prob.independence.p_a_times_p_b_expected) + " expected under independence; P(Clear | Congestion) = " + f4(prob.p_clear_given_congestion) + "; P(High temp | Congestion) = " + f4(prob.p_high_temp_given_congestion) + ". The events are not strictly independent (χ² = " + f1(prob.independence.chi_square) + ", p ≈ 10⁻⁹) but the effect size φ = " + f3(prob.independence.phi_coefficient) + " is negligible. The odds ratio of congestion in clear versus cloudy weather is **" + f3(prob.odds_ratio_clear_vs_cloudy.odds_ratio) + "** (95% CI " + f3(prob.odds_ratio_clear_vs_cloudy.ci95_low) + "–" + f3(prob.odds_ratio_clear_vs_cloudy.ci95_high) + "), which naively says clear skies reduce congestion by 26%. Controlling for hour of day, the clear-versus-cloudy gap in congestion rate collapses to **" + (100 * sp.task3_probability.hour_controlled.mean_rate_difference_within_hour).toFixed(2) + " percentage points**. Clear weather is a night-time condition and cloud an afternoon one; the odds ratio measures when weather happens, not how drivers respond to it." },
  { h2: "2.3 Power BI dashboard" },
  { p: "Power BI Desktop cannot run in the project's Linux environment, so the dashboard is delivered as a complete Power Query M script (16 named, auditable steps), the prepared 40,575 × 19 table it produces, DAX for the three KPI cards, and a build guide with every answer pre-computed and verified against the data. The dashboard's findings: 2017 hourly traffic peaks at **" + hh(b17.evening_peak_hour) + " (" + f0(b17.evening_peak_average) + " vehicles/hour)**, 21% above the morning peak at " + hh(b17.morning_peak_hour) + " (" + f0(b17.morning_peak_average) + "); the highest-average weather condition is **" + cW.highest_avg_weather + " (" + f0(cW.highest_avg_value) + ")** and the lowest interpretable one **" + cW.reliable_only_min_100_hours.lowest_weather + " (" + f0(cW.reliable_only_min_100_hours.lowest_value) + ")**, a gap of " + f0(cW.reliable_only_min_100_hours.difference) + " — Squall's literal minimum of 420 rests on one hour and is flagged as uninterpretable; KPI cards show " + f0(pb.task_4_2_and_4_3.KPI_cards.total_hours_analysed) + " hours, average volume " + f1(pb.task_4_2_and_4_3.KPI_cards.average_traffic_volume) + " and average temperature " + f1(pb.task_4_2_and_4_3.KPI_cards.average_temperature_celsius) + " °C." },

  { h1: "3. Part 2 — The reproducible pipeline" },
  { p: "Four scripts, each an entry point with its own `main()` and `--debug` flag, share one logging configuration that attaches a console handler at INFO and a file handler at DEBUG to the root logger; library code uses `logging.getLogger(__name__)` and attaches nothing. `pipeline.py` validates the schema before any processing, then performs the repairs in Section 1 as separately logged steps, each WARNING carrying the row count and the reason. `feature_engineering.py` adds 57 columns without dropping a row: calendar and rush-hour features, sine/cosine encodings of hour, weekday, month and day-of-year, an ordinal weather-severity score with binary condition indicators and 11 one-hot columns, z-score and min-max versions of four continuous variables, and two targets — the quartile-based `congestion_category` (Low ≤ " + f0(fm.congestion_thresholds.q1_25th_percentile) + ", Medium ≤ " + f0(fm.congestion_thresholds.q2_median) + ", High ≤ " + f0(fm.congestion_thresholds.q3_75th_percentile) + ", Severe above; 25% of hours each, thresholds logged at DEBUG) and the fixed-threshold `is_congested` from Part 1. `visualizations.py` produces six figures with written interpretations; `mini_app/traffic_app.py` exposes six commands, logs every invocation with its arguments, and converts every invalid input into a single ERROR line and a readable message rather than a traceback." },
  { image: F2("03_hour_day_heatmap.png"), width: 600, caption: "Figure 1 — Hour × weekday heatmap. Weekdays share a twin-peak signature; weekends lose the morning block entirely. The ringed cell is the busiest hour of the week." },

  { h1: "4. Part 3 — Machine learning and AI" },
  { h2: "4.1 Feature set and evaluation protocol" },
  { p: "All Part 3 models share one feature set of **" + sv.feature_count + " columns**: time features including the cyclical encodings, weather encodings and derived indicators, the holiday flag, and the one-hot weather columns. Five columns are barred as leakage — `traffic_volume`, `congestion_category`, `congestion_level`, `is_congested` and `high_risk` — because each is derived from the target; the guard is enforced in code and logged. `year` is excluded by design because the split is chronological and a tree model cannot extrapolate to an unseen year." },
  { p: "**Every headline metric uses a chronological split**: the first 80% of the timeline (to 26 October 2017) trains, the final 20% (to 30 September 2018) tests. A random split would place 15:00 and 17:00 of the same afternoon on opposite sides and let the model read answers off its neighbours. The random-split figure is computed alongside and reported only to quantify the optimism: it adds " + f4(rf.optimism_from_random_split) + " to the Random Forest's AUC. Small here, because time features already capture most of the signal; it would not be small on a noisier target." },

  { h2: "4.2 Supervised learning (Task 1)" },
  { table: {
    header: ["Classification — proxy risk", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "Avg precision"],
    align: ["left", "right", "right", "right", "right", "right", "right"],
    widths: [2800, 1200, 1200, 1200, 1100, 1200, 1392],
    rows: [
      ["Majority-class baseline", f4(sv.classification.baseline_majority_class.accuracy), "—", "—", "—", "0.5000", "—"],
      ["Logistic regression (balanced)", f4(lr.accuracy), f4(lr.precision), f4(lr.recall), f4(lr.f1), f4(lr.roc_auc), f4(lr.average_precision)],
      ["Random Forest (300 trees)", f4(rf.accuracy), f4(rf.precision), f4(rf.recall), f4(rf.f1), "**" + f4(rf.roc_auc) + "**", f4(rf.average_precision)],
    ],
  }},
  { table: {
    header: ["Regression — hourly volume", "MAE (veh/h)", "MAE % of mean", "RMSE", "R²", "Train time"],
    align: ["left", "right", "right", "right", "right", "right"],
    widths: [3000, 1500, 1500, 1300, 1200, 1592],
    rows: [
      ["Mean-prediction baseline", f1(base.mae), f1(base.mae_as_pct_of_mean) + "%", f1(base.rmse), f4(base.r2), "—"],
      ["Linear regression (scaled)", f1(lin.mae), f1(lin.mae_as_pct_of_mean) + "%", f1(lin.rmse), f4(lin.r2), f2(lin.training_seconds) + " s"],
      ["HistGradientBoosting", "**" + f1(hgb.mae) + "**", f1(hgb.mae_as_pct_of_mean) + "%", f1(hgb.rmse), "**" + f4(hgb.r2) + "**", f2(hgb.training_seconds) + " s"],
      ["LSTM (Task 3, for comparison)", "**" + f1(lstm.mae) + "**", f1(lstm.mae_as_pct_of_mean) + "%", f1(lstm.rmse), "**" + f4(lstm.r2) + "**", f0(lstm.training_seconds) + " s"],
    ],
  }},
  { p: "Both classifiers sit near the ceiling, and the top Random Forest importances — `weather_severity`, `adverse_conditions_score`, `hour_cos`, `is_severe_weather` — say why: the label is a function of weather (given) and volume (inferable from the clock). Recall of " + pct1(rf.recall) + " with precision " + pct1(rf.precision) + " means the forest misses fewer than 2 in 100 proxy-positive hours and raises a false flag on about 8 in 100 of its alerts; balanced class weights were used because the positive class is only " + pct1(rf.positive_rate_actual) + " of the test period. On regression, gradient boosting removes 65% of the linear model's error, and the linear model's residual pattern shows why — it cannot represent the twin-peak day with additive terms. The regression diagnostics figure shows the error is not uniform: MAE at " + hh(worstHour) + " is " + f0(byHour[worstHour]) + " vehicles against " + f0(byHour[bestHour]) + " at " + hh(bestHour) + ", a " + f1(byHour[worstHour] / byHour[bestHour]) + "× spread that the fairness report takes up." },
  { image: F3("09_regression_diagnostics.png"), width: 600, caption: "Figure 2 — Gradient-boosting diagnostics. Left: predicted against actual on the held-out year. Right: mean absolute error by hour of day — error scales with volume." },

  { h2: "4.3 Unsupervised learning (Task 2)" },
  { p: "**K-means** was run on hour, weather severity, traffic volume and temperature (standardised). The elbow criterion chose k = " + km.chosen_k + "; the silhouette peaked at k = " + km.k_selection.silhouette_best_k + " but every value lay between 0.32 and 0.35, and the script logs a warning to that effect: traffic conditions form a continuum, and the clusters are a useful partition of it rather than discovered categories. The four operating states are:" },
  { table: {
    header: ["Cluster", "Hours", "Share", "Mean volume", "Mean temp", "Reading"],
    align: ["left", "right", "right", "right", "right", "left"],
    widths: [900, 1000, 900, 1300, 1200, 4792],
    rows: Object.entries(km.cluster_profiles).map(([id, p]) => [
      id, f0(p.hours), f1(p.share_pct) + "%", f0(p.mean_traffic_volume), f1(p.mean_temp_celsius) + " °C", p.label,
    ]),
  }},
  { p: "Cluster 0 is the corridor asleep (84% Low congestion). Cluster 1 is daytime under adverse weather, and is the only cluster whose congestion mix is spread almost evenly across Medium, High and Severe — adverse weather does not empty the road, it flattens the peaks. Clusters 2 and 3 are the same daytime traffic in summer and in winter; temperature, not volume, separates them, which is itself evidence that the corridor's daytime demand is remarkably insensitive to season." },
  { p: "**Association rules** were mined with Apriori (support ≥ 2%, confidence ≥ 30%) over baskets of discretised time band, day type, weather category, freezing flag and holiday flag, keeping only rules whose consequent is a congestion level. " + f0(un.association_rules.congestion_rules) + " such rules survived from " + f0(un.association_rules.total_rules) + ". The highest-lift rules are unanimous about one thing:" },
  { table: {
    header: ["Rule", "Support", "Confidence", "Lift"],
    align: ["left", "right", "right", "right"],
    widths: [6000, 1200, 1400, 1492],
    rows: rules.slice(0, 6).map((r) => [
      r.antecedent.replace(/time=|day=|weather=|temp=/g, "") + "  →  " + r.consequent.replace("congestion=", ""),
      f3(r.support), f3(r.confidence), f2(r.lift),
    ]),
  }},
  { p: "In plain language, the top rule says: **when it is a weekend night and below freezing, congestion is Low " + pct1(rules[0].confidence) + " of the time, " + f2(rules[0].lift) + " times more often than Low occurs overall.** Every rule with lift above 3 concludes *Low* and has *Night* in its antecedent; freezing temperatures and weekends strengthen it. No rule predicting *Severe* clears the same bar, which is informative: the busy states are spread across many combinations of conditions, while the empty state is concentrated in a few. For a recommender that means the confident advice is always about when the road is clear, not about which busy hour is worst." },
  { image: F3("12_association_rules.png"), width: 580, caption: "Figure 3 — Top association rules by lift. All conclude Low congestion and all involve the night band." },

  { h2: "4.4 Deep learning with explainability (Task 3)" },
  { p: "An LSTM was chosen because the data is naturally sequential and the brief recommends it. The network (64 → 32 LSTM units with dropout, then a dense head; 32,161 parameters) predicts the next hour's volume from the preceding 24 hours of ten variables — past volume, temperature, weather severity, weekend and holiday flags, precipitation, and the cyclical hour and weekday encodings. Two implementation details matter more than the architecture. First, **sequences were built only within unbroken hourly runs**: the sensor outages mean consecutive rows are often not consecutive hours, and a naive sliding window would present a jump across a six-month gap as a continuous day. 2,589 runs were found, 173 exceed 24 hours, and " + f0(lstm.n_train_sequences + lstm.n_test_sequences) + " sequences were formed (7,552 rows could not be used). Second, **op determinism was enabled** after two otherwise identical runs produced MAEs of 201 and 233 through CPU thread ordering alone; the result is now bit-reproducible." },
  { p: "On the chronological hold-out the LSTM reaches **MAE " + f1(lstm.mae) + ", RMSE " + f1(lstm.rmse) + ", R² " + f4(lstm.r2) + "** after " + lstm.epochs_run + " epochs, against a persistence baseline (predict the previous hour) of " + f1(lstm.persistence_baseline_mae) + " — a " + f0(100 * lstm.improvement_over_persistence / lstm.persistence_baseline_mae) + "% reduction — and " + (lstmBetter ? f0(lstmGap) + " vehicles better than" : f0(lstmGap) + " vehicles worse than") + " gradient boosting, at " + f0(lstm.training_seconds / hgb.training_seconds) + "× the training cost. " + (lstmBetter ? "" : "An earlier run had reported MAE 201.8, comfortably ahead of gradient boosting; that figure was an artefact of a bug in which sequences were split in run-size order rather than time order, so the network was tested on hours it had trained beside. Corrected, the sequence model's advantage disappears. The honest reading is that on this corridor the clock and the calendar carry almost all of the predictable signal, and the previous 24 hours add little that a well-specified tabular model does not already know.") },
  { image: F3("14_lstm_predictions.png"), width: 600, caption: "Figure 4 — LSTM against actual volume over the first two weeks of the held-out period." },
  { p: "**Why SHAP is applied to a surrogate.** The brief allows explainability to be applied to a comparable tree or linear model when the chosen deep model does not explain cleanly, provided the choice is justified. An LSTM's attributions distribute across 24 timesteps × 10 features; a mobility team cannot act on \"the temperature 17 hours ago mattered\". SHAP was therefore run on a Random Forest trained on the same target with the same 44 tabular features (MAE " + f1(dl.shap.surrogate_metrics.mae) + ", R² " + f3(dl.shap.surrogate_metrics.r2) + " — within 5% of the LSTM, so it is a faithful proxy for what the problem rewards). TreeExplainer on a 1,200-row test sample gives the following mean absolute attributions, in vehicles per hour:" },
  { table: {
    header: ["Feature", "Mean |SHAP|", "Feature", "Mean |SHAP|"],
    align: ["left", "right", "left", "right"],
    widths: [3000, 1800, 3000, 1838],
    rows: [0, 1, 2, 3, 4].map((i) => [
      shapTop[i][0], f0(shapTop[i][1]), shapTop[i + 5][0], f0(shapTop[i + 5][1]),
    ]),
  }},
  { p: "Hour of day, through its cosine encoding, moves a prediction by " + f0(shapTop[0][1]) + " vehicles on average — more than every other feature combined. Day-of-week and weekend features come next; the first weather variable appears far down the list. This is the quantitative form of Part 1's conclusion: on this corridor, *when* explains traffic and *weather* barely does. The SHAP summary plot (repository figure 15) adds the direction: high `hour_cos` (small hours) pushes predictions strongly down, low `hour_cos` (afternoon) pushes them up, and `is_weekend` = 1 pushes down. The limit of the substitution is that the surrogate cannot speak to what the LSTM learned from the *sequence* — the recent trajectory of volume — which is the one thing the tabular model does not see" + (lstmBetter ? ", and which accounts for the LSTM's remaining edge." : ". That the LSTM does not outperform the tabular models suggests the trajectory adds little beyond what the clock already implies.") },
  { image: F3("16_shap_feature_importance.png"), width: 520, caption: "Figure 5 — Mean absolute SHAP value per feature (surrogate Random Forest)." },

  { h2: "4.5 Advanced technique: MLflow experiment tracking (Task 4)" },
  { p: "**Why.** Of the four options — quantisation, a GAN, self-supervised anomaly detection, MLflow — MLflow is the only one that changes how every other model in the project is *managed* rather than adding one more model, and it connects directly to the Task 6 MLOps requirements. Quantisation has little to offer a 32,000-parameter network already serving in under a millisecond; a GAN for synthetic traffic would need a downstream consumer this project does not have." },
  { p: "**How.** A SQLite-backed tracking store lives inside the repository (`part3_machine_learning/mlflow.db`), with artefacts under `mlruns/`. Three experiments are tracked — classification, regression and LSTM forecasting. Each run records the algorithm, every hyperparameter (`hp_*`), feature count, split method and row counts as parameters; every evaluation metric, including the random-split comparison, as metrics; and the fitted estimator as a logged model. The LSTM additionally logs per-epoch training and validation loss. `model_registry.py` then registers each logged model under a stable name (`traffic_volume_regressor`, `traffic_risk_classifier`, `traffic_lstm_forecaster`), assigns `champion` and `challenger` aliases by the ranking metric, tags each version with its score, and is idempotent. The UI opens with `mlflow ui --backend-store-uri sqlite:///part3_machine_learning/mlflow.db`." },
  { p: "**Value.** Every number in this report can be traced to a run ID with its exact parameters. When the feature-selection bug described in Section 5 was fixed, the before-and-after runs sat side by side in the UI. Re-running a script adds runs rather than overwriting, so the history is the audit trail. **Limitations.** A file-backed SQLite store is single-user and not a server; MLflow tracks what the code tells it and nothing else, so a parameter that is not logged is not tracked; and the registry's aliases are labels, not enforcement — nothing stops the API from loading a challenger. Each of these is what a production MLflow deployment (a tracking server, a database, access control) exists to address." },

  { h2: "4.6 Travel-timing recommendation system (Task 5)" },
  { p: "Because the data describes one carriageway, the recommender optimises *when* rather than *which way*. It has two modes. **Historical mode** ranks candidate departure hours by what the corridor has actually carried under matching conditions — day type, and weather where at least 200 hours of matching history exist — from a precomputed profile. **Model mode**, triggered by a specific date, builds the full 44-feature row for each hour of that day under the assumed weather and temperature and asks the champion regressor to predict it, so holidays and season are accounted for. Both score windows of any requested length as the mean expected volume across the hours spanned, respect an earliest/latest departure range, classify each window against the quartile thresholds, and produce a plain-language sentence with caveats. Two sample outputs:" },
  { callout: "*Weekday, 07:00–19:00 range, two-hour journey (historical mode):* \"For a weekday journey, consider travelling between 19:00 and 21:00, when historical traffic volumes average 3,105 vehicles per hour — 13% below the daily average of 3,572. Avoid 16:00–18:00, the busiest window in your range at 6,032. If that does not suit, 18:00–20:00 is the next best option.\"" },
  { callout: "*Wednesday 4 July 2018, rain, 06:00–20:00 (model mode):* \"For your journey on Wednesday 04 July 2018 in rain, consider travelling between 20:00 and 21:00, when predicted traffic volumes average 2,921 vehicles per hour — 2% below the daily average of 2,983. Avoid 16:00–17:00, the busiest window in your range at 4,605.\" Caveat attached: the date is a public holiday and the model has seen few of them." },
  { p: "The second example is the more interesting: the model recognises the holiday, flattens the morning peak and lowers the whole day, and the recommender flags its own uncertainty. The system is exposed as a CLI (`recommender.py`), as a Python function, and as the `/recommend` endpoint of the API." },

  { h2: "4.7 MLOps: versioning, deployment and monitoring (Task 6)" },
  { p: "**Versioning and tracking (6.1, 6.2)** are covered by the MLflow setup in 4.5; `outputs/MODEL_REGISTRY.md` is the human-readable version table, with a stated promotion policy — a challenger is promoted by a person, after review of the monitoring dashboard, never automatically." },
  { p: "**Deployment (6.3).** `api/app.py` is a FastAPI service that loads the champion regressor and classifier at start-up and exposes `/health`, `/models`, `/predict/volume`, `/predict/risk` and `/recommend`, with Pydantic validation (an hour of 25 or a weather of \"Tornado\" returns a 422 with the accepted values), OpenAPI documentation at `/docs`, and a disclaimer in every risk response. Five smoke tests run the app in-process: health, peak-versus-night volume ordering, risk ordering, clean validation errors, and a full recommendation. It is explicitly a simulation — one process, no authentication, no rate limiting — and the report to the mobility team lists what production would add." },
  { p: "**Monitoring and alerting (6.4, 6.5).** `monitoring.py` replays the champion regressor month by month across the held-out year and checks two things: prediction-error drift (MAE against a baseline) and feature-distribution drift (Population Stability Index and Kolmogorov–Smirnov statistic for seven inputs). Getting the baselines right was the substance of the task. A first version compared each month against the whole five-year training period and fired on all eleven months, because January's temperature distribution *is* different from the all-season pooled one — that is seasonality, not drift. The corrected design compares each month with **the same calendar month** in the training years, judges error against a **season-spanning out-of-sample baseline** (a refit on all but the last training year, scored per calendar month), and **calibrates each feature's PSI threshold to the 95th percentile of the month-to-month variation the training years themselves exhibited**, never below the conventional 0.25. With that, " + mo.windows_pass + " of " + mo.windows_evaluated + " months PASS and " + mo.windows_alert + " ALERT: April 2018, where error rose to " + f2(mo.real_data_windows.find((w) => w.window === "2018-04").error_ratio) + "× baseline alongside anomalous rainfall drift, and June 2018 on rainfall drift alone. Three synthetic scenarios — a +12 K temperature-sensor bias, a 35% demand increase, and a two-hour shift in the peak — all raise ALERT, the first on feature drift and the other two on error drift, which demonstrates the detector responds to both failure modes. The dashboard (`outputs/MONITORING_DASHBOARD.md` and Figure 6) reports PASS / ALERT per window with the reason." },
  { image: F3("17_monitoring_dashboard.png"), width: 600, caption: "Figure 6 — Monitoring across the held-out year. Left: error ratio against the same-month baseline. Right: worst-feature drift relative to its calibrated threshold." },

  { h2: "4.8 Responsible and sustainable AI (Task 7)" },
  { p: "The bias and fairness report is a separate document. In summary: the data covers one carriageway with a 22.8% coverage gap and predates 2020; the proxy label is circular and may locate risk in the wrong hours, since crash severity is highest in exactly the free-flowing night conditions it marks as safe; and error concentrates on peak hours (" + f1(byHour[worstHour] / byHour[bestHour]) + "× the overnight level), weekends (" + f0(100 * (sv.regression.mae_weekend / sv.regression.mae_weekday - 1)) + "% worse than weekdays) and winter, which means the tool serves worst the travellers with least flexibility. Governance recommendations before any real-world use include a model card with an explicit prohibition on safety framing, out-of-time validation as a release gate, human-owned promotion, calibrated monitoring with a defined response, upstream data rules at the sensor gateway, and a retraining sunset date. On sustainability, the LSTM costs " + f0(lstm.training_seconds / hgb.training_seconds) + "× the compute of gradient boosting and " + (lstmBetter ? "buys " + f0(lstmGap) + " vehicles of accuracy that no driver could perceive" : "does not beat it") + "; the gradient-boosting model is the recommended production choice." },

  { h1: "5. Things that went wrong, and what they taught" },
  { bullets: [
    "**The Power BI \"lowest weather\" answer is Squall at 420 vehicles/hour, from a single hour.** The literal answer to the brief's question is uninterpretable, so both it and the defensible answer (Fog, 2,654, from 617 hours) are reported, with the sample size on the visual.",
    "**A first congestion-over-time chart showed demand collapsing through 2014–15.** It was plotting months with a handful of hours as if they were complete. Masking months under 400 hours removed the collapse; the outage was the finding, not the traffic.",
    "**A first monitoring design alerted on every month.** Weather compared against an all-season reference always drifts. Same-calendar-month references and empirically calibrated thresholds turned a useless alarm into one that fired twice, both times for a reason.",
    "**The feature selector picked up `weather_severity_zscore` and `_minmax` as weather columns** because they start with `weather_`. Harmless to accuracy, but redundant and sloppy; the fix is in `common.py` and the before/after runs are both in MLflow.",
    "**Two identical LSTM runs gave MAE 201 and 233.** CPU reductions are not order-stable by default. Op determinism fixed it; the report never had to choose which number to believe.",
    "**LSTM sequences were first built in run-size order, not time order,** so the \"chronological\" split was not, and the network was quietly being tested on periods it had trained beside. The uncorrected run reported MAE 201.8; the corrected run reports " + f1(lstm.mae) + ". A sort by timestamp fixed it, and the prediction-against-time plot is what exposed it — the x-axis ran from 2013 to 2018 for what was supposed to be two weeks. Always plot predictions against time.",
  ]},

  { h1: "6. Recommendations for the Smart City Mobility Team" },
  { numbered: [
    "**Target the afternoon.** The 16:00 hour carries 21% more traffic than 07:00 and stays elevated twice as long. Ramp metering, variable messaging and employer staggered-hours agreements should be aimed at 14:00–18:00, and at Wednesday–Friday afternoons in particular.",
    "**Publish travel-timing advice from the gradient-boosting model, with hour-specific confidence.** It is accurate to ~7% of the mean, retrains in seconds, and the recommender already produces the sentence. State plainly that it is least accurate at peak and on weekends.",
    "**Do not deploy the risk classifier under any safety framing** until a real crash dataset replaces the proxy. The pipeline is ready for it; only the label needs to change.",
    "**Fix the feed at source.** De-duplicate hours, reject physically impossible readings, and label whole holidays at the gateway. Each is one line; each changed a conclusion here.",
    "**Retrain on post-2020 data before any operational use,** and set a sunset date. The monitoring design will tell the team when the model has drifted; only new data will tell it what to.",
  ]},

  { h1: "Appendix — Repository and reproducibility" },
  { p: "`run_all.sh` regenerates every output in this report from the raw CSV in about twelve minutes on a laptop CPU (`--quick` skips the LSTM and SHAP). Part 1: `part1_data_analytics/` (SQL files, runner, statistics script, Power BI script and guide, this report's companion). Part 2: `part2_python/` (pipeline, features, visualisations, mini-app, logs, figures with interpretations, methodology report). Part 3: `part3_machine_learning/` (`src/` for all training and analysis scripts, `api/` for the service and tests, `models/` for artefacts, `outputs/` for results JSON, the model registry and monitoring dashboard, `mlflow.db` and `mlruns/` for tracking, `reports/` for this document and the bias and fairness report). Logging writes to `part2_python/logs/pipeline.log` and `app.log`, and `part3_machine_learning/logs/part3.log` and `api.log`, at DEBUG in the file and INFO on the console. The root README lists the commands for each stage." },
];

buildDocument({
  title: "Final Capstone Report",
  subtitle: "Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility — methodology and findings across Parts 1, 2 and 3",
  meta: [["Student", "Sora"], ["Programme", "NUS/Emeritus Applied Machine Learning and Data Science"],
         ["Dataset", "Metro Interstate Traffic Volume — westbound I-94, Minneapolis–St Paul, 2012–2018"]],
  blocks, outFile: OUT, footerText: "Final Capstone Report",
}).then((f) => console.log("written", f)).catch((e) => { console.error(e); process.exit(1); });
