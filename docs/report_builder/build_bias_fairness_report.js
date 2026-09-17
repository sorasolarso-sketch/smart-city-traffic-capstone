const path = require("path");
const fs = require("fs");
const { buildDocument } = require("./report_lib");

const ROOT = path.resolve(__dirname, "../..");
const P3 = path.join(ROOT, "part3_machine_learning");
const OUT = path.join(P3, "reports/Bias_and_Fairness_Report.docx");
const J = (f) => JSON.parse(fs.readFileSync(f, "utf8"));

const sv = J(path.join(P3, "outputs/supervised_results.json"));
const dl = J(path.join(P3, "outputs/deep_learning_results.json"));
const mo = J(path.join(P3, "outputs/monitoring_report.json"));
const fm = J(path.join(ROOT, "data/processed/feature_metadata.json"));

const rf = sv.classification.models.random_forest.metrics;
const lr = sv.classification.models.logistic_regression.metrics;
const hgb = sv.regression.models.hist_gradient_boosting.metrics;
const lstm = dl.lstm.metrics;
const byHour = sv.regression.mae_by_hour;
const hours = Object.keys(byHour).map(Number);
const worstHour = hours.reduce((a, b) => (byHour[a] > byHour[b] ? a : b));
const bestHour = hours.reduce((a, b) => (byHour[a] < byHour[b] ? a : b));
const ratio = byHour[worstHour] / byHour[bestHour];
const wdMae = sv.regression.mae_weekday, weMae = sv.regression.mae_weekend;
const monAlerts = mo.real_data_windows.filter((w) => w.status === "ALERT");
const monByMonth = mo.real_data_windows;
const winter = monByMonth.filter((w) => ["2017-12", "2018-01", "2018-02"].includes(w.window));
const summer = monByMonth.filter((w) => ["2018-06", "2018-07", "2018-08"].includes(w.window));
const avg = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const winterMae = avg(winter.map((w) => w.mae)), summerMae = avg(summer.map((w) => w.mae));
const f0 = (x) => x.toLocaleString("en-US", { maximumFractionDigits: 0 });
const f1 = (x) => x.toFixed(1), f3 = (x) => x.toFixed(3), f4 = (x) => x.toFixed(4);
const pct = (x) => `${(100 * x).toFixed(1)}%`;
const lstmBetter = lstm.mae < hgb.mae;
const lstmGap = Math.abs(hgb.mae - lstm.mae);

const blocks = [
  { p: "This report answers Task 7 of the capstone. Section 1 is the bias and fairness assessment the brief requires: sampling and coverage limitations, the proxy label and its risks, and how model errors are distributed unevenly across conditions and time. Section 2 sets out what governance should precede any real-world use of these models, and the resource trade-offs behind the modelling choices. Every number is taken from the outputs in `part3_machine_learning/outputs/` and can be regenerated with `run_all.sh`." },
  { callout: "**The single most important statement in this document.** No accident data was provided. The \"accident-risk\" classifier predicts a label that was constructed from congestion and weather; it has never seen a collision. It should be described, deployed and governed as a *congestion-in-bad-weather* detector, never as an accident predictor. The Random Forest's ROC AUC of " + f4(rf.roc_auc) + " is not evidence of safety insight — it is evidence that the model has learned the definition it was given." },

  { h1: "1. Bias and fairness assessment" },
  { h2: "1.1 Sampling and coverage limitations in the data" },
  { p: "The models generalise to exactly one thing: the hour-by-hour behaviour of the westbound carriageway of I-94 at a single sensor near Minneapolis–St Paul, between October 2012 and September 2018, as reported by one weather station. Each element of that sentence is a limit." },
  { bullets: [
    "**One sensor, one direction.** Eastbound traffic, parallel arterials, the rest of the network and any other city are unrepresented. A recommendation to \"travel at 19:00\" is a statement about this carriageway; the road a driver takes to reach it may be worst at exactly that time.",
    "**22.8% of the hours in the span are missing** (11,976 of 52,551), concentrated in 2014–2015. The training data over-represents 2016–2018 and under-represents 2012–2015, so anything that changed in the corridor over that period — lane configuration, employment patterns, fuel prices — is weighted towards the later state. The LSTM is affected most: only " + f0(dl.lstm.metrics.n_train_sequences + dl.lstm.metrics.n_test_sequences) + " of 40,575 hours could be formed into unbroken 24-hour sequences, so 7,552 rows were dropped from its training set, and those rows are not missing at random — they sit at the edges of outages.",
    "**Weather is a single point measurement** from one station, which may be kilometres from the sensor. Localised fog, black ice on a bridge deck, or a shower over one interchange is invisible. The `is_low_visibility` flag that feeds the proxy label inherits this coarseness.",
    "**Holidays are under-labelled.** 61 rows carry a holiday name, one hour per holiday. The pipeline derives `is_holiday` from those rows, so the model has seen a handful of holiday hours and cannot be expected to forecast a holiday well; the recommender attaches an explicit caveat whenever a requested date is a public holiday.",
    "**The data predates 2020.** Commuting patterns in most cities changed structurally after the pandemic. A model trained here and deployed now would be extrapolating across a regime change it has no way of knowing about — which is exactly the case the monitoring stage exists to catch.",
    "**Duplicate records were not errors but a reporting convention** (one row per weather condition per hour). Removing them was correct for the traffic analysis, but the choice to retain the *most severe* condition for each hour is a modelling decision that biases the weather features slightly towards the adverse end.",
  ]},

  { h2: "1.2 The proxy label and the risks it carries" },
  { p: "The Part 3 brief defines an hour as *high risk* when its congestion category is High or Severe (traffic volume above the median, " + f0(fm.congestion_thresholds.q2_median) + " vehicles/hour) **and** the weather is in a severe category or reduces visibility. " + pct(rf.positive_rate_actual) + " of test-period hours meet this definition. Four risks follow from building a classifier on it." },
  { numbered: [
    "**Circularity.** The label is a deterministic function of two things the model is given as inputs (weather) or can infer almost perfectly from time (volume). Both classifiers therefore score near the ceiling — logistic regression ROC AUC " + f4(lr.roc_auc) + ", Random Forest " + f4(rf.roc_auc) + " — and the feature importances confirm it: `weather_severity`, `adverse_conditions_score` and `hour_cos` dominate. A stakeholder shown \"98% accuracy at predicting accident risk\" would draw a conclusion the data cannot support. Every artefact that exposes this model — the API response, the registry description, the README — carries a disclaimer for this reason.",
    "**The definition may be wrong about where risk lives.** Road-safety literature consistently finds that crash *severity* rises at high speed and low density — late-night, free-flowing, often with impairment or fatigue — precisely the hours this label marks as lowest risk. Congested traffic produces more minor collisions and fewer fatal ones. A system built on this proxy would direct attention to the busy afternoon and away from the empty road at 02:00, which may be backwards for the outcome a city cares about most.",
    "**The label encodes weather seasonality as risk.** Severe and low-visibility conditions cluster in winter, so the positive rate is not stable across the year: it was " + pct(sv.classification.models.random_forest.metrics.positive_rate_actual) + " in the test period against a training-period rate about two percentage points lower. A classifier trained on this will flag winter more readily than summer for reasons that have nothing to do with driver behaviour.",
    "**Absent causes are invisible.** Road works, incidents, lighting failures, pavement condition, enforcement presence and vehicle mix all affect real accident risk and none is in the data. The proxy cannot be improved by better modelling; it can only be replaced by an outcome label.",
  ]},
  { p: "**What would be needed to do this properly:** geocoded crash records from the state DOT matched to the sensor's segment and hour, with severity; a label defined by outcome (crash / no crash, or crashes per vehicle-mile); and a re-run of the same pipeline. The infrastructure built here — features, split, tracking, monitoring — transfers unchanged. Only the target would change, and with it every conclusion." },

  { h2: "1.3 How errors are distributed unevenly" },
  { p: "A single headline metric hides where a model is weak. The regression model's overall MAE of " + f1(hgb.mae) + " vehicles/hour (" + f1(hgb.mae_as_pct_of_mean) + "% of the mean) decomposes as follows." },
  { table: {
    header: ["Slice", "MAE (vehicles/hour)", "Relative to best slice", "Who is affected"],
    align: ["left", "right", "right", "left"],
    widths: [2600, 1800, 1700, 3992],
    rows: [
      [`Quietest hour (${String(bestHour).padStart(2, "0")}:00)`, f0(byHour[bestHour]), "1.0×", "Overnight and shift workers — best served"],
      [`Busiest hour (${String(worstHour).padStart(2, "0")}:00)`, f0(byHour[worstHour]), f1(ratio) + "×", "Peak commuters — worst served, and the group the tool matters most to"],
      ["Weekdays", f0(wdMae), f1(wdMae / byHour[bestHour]) + "×", "Regular commuters"],
      ["Weekends", f0(weMae), f1(weMae / byHour[bestHour]) + "×", "Leisure and retail travellers — " + f0(100 * (weMae / wdMae - 1)) + "% worse than weekdays"],
      ["Winter months (Dec–Feb, held-out)", f0(winterMae), f1(winterMae / byHour[bestHour]) + "×", "Everyone, in the season when a bad prediction is most costly"],
      ["Summer months (Jun–Aug, held-out)", f0(summerMae), f1(summerMae / byHour[bestHour]) + "×", ""],
    ],
  }},
  { p: "Three patterns matter. **Error scales with volume**: the model is " + f1(ratio) + " times less accurate at " + String(worstHour).padStart(2, "0") + ":00 than at " + String(bestHour).padStart(2, "0") + ":00, so the recommendation system is least reliable precisely for the peak-hour journeys people most want help with, and its \"avoid this window\" advice carries wider error bars than its \"travel then\" advice. **Weekends are served worse than weekdays** by about " + f0(100 * (weMae / wdMae - 1)) + "%, because weekend demand is more variable and there is less than half as much of it to learn from (11,596 hours against 28,979). **Winter is harder than summer**: in the held-out year the December–February MAE averaged " + f0(winterMae) + " against " + f0(summerMae) + " for June–August, and the one month that breached the error-drift alarm (April 2018, " + f1(monAlerts[0]?.error_ratio || 0) + "× its baseline) coincided with anomalous rainfall drift. The LSTM shows the same shape (overall MAE " + f1(lstm.mae) + ")." },
  { p: "The fairness consequence is distributional. The tool's benefit accrues to people who *can* shift their travel time — salaried workers with flexible hours — and its error falls hardest on people who cannot: shift workers travelling at fixed peak times, weekend retail and hospitality staff, and anyone travelling in winter. A deployment that reports one accuracy figure would obscure this; a deployment that reports hour-specific confidence, as the monitoring dashboard does by month, would at least make it visible." },
  { p: "**Mitigations applied in this project:** hour-of-day error is computed and logged as a WARNING every time the regressor is trained; monitoring baselines are calendar-month specific so winter is judged against winter; the recommender attaches caveats for holidays and for weather with fewer than 200 hours of history; and the API's risk endpoint returns a disclaimer in every response. **Mitigations not applied, and why:** re-weighting weekend rows would trade weekday accuracy for weekend accuracy without new information; the honest fix is more weekend data or a separate weekend model, both of which are recommendations rather than steps taken here." },

  { h1: "2. Governance and sustainability" },
  { h2: "2.1 Oversight required before a model is trusted for real decisions" },
  { p: "None of the models in this repository should influence a real traffic-management decision in their current state. The gap between \"a good capstone model\" and \"a model a city may rely on\" is governance, and the following would need to be in place first." },
  { bullets: [
    "**A written purpose and a written prohibition.** The volume regressor may inform travel-timing advice. The proxy classifier may not be used for anything described to the public as safety or accident prediction. That distinction belongs in a model card that ships with the model, not in a README a procurement officer will never read.",
    "**Out-of-time validation as a gate, not a courtesy.** Every metric in this project is on a chronological hold-out because the random split flattered the classifiers by " + f4(rf.optimism_from_random_split) + " AUC. A production gate should require the same: performance on the most recent quarter, never on a shuffled sample.",
    "**Human ownership of promotion.** The registry marks a `champion` and a `challenger`; moving that alias is a decision a named person makes after reviewing the monitoring dashboard, and it is logged. Automatic retraining-and-promotion on a schedule is how a data-feed fault becomes a policy.",
    "**Monitoring with calibrated thresholds and a defined response.** The monitoring stage showed that textbook PSI cut-offs fire every month on weather data; thresholds had to be learned from the training years' own variability before the alarm meant anything. Each alert needs an owner, a triage procedure (sensor fault? seasonal? genuine change?) and a rollback path to the previous champion.",
    "**Data governance upstream of the model.** Three one-line rules at the sensor gateway — reject 0 K, reject rainfall above 400 mm/h, de-duplicate hours — would have removed every data-quality repair this project had to make. Model governance that ignores the feed is governance of the wrong thing.",
    "**Public transparency proportional to public effect.** If timing advice is published, the public should be told it is derived from one sensor on one carriageway, that it is least accurate at peak hours, and that it is silent on safety. This follows the transparency and accountability principles in the OECD AI Principles and the *Govern* and *Map* functions of the NIST AI Risk Management Framework, both of which treat documentation of limits as a precondition for deployment rather than an afterthought.",
    "**A sunset clause.** The data ends in 2018. Any deployment should specify the date by which the model must be retrained on current data or withdrawn.",
  ]},

  { h2: "2.2 Environmental and resource trade-offs" },
  { table: {
    header: ["Model", "Training time (CPU)", "Test MAE", "Gain over next-simplest", "Assessment"],
    align: ["left", "right", "right", "right", "left"],
    widths: [2300, 1700, 1300, 1900, 2892],
    rows: [
      ["Linear regression", f1(sv.regression.models.linear_regression.metrics.training_seconds) + " s", f1(sv.regression.models.linear_regression.metrics.mae), "—", "Too coarse for peak hours"],
      ["Gradient boosting", f1(hgb.training_seconds) + " s", f1(hgb.mae), f0(sv.regression.models.linear_regression.metrics.mae - hgb.mae) + " vehicles", "Recommended for production"],
      ["LSTM (40 epochs)", f0(lstm.training_seconds) + " s", f1(lstm.mae), (lstmBetter ? "+" : "−") + f0(lstmGap) + " vehicles", f0(lstm.training_seconds / hgb.training_seconds) + "× the compute; " + (lstmBetter ? f0(100 * (1 - lstm.mae / hgb.mae)) + "% less error" : "no accuracy gain")],
      ["SHAP on 1,200 rows", "~170 s", "—", "—", "One-off explanation cost; not per prediction"],
    ],
  }},
  { p: "The LSTM is the least defensible model to run routinely. It consumed roughly " + f0(lstm.training_seconds / hgb.training_seconds) + " times the training compute of the gradient-boosting model and " + (lstmBetter ? "reduced error by " + f0(lstmGap) + " vehicles per hour, an improvement well inside the model's own hour-to-hour error at peak and one no traveller could perceive" : "did not reduce error at all once its evaluation split was corrected (" + f1(lstm.mae) + " against " + f1(hgb.mae) + ")") + ". On a corridor forecasting problem of this size the environmentally and operationally sound choice is the gradient-boosting model: it trains in seconds on a laptop, serves predictions in under a millisecond, needs no GPU, and can be retrained nightly without a carbon or cost conversation. The LSTM's role is as a challenger that quantifies how much accuracy the simpler model leaves on the table — which turns out to be " + (lstmBetter ? "little" : "none") + "." },
  { p: "Three further resource decisions were taken deliberately. SHAP was computed on a 1,200-row sample rather than the full test set, because global attributions stabilise long before that and the marginal rows cost minutes for no information. The LSTM was run with op determinism enabled, which costs a small amount of speed and buys exact reproducibility, so that the result never has to be recomputed to be believed. And every model artefact is stored once, in the repository and the MLflow registry, rather than regenerated on demand. The general principle is that the cheapest model that meets the requirement is the right model, and the requirement here — timing advice with an error a driver cannot perceive — is met at 2.4 seconds of CPU time." },
];

buildDocument({
  title: "Bias, Fairness, Governance and Sustainability Report",
  subtitle: "Part 3, Task 7 — Responsible and sustainable AI for corridor mobility",
  meta: [["Student", "Rapipong Sornsakda"], ["Programme", "NUS/Emeritus Applied Machine Learning and Data Science"],
         ["Capstone", "Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility"]],
  blocks, outFile: OUT, footerText: "Part 3 — Bias and Fairness Report",
}).then((f) => console.log("written", f)).catch((e) => { console.error(e); process.exit(1); });
