#!/usr/bin/env bash
# =============================================================================
# Rebuild every output in this repository from the raw CSV.
#
#   bash run_all.sh            # full run (~12 minutes on a laptop CPU)
#   bash run_all.sh --quick    # skips the LSTM/SHAP stage (~2 minutes)
#
# Each stage is a separate script and can also be run on its own; the
# per-part READMEs list the individual commands.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export MLFLOW_DISABLE_AGENT_HINT=1
export TF_CPP_MIN_LOG_LEVEL=3
PY=${PYTHON:-python3}
QUICK=${1:-}

step () { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

step "Part 1 — SQL analysis (SQLite)"
$PY part1_data_analytics/run_sql_analysis.py

step "Part 1 — statistics and probability"
$PY part1_data_analytics/statistics_probability.py

step "Part 1 — Power BI preparation and dashboard answers"
$PY part1_data_analytics/powerbi_prep.py

step "Part 2 — data pipeline"
$PY part2_python/pipeline.py

step "Part 2 — feature engineering"
$PY part2_python/feature_engineering.py

step "Part 2 — visualisations"
$PY part2_python/visualizations.py

step "Part 2 — mini application (sample commands)"
$PY part2_python/mini_app/traffic_app.py summary
$PY part2_python/mini_app/traffic_app.py query 2017-08-31 --hour 17
$PY part2_python/mini_app/traffic_app.py peak --top 3 --day-type weekday
$PY part2_python/mini_app/traffic_app.py compare
$PY part2_python/mini_app/traffic_app.py recommend --day-type weekday --earliest 7 --latest 19
$PY part2_python/mini_app/traffic_app.py weather --condition Snow

step "Part 3 — supervised models (classification + regression, MLflow tracked)"
$PY part3_machine_learning/src/supervised.py

step "Part 3 — unsupervised (K-means + association rules)"
$PY part3_machine_learning/src/unsupervised.py

if [[ "$QUICK" == "--quick" ]]; then
  step "Part 3 — LSTM + SHAP skipped (--quick)"
else
  step "Part 3 — LSTM + SHAP explainability"
  $PY part3_machine_learning/src/deep_learning.py --epochs 40
fi

step "Part 3 — monitoring and alerting simulation"
$PY part3_machine_learning/src/monitoring.py

step "Part 3 — model registry and experiment summary"
$PY part3_machine_learning/src/model_registry.py

step "Part 3 — recommendation system (sample)"
$PY part3_machine_learning/src/recommender.py --day-type weekday --earliest 7 --latest 19 --duration 2
$PY part3_machine_learning/src/recommender.py --date 2018-07-04 --weather Rain --earliest 6 --latest 20

step "Part 3 — API smoke tests"
$PY part3_machine_learning/api/test_api.py

step "Done. Logs: part2_python/logs/pipeline.log, part2_python/logs/app.log, part3_machine_learning/logs/part3.log"
