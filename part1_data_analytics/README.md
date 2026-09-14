# Part 1 — Data Analytics: Understanding Traffic Patterns

SQL, statistics, probability and Power BI on the Metro Interstate Traffic
Volume dataset. Every figure quoted in the insights report and the dashboard
guide is produced by the scripts here and written to `outputs/`.

## Contents

| Path | Task | What it is |
| --- | --- | --- |
| `sql/01_load_and_verify.sql` | 1.1 | Typed schema, indexes, ten verification queries (row count, nulls, ranges, duplicates, coverage by year) |
| `sql/02_annual_traffic_trends.sql` | 1.2 | Yearly totals and averages 2012–2017 (de-duplicated), size of the double-counting error, year-on-year change on both measures, seasonally adjusted index, monthly coverage matrix |
| `sql/03_holiday_temperature.sql` | 1.3 | New Year's Day and Labor Day 2015–2017 by calendar date, with the flag-only view shown for contrast, YoY change and ordinary-day baselines |
| `run_sql_analysis.py` | 1 | Rebuilds `outputs/traffic.db` from the raw CSV and executes every statement above, writing `outputs/sql_results.md` |
| `statistics_probability.py` | 2, 3 | Mean/median/SD/variance/range, Pearson and Spearman correlation, P(Congestion), P(Clear), joint and conditional probabilities, independence test with χ² and φ, odds ratio with 95% CI, and an hour-of-day-controlled comparison |
| `powerbi_prep.py` | 4 | Applies the Power Query transformations in Python, writes the prepared CSV and every number the dashboard must display to `outputs/powerbi_answers.json` |
| `powerbi/power_query_script.m` | 4.1 | Sixteen named Power Query steps: typed load, standardisation, severity ranking, de-duplication, impossible-value repair, Hour, TempCelsius, TrafficCategory |
| `powerbi/traffic_powerbi_ready.csv` | 4 | The prepared table (40,575 × 19) for direct import |
| `powerbi/DASHBOARD_BUILD_GUIDE.md` | 4.2, 4.3 | Visual-by-visual build instructions, DAX for the KPI cards, slicer setup, and the answers to every dashboard question, verified against the data |
| `report/Part1_Data_Analytics_Insights_Report.docx` | deliverable | Two-page insights report for the mobility team |

## Run

From the repository root:

```bash
python part1_data_analytics/run_sql_analysis.py
python part1_data_analytics/statistics_probability.py
python part1_data_analytics/powerbi_prep.py
```

To use the SQL files directly in the `sqlite3` shell instead:

```bash
sqlite3 part1_data_analytics/outputs/traffic.db < part1_data_analytics/sql/02_annual_traffic_trends.sql
```

## Power BI

Power BI Desktop is Windows-only and cannot be run in this project's
environment, so the `.pbix` is not committed. To build the dashboard:

1. Open Power BI Desktop → Transform data → Advanced Editor.
2. Paste `powerbi/power_query_script.m`, edit `SourcePath` to your copy of the
   raw CSV, Close & Apply.
3. Follow `powerbi/DASHBOARD_BUILD_GUIDE.md` for the four visuals, three KPI
   cards and three slicers. Every number the guide quotes is in
   `outputs/powerbi_answers.json`, so the finished dashboard can be checked
   against it.

Alternatively, import `powerbi/traffic_powerbi_ready.csv` directly — it is
the output of the same transformation chain.

## Key results (all in `outputs/`)

- Yearly totals must be de-duplicated (7,629 repeated timestamps inflate them
  by 7–21%) and read alongside hours recorded (2,103 in 2012 vs 8,713 in
  2017). On a seasonally adjusted basis demand dips to its low in 2016 and
  rises sharply in 2017.
- Holiday temperature changes move in opposite directions on the two holidays
  while traffic rises on both; the holiday effect (30–40% below ordinary
  days) is what matters.
- Volume: mean 3,259.8, median 3,380, SD 1,986.9, range 0–7,280; bimodal
  (excess kurtosis −1.31).
- Temperature vs volume: r = +0.130 (weak, positive); a time-of-day confound.
- P(Congestion) = 0.147, P(Clear) = 0.278, P(both) = 0.037 vs 0.041 under
  independence (χ² rejects, φ = 0.027 negligible). Odds ratio clear vs cloudy
  = 0.735 — which collapses to a +0.03 pp difference once hour of day is held
  constant.
- Dashboard: 2017 peak at 16:00 (5,834 veh/h) vs 07:00 (4,820); highest
  weather average Clouds (3,617), lowest interpretable Fog (2,654), gap 963;
  KPIs 40,575 hours, 3,290.7 average volume, 8.2 °C average temperature.
