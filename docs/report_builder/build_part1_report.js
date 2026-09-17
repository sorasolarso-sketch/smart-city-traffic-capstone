const path = require("path");
const { buildDocument } = require("./report_lib");

const ROOT = path.resolve(__dirname, "../..");
const OUT = path.join(ROOT, "part1_data_analytics/report/Part1_Data_Analytics_Insights_Report.docx");

const blocks = [
  { h1: "1. Purpose and data" },
  { p: "This report summarises what the Smart City Mobility Analytics Team learned from 48,204 hourly records of westbound I-94 traffic near Minneapolis–St Paul (October 2012 to September 2018), and what those findings mean for how the corridor should be managed. SQLite was used for trend queries, Python for statistics and probability, and Power BI for the dashboard; every figure is reproducible from the repository scripts." },

  { h1: "2. Four data quality findings that change the answers" },
  { bullets: [
    "**7,629 rows (15.8%) repeat a timestamp already in the file.** The weather provider emits one record per condition observed, so an hour with rain and mist appears twice with the same traffic count. Summing by year without removing them inflates every annual total by 6.9–21.1%. All figures here use one row per hour.",
    "**No column contains a null, but three contain impossible values:** ten temperatures of 0 K, one rainfall of 9,831 mm in an hour (the world record is ~305 mm), and two mid-morning readings of zero vehicles. All were repaired by median imputation for the same calendar month.",
    "**The holiday flag marks only the midnight hour of each holiday,** not the whole day, and New Year's Day 2015 is absent entirely. Holiday analysis therefore selected by calendar date rather than by the flag.",
    "**Coverage is severely uneven across years:** 2012 has 24% of a year, 2015 41%, 2017 99.5%. The sensor was offline for most of 2014–2015, so raw yearly totals mostly measure sensor uptime.",
  ]},

  { h1: "3. Traffic volume trends, 2012–2017" },
  { p: "Because coverage differs so much, the comparison rests on average hourly volume and a seasonally adjusted index (each year's months against the same months pooled across all years), not on raw totals." },
  { table: {
    header: ["Year", "Hours recorded", "Total volume (de-duplicated)", "Avg vehicles / hour", "Seasonally adjusted index", "Direction"],
    align: ["left", "right", "right", "right", "right", "left"],
    widths: [900, 1500, 2200, 1700, 1900, 1438],
    rows: [
      ["2012", "2,103", "6,785,754", "3,226.7", "+19.0", "baseline"],
      ["2013", "7,294", "24,139,878", "3,309.6", "+34.1", "increase"],
      ["2014", "4,501", "14,718,915", "3,270.1", "+28.4", "decrease"],
      ["2015", "3,593", "11,706,145", "3,258.0", "−46.8", "decrease"],
      ["2016", "7,838", "25,032,183", "3,193.7", "−88.9", "decrease"],
      ["2017", "8,713", "29,420,221", "3,376.6", "+89.1", "increase"],
    ],
  }},
  { p: "**Observation 1 — the raw total is misleading by a factor of ten.** The de-duplicated total rose 256% from 2012 to 2013 and fell 39% from 2013 to 2014. Neither number describes demand: the first reflects a jump from 2,103 to 7,294 recorded hours, the second a drop to 4,501. On the average-per-hour measure, the same two changes are +2.6% and −1.2%." },
  { p: "**Observation 2 — demand was flat to slightly declining from 2013 to 2016, then rose sharply in 2017.** The seasonally adjusted index falls from +34 vehicles/hour above the corridor norm in 2013 to −89 in 2016, then jumps to +89 in 2017, a swing of 178 vehicles/hour and the largest single-year move in the series. 2017 is also the only year with near-complete coverage, so it is the most reliable point. The mobility team should treat 2017 as the demand baseline for planning, not the six-year average." },
  { p: "**Observation 3 — 2016 is the weakest year on every measure,** and the dip survives seasonal adjustment, so it is real rather than a coverage artefact; operations records should be checked for road works or recalibration." },

  { h1: "4. Temperature around holidays" },
  { table: {
    header: ["Holiday", "2015", "2016", "2017", "Year-on-year"],
    align: ["left", "right", "right", "right", "left"],
    widths: [2200, 1700, 1700, 1700, 2338],
    rows: [
      ["New Year's Day, avg °C", "no data (outage)", "−6.1 (18 h)", "−3.1", "+3.0 K warmer in 2017"],
      ["New Year's Day, avg traffic", "—", "1,832", "2,128", "+296 vehicles/h"],
      ["Labor Day, avg °C", "22.2", "21.9", "18.1", "−3.7 K colder in 2017"],
      ["Labor Day, avg traffic", "2,430", "2,152", "2,395", "−279 then +244"],
    ],
  }},
  { p: "New Year's Day 2017 was 3 K warmer than 2016 and carried 16% more traffic; Labor Day 2017 was 3.7 K colder and carried 11% more. Temperature moved in opposite directions while traffic rose in both cases, so **the year-on-year temperature changes do not appear relevant to traffic on these holidays.** The holiday itself is what matters: both run 30–40% below the ordinary-day average for their month (Labor Day 2017: 2,395 against 3,493 for other September days)." },

  { h1: "5. Descriptive statistics and correlation" },
  { p: "Across all 48,204 rows, hourly volume has a mean of **3,259.8**, median **3,380**, standard deviation **1,986.9**, variance **3,947,615** and range **0 to 7,280** (coefficient of variation 0.61, excess kurtosis −1.31). The kurtosis is the telling figure: the distribution is bimodal — a spike of near-empty overnight hours and a broad daytime mass around 4,000–6,000 with little between. Mean and median both fall in the valley between the modes, so **neither describes a typical hour**; an \"average traffic\" KPI card should always sit beside the hourly profile." },
  { p: "Temperature and traffic correlate at **Pearson r = +0.130** (r² = 0.017; Spearman ρ = 0.133; p < 10⁻¹⁸⁰): positive in direction, weak in strength, explaining under 2% of variance. Correlation is not causation here: warm hours and busy hours are both daytime summer hours, and the clock drives both. Within any single time-of-day band the relationship vanishes. The significance reflects 48,000 observations, not practical importance." },

  { h1: "6. Probability of congestion" },
  { p: "Congestion is defined as more than 5,500 vehicles per hour; high temperature as above 292 K (18.9 °C)." },
  { table: {
    header: ["Quantity", "Value", "Quantity", "Value"],
    align: ["left", "right", "left", "right"],
    widths: [2900, 1500, 3300, 1938],
    rows: [
      ["P(Congestion)", "0.1473", "P(Clear | Congestion)", "0.2483"],
      ["P(Clear weather)", "0.2778", "P(High temp | Congestion)", "0.2630"],
      ["P(Congestion ∩ Clear)", "0.0366", "P(Congestion | Clear)", "0.1317"],
      ["P(Congestion) × P(Clear)", "0.0409", "P(Congestion | Cloudy)", "0.1709"],
    ],
  }},
  { p: "**Independence.** Observed P(Congestion ∩ Clear) = 0.0366 against 0.0409 expected under independence, a shortfall of 10.6%. A chi-square test rejects independence (χ² = 35.9, p = 2 × 10⁻⁹), but the effect size φ = 0.027 is negligible. The two events are not strictly independent, but they are very nearly so." },
  { p: "**Odds ratio.** Odds of congestion in clear weather (0.152) against cloudy weather (0.206) give an **odds ratio of 0.735** (95% CI 0.689–0.785): congestion appears 26% less likely under clear skies. The naive reading is backwards. Clear skies are disproportionately a night-time condition here (at 23:00 clear hours outnumber cloudy 737 to 512) and cloud an afternoon one (at 14:00, 843 cloudy to 440 clear). **Holding hour of day constant, the clear-versus-cloudy difference in congestion rate collapses to +0.03 percentage points.** The odds ratio measures when each kind of weather happens, not how drivers respond to it." },
  { p: "**What the probability analysis suggests.** Weather is not a useful predictor of congestion on this corridor once the time of day is known. The Task 4 weather chart shows Clouds highest (3,617) and Fog lowest (2,654), and a stakeholder will reasonably conclude that fog empties the road. It does not; fog forms at 04:00. Every weather visual should carry the hour-of-day control beside it." },

  { h1: "7. Implications for the mobility team" },
  { numbered: [
    "**Re-aim interventions at the afternoon.** In 2017 the 16:00 hour averaged 5,834 vehicles against 4,820 at 07:00, and the afternoon stays above 5,000 for four hours where the morning does so for two. Ramp metering, messaging and staggered-hours agreements should target 14:00–18:00 first.",
    "**Stop treating weather as a congestion trigger.** The data does not support weather-conditional management for volume; it may still be justified for safety, which this dataset cannot speak to.",
    "**Fix the feed before building on it.** De-duplicate hours at ingestion, reject impossible readings at the gateway, label every hour of a holiday, and annotate coverage gaps on every time-series visual. Treat 2017 as the planning baseline year.",
  ]},
];

buildDocument({
  title: "Data Analytics Insights Report",
  subtitle: "Part 1 — Understanding traffic patterns on westbound I-94",
  meta: [["Student", "Rapipong Sornsakda"], ["Programme", "NUS/Emeritus Applied Machine Learning and Data Science"],
         ["Capstone", "Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility"]],
  blocks, outFile: OUT, footerText: "Part 1 — Data Analytics Insights Report",
}).then((f) => console.log("written", f)).catch((e) => { console.error(e); process.exit(1); });
