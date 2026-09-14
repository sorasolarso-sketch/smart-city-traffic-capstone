# Part 1, Task 4 — Power BI Traffic Intelligence Dashboard

This guide builds the dashboard required by Task 4 and records the answers
to every question the task asks. All figures were computed from the data by
`part1_data_analytics/powerbi_prep.py` and written to
`part1_data_analytics/outputs/powerbi_answers.json`, so the numbers quoted
here can be checked against the source rather than taken on trust.

Two files support the build:

- `power_query_script.m` — the full Power Query transformation chain, ready
  to paste into the Advanced Editor.
- `traffic_powerbi_ready.csv` — the prepared table (40,575 rows × 19
  columns) if you would rather load the finished model directly.

---

## 4.1 Data quality and preparation

### Rows and columns

The raw file contains **48,204 data rows and 9 columns**. After preparation
the model holds **40,575 rows and 19 columns**.

### Null and missing values

**No column contains a true null.** This is the finding that matters most,
because it is misleading: the dataset encodes absence with sentinel values
rather than with blanks, and three of them will corrupt the analysis if they
are accepted at face value.

| Column | Sentinel | Rows | Why it matters |
| --- | --- | ---: | --- |
| `holiday` | the literal string `"None"` | 48,143 | Only 61 rows name a holiday, and the name is attached to the 00:00 hour alone rather than to the whole day |
| `temp` | `0` Kelvin | 10 | Absolute zero; the true value is unrecoverable, so it is imputed |
| `rain_1h` | `9831.3` mm | 1 | The world record for hourly rainfall is roughly 305 mm |
| `traffic_volume` | `0` | 2 | Plausible at 03:00, but both occur mid-morning and indicate sensor dropout |

A second data-quality problem is not visible as a missing value at all:
**7,629 rows repeat a timestamp already present on another row**. These are
not duplicates in the ordinary sense. The weather provider emits one record
per condition observed, so an hour with rain and mist appears twice, each
copy carrying an identical traffic count. Summing `traffic_volume` without
removing them inflates annual totals by between 6.9% and 21.1%.

### Data types

Power Query's automatic type detection is unreliable on this file. `holiday`
is inferred as a logical column under some locales because the modal value
is the word "None", and `date_time` is inferred as text when the locale
expects a different date order. Step 3 of the M script therefore sets all
nine types explicitly rather than accepting the guess.

### Required derived columns

| Column | Rule | M script step |
| --- | --- | --- |
| `Hour` | `Time.Hour(DateTime.Time([date_time]))` | 10 |
| `TempCelsius` | `[temp] - 273.15`, rounded to 2 dp | 12 |
| `TrafficCategory` | `< 4500` → Low; `4500–5500` → Medium; `> 5500` → High | 13 |

Resulting distribution: **Low 26,060 hours (64.2%)**, **Medium 8,408
(20.7%)**, **High 6,107 (15.1%)**.

> Note: this fixed-threshold `TrafficCategory` is deliberately different
> from the quartile-based `congestion_category` used in Part 3. The Part 1
> brief specifies cut-points; the Part 3 brief derives them from the data.
> Both exist in the repository and are never used interchangeably.

---

## 4.2 Dashboard analysis

### A. Daily traffic trends, 2015–2017

Build a line chart: `Date` on the axis, `Average of traffic_volume` as the
value, `Year` as the legend, filtered to 2015–2017.

| Year | Days with data | Mean of daily averages | Lowest day | Highest day |
| --- | ---: | ---: | --- | --- |
| 2015 | 195 | 3,263.7 | 2015-12-25 (1,573.6) | 2015-06-19 (4,681.0) |
| 2016 | 366 | 3,199.3 | 2016-07-23 (277.3) | 2016-04-27 (4,057.4) |
| 2017 | 365 | 3,377.2 | 2017-12-25 (1,889.8) | 2017-08-31 (4,055.5) |

**Interpretation.** Christmas Day is the quietest day of the year in both
years where December is covered — the single most reliable calendar signal
in the dataset. 2015 has only 195 days of data, so its line is visibly
broken and its annual mean is not comparable to the others without
adjustment. The 2016-07-23 figure of 277 vehicles per hour is a partial-day
sensor outage rather than a real collapse in demand, and should be labelled
as such on the dashboard rather than left to look like a genuine minimum.

### B. Hourly traffic patterns, 2017

Build a column chart: `Hour` on the axis, `Average of traffic_volume` as the
value, filtered to `Year = 2017`.

| Hour | Avg | Hour | Avg | Hour | Avg | Hour | Avg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 920 | 06 | 4,222 | 12 | 4,807 | 18 | 4,416 |
| 01 | 554 | 07 | 4,820 | 13 | 4,811 | 19 | 3,421 |
| 02 | 409 | 08 | 4,684 | 14 | 5,001 | 20 | 2,975 |
| 03 | 386 | 09 | 4,476 | 15 | 5,333 | 21 | 2,797 |
| 04 | 728 | 10 | 4,278 | 16 | **5,834** | 22 | 2,349 |
| 05 | 2,162 | 11 | 4,575 | 17 | 5,484 | 23 | 1,577 |

**Interpretation.** The corridor has two peaks, and they are not equal. The
evening peak at **16:00 (5,834 vehicles/hour)** is 21% higher than the
morning peak at **07:00 (4,820)**, and the evening peak is also broader,
holding above 5,000 from 14:00 to 17:00 while the morning sits above 4,600
for only two hours. The quietest hour is **03:00 (386)**, roughly one
fifteenth of the evening peak. Midday never returns to overnight levels —
the 10:00–14:00 trough still runs at 4,300–4,800, so this is a corridor with
sustained all-day demand rather than two isolated commuter spikes.

### C. Weather impact

Build a bar chart: `weather_main` on the axis, `Average of traffic_volume`
as the value, sorted descending.

| Weather | Hours | Avg traffic |
| --- | ---: | ---: |
| Clouds | 15,123 | 3,616.7 |
| Haze | 839 | 3,540.2 |
| Rain | 4,513 | 3,398.0 |
| Drizzle | 370 | 3,152.8 |
| Clear | 13,364 | 3,055.1 |
| Snow | 2,786 | 3,011.6 |
| Thunderstorm | 1,010 | 3,009.8 |
| Mist | 1,936 | 2,792.9 |
| Smoke | 16 | 2,772.4 |
| Fog | 617 | 2,653.8 |
| Squall | 1 | 420.0 |

**Which weather condition has the highest average traffic?**
**Clouds, at 3,616.7 vehicles per hour** across 15,123 hours.

**Which has the lowest?**
Taken literally, **Squall at 420.0** — but that category contains a **single
hour**, so the figure is an individual observation, not an average, and
cannot support any conclusion. Restricting to categories with at least 100
hours of data, the lowest is **Fog at 2,653.8** across 617 hours.

**What is the difference between the highest and lowest average traffic?**

- Literal answer, including single-observation categories: **3,196.7
  vehicles per hour** (Clouds 3,616.7 − Squall 420.0).
- Defensible answer, categories with ≥ 100 hours: **962.9 vehicles per
  hour** (Clouds 3,616.7 − Fog 2,653.8).

The second figure is the one to quote in the report. Put a count-of-hours
tooltip on this visual so a reader can see immediately that the Squall bar
rests on one observation.

**Interpretation, and a warning about reading this chart.** The ordering
here is close to meaningless as a causal claim. Traffic is not high *because*
it is cloudy. Cloud cover is the most common daytime condition in
Minneapolis, and clear skies are disproportionately common overnight, so
this chart is substantially measuring *when each condition tends to occur*
rather than how drivers respond to it. The probability analysis in Task 3
demonstrates this directly: the raw odds ratio suggests congestion is 26%
less likely in clear weather, but once hour of day is held constant the
average difference between clear and cloudy hours collapses to **+0.03
percentage points**. Fog sits at the bottom of this chart for the same
reason — fog forms in the small hours, when the road is empty.

> A note on the hour counts: the Mist category falls from 5,950 raw rows to
> 1,936 hours after de-duplication, because mist is usually reported
> alongside a more severe condition and the de-duplication rule keeps the
> more severe label. The prepared counts describe *hours whose most severe
> condition was X*, which is the correct basis for an average.

### D. Temperature and traffic

Build a scatter chart: `TempCelsius` on the X axis, `traffic_volume` on the
Y axis, with `Hour` on the legend or as a play axis.

**Is there a visible relationship?** Only a weak one. Pearson r = **+0.139**
(r² = 0.019), so temperature explains under **2%** of the variance in
traffic volume. The direction is positive — warmer hours carry slightly more
traffic — but the scatter is dominated by a far stronger structure: two
dense horizontal bands, one near zero and one around 4,500–6,000, which are
night and day rather than cold and warm.

**Which temperature range is associated with higher traffic?** Average
traffic rises fairly steadily with temperature and peaks in the **30–35 °C**
band. Below −20 °C the average falls away sharply. The honest reading is
that this reflects season and daylight hours, not a thermal response: the
hottest hours are summer afternoons, which is also when the corridor is
busiest for reasons that have nothing to do with temperature.

**Notable outliers.**

- **102 hours** sit more than four standard deviations from the mean for
  their own hour of day. These are the genuine anomalies, and they cluster
  around holidays and severe-weather events.
- **2 hours record exactly zero vehicles**, both mid-morning, when the
  corridor should be carrying over 4,000. These are sensor dropouts.
- The 9,831.3 mm rainfall reading — repaired in preparation — would
  otherwise compress the entire X axis of any rainfall visual into a single
  pixel.

---

## 4.3 KPI cards and filters

### KPI cards

| Card | Measure | Value |
| --- | --- | ---: |
| Total hours analysed | `COUNTROWS(traffic)` | **40,575** |
| Average traffic volume | `AVERAGE(traffic[traffic_volume])` | **3,290.65** |
| Average temperature | `AVERAGE(traffic[TempCelsius])` | **8.24 °C** (281.39 K) |

Suggested DAX, added as explicit measures rather than implicit
aggregations so that they can be formatted and reused:

```dax
Total Hours Analysed = COUNTROWS ( traffic )

Average Traffic Volume = AVERAGE ( traffic[traffic_volume] )

Average Temperature (C) = AVERAGE ( traffic[TempCelsius] )

Total Vehicles = SUM ( traffic[traffic_volume] )

High Traffic Hours % =
DIVIDE (
    CALCULATE ( COUNTROWS ( traffic ), traffic[TrafficCategory] = "High" ),
    COUNTROWS ( traffic )
)
```

For reference, `Total Vehicles` returns **133,518,143** and
`High Traffic Hours %` returns **15.05%**.

### Slicers

| Slicer | Field | Type |
| --- | --- | --- |
| Hour range | `Hour` | Between (numeric range), 0–23 |
| Weather condition | `weather_main` | Dropdown, multi-select, 11 values |
| Traffic category | `TrafficCategory` | Tile / button, Low · Medium · High |

Set the `TrafficCategory` sort order explicitly (Column tools → Sort by
column) so the tiles read Low, Medium, High rather than alphabetically as
High, Low, Medium.

### Layout

Put the three KPI cards across the top, the three slicers down a narrow
left-hand rail, and the four visuals in a 2 × 2 grid: daily trend (A) and
hourly pattern (B) on the upper row, weather impact (C) and the
temperature scatter (D) below. Sync the slicers across pages if you add a
second page.

---

## What a non-technical stakeholder should take away

The dashboard supports three statements that a mobility team can act on:

1. **The evening peak is the problem, not the morning one.** 16:00 carries
   21% more traffic than 07:00 and stays elevated for four hours rather than
   two. Interventions aimed at the morning commute are aimed at the smaller
   of the two peaks.
2. **Weather is a very weak predictor of traffic volume on this corridor.**
   It looks meaningful until hour of day is controlled for, at which point
   the effect nearly vanishes. Any dashboard that ranks weather conditions
   without showing when they occur will invite a wrong conclusion.
3. **The data has real coverage gaps, and they must be shown.** 2015 has
   195 days, 2016 has a day that reads as 277 vehicles per hour. Both are
   sensor artefacts. A dashboard that plots them without annotation will be
   read as reporting a collapse in demand that never happened.
