-- =====================================================================
-- Capstone Part 1 - Task 1.2 : Annual traffic volume trends, 2012-2017
-- =====================================================================
--
-- METHODOLOGICAL NOTE - WHY THE QUERIES DE-DUPLICATE FIRST
-- --------------------------------------------------------
-- The raw feed contains 7,629 rows whose timestamp is already present on
-- another row. These are not sensor errors: the weather provider splits a
-- single hour into several records when more than one condition is
-- observed (for example 'Rain / light rain' and 'Drizzle / light intensity
-- drizzle' for the same hour). Verification confirms that every one of the
-- 5,445 affected hours carries an IDENTICAL traffic_volume on each of its
-- rows, so a naive SUM(traffic_volume) counts those hours two or three
-- times and inflates the annual totals.
--
-- Every query below therefore starts from `hourly`, a CTE that keeps one
-- row per distinct timestamp. Query A2 quantifies the size of the error
-- that this avoids.
-- ---------------------------------------------------------------------

-- ---------------------------------------------------------------------
-- A1. Total and average traffic volume per year, 2012-2017
-- ---------------------------------------------------------------------
WITH hourly AS (
    SELECT
        date_time,
        MIN(traffic_volume) AS traffic_volume,   -- identical across duplicates
        MIN(temp)           AS temp
    FROM traffic
    GROUP BY date_time
)
SELECT
    CAST(substr(date_time, 1, 4) AS INTEGER) AS year,
    COUNT(*)                                  AS hours_recorded,
    ROUND(COUNT(*) * 100.0 / 8760, 1)         AS pct_of_year_covered,
    SUM(traffic_volume)                       AS total_traffic_volume,
    ROUND(AVG(traffic_volume), 1)             AS avg_hourly_traffic_volume
FROM hourly
WHERE CAST(substr(date_time, 1, 4) AS INTEGER) BETWEEN 2012 AND 2017
GROUP BY year
ORDER BY year;

-- ---------------------------------------------------------------------
-- A2. Size of the double-counting error avoided by de-duplication
-- ---------------------------------------------------------------------
WITH raw_totals AS (
    SELECT CAST(substr(date_time, 1, 4) AS INTEGER) AS year,
           SUM(traffic_volume) AS raw_total
    FROM traffic
    GROUP BY year
),
dedup_totals AS (
    SELECT year, SUM(traffic_volume) AS dedup_total
    FROM (
        SELECT CAST(substr(date_time, 1, 4) AS INTEGER) AS year,
               date_time,
               MIN(traffic_volume) AS traffic_volume
        FROM traffic
        GROUP BY date_time
    )
    GROUP BY year
)
SELECT
    r.year,
    r.raw_total,
    d.dedup_total,
    r.raw_total - d.dedup_total AS vehicles_double_counted,
    ROUND((r.raw_total - d.dedup_total) * 100.0 / d.dedup_total, 2) AS pct_inflation
FROM raw_totals r
JOIN dedup_totals d ON d.year = r.year
WHERE r.year BETWEEN 2012 AND 2017
ORDER BY r.year;

-- ---------------------------------------------------------------------
-- A3. Year-on-year change, on BOTH measures
--
--     total_traffic_volume   -> the raw yearly sum requested by the task
--     avg_hourly_traffic_volume -> the coverage-robust comparison
--
--     Because the number of hours actually recorded swings from 2,103
--     (2012) to 8,713 (2017), the change in the yearly TOTAL is dominated
--     by sensor uptime rather than by real demand. The change in the
--     AVERAGE is the figure that supports a defensible conclusion.
-- ---------------------------------------------------------------------
WITH hourly AS (
    SELECT date_time, MIN(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
),
yearly AS (
    SELECT
        CAST(substr(date_time, 1, 4) AS INTEGER) AS year,
        COUNT(*)            AS hours_recorded,
        SUM(traffic_volume) AS total_volume,
        AVG(traffic_volume) AS avg_volume
    FROM hourly
    WHERE CAST(substr(date_time, 1, 4) AS INTEGER) BETWEEN 2012 AND 2017
    GROUP BY year
)
SELECT
    year,
    hours_recorded,
    hours_recorded - LAG(hours_recorded) OVER (ORDER BY year) AS change_in_hours_recorded,
    total_volume,
    total_volume - LAG(total_volume) OVER (ORDER BY year)     AS change_in_total,
    ROUND(
        (total_volume - LAG(total_volume) OVER (ORDER BY year)) * 100.0
        / LAG(total_volume) OVER (ORDER BY year), 2)          AS pct_change_in_total,
    ROUND(avg_volume, 1)                                      AS avg_volume,
    ROUND(avg_volume - LAG(avg_volume) OVER (ORDER BY year), 1) AS change_in_avg,
    ROUND(
        (avg_volume - LAG(avg_volume) OVER (ORDER BY year)) * 100.0
        / LAG(avg_volume) OVER (ORDER BY year), 2)            AS pct_change_in_avg,
    CASE
        WHEN LAG(avg_volume) OVER (ORDER BY year) IS NULL THEN 'baseline year'
        WHEN avg_volume > LAG(avg_volume) OVER (ORDER BY year) THEN 'increase'
        WHEN avg_volume < LAG(avg_volume) OVER (ORDER BY year) THEN 'decrease'
        ELSE 'no change'
    END AS direction_on_average
FROM yearly
ORDER BY year;

-- ---------------------------------------------------------------------
-- A4. Seasonally adjusted comparison (month-adjusted index)
--
--     A like-for-like comparison on a shared set of calendar months is
--     impossible here: query A5 shows that NO month is present in all six
--     years. 2012 begins in October, 2014 ends in August, and 2015 has no
--     data before June, so any fixed set of months is empty.
--
--     This query instead removes the seasonal-composition bias
--     arithmetically. For each (year, month) it computes that month's mean
--     volume, subtracts the mean for the SAME month pooled across all
--     years, and averages the resulting deviations over whichever months
--     the year actually contains. The result reads as "vehicles per hour
--     above or below what this corridor normally carries at that time of
--     year", and is comparable across years with different coverage.
-- ---------------------------------------------------------------------
WITH hourly AS (
    SELECT date_time, MIN(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
),
labelled AS (
    SELECT
        CAST(substr(date_time, 1, 4) AS INTEGER) AS year,
        substr(date_time, 6, 2)                  AS month,
        traffic_volume
    FROM hourly
    WHERE CAST(substr(date_time, 1, 4) AS INTEGER) BETWEEN 2012 AND 2017
),
year_month AS (
    SELECT year, month, AVG(traffic_volume) AS ym_mean, COUNT(*) AS hours
    FROM labelled
    GROUP BY year, month
),
month_norm AS (
    SELECT month, AVG(traffic_volume) AS month_mean
    FROM labelled
    GROUP BY month
)
SELECT
    ym.year,
    COUNT(*)                                       AS months_present,
    SUM(ym.hours)                                  AS hours_recorded,
    ROUND(AVG(ym.ym_mean), 1)                      AS raw_avg_of_monthly_means,
    ROUND(AVG(ym.ym_mean - mn.month_mean), 1)      AS seasonally_adjusted_index,
    ROUND(AVG(ym.ym_mean - mn.month_mean)
          - LAG(AVG(ym.ym_mean - mn.month_mean)) OVER (ORDER BY ym.year), 1)
                                                   AS change_vs_prior_year,
    CASE
        WHEN LAG(AVG(ym.ym_mean - mn.month_mean)) OVER (ORDER BY ym.year) IS NULL
            THEN 'baseline year'
        WHEN AVG(ym.ym_mean - mn.month_mean)
             > LAG(AVG(ym.ym_mean - mn.month_mean)) OVER (ORDER BY ym.year)
            THEN 'increase'
        ELSE 'decrease'
    END                                            AS direction_adjusted
FROM year_month ym
JOIN month_norm mn ON mn.month = ym.month
GROUP BY ym.year
ORDER BY ym.year;

-- ---------------------------------------------------------------------
-- A5. Supporting detail: monthly coverage matrix
--     Documents exactly which months are missing in which year.
-- ---------------------------------------------------------------------
SELECT
    substr(date_time, 6, 2) AS month,
    SUM(CASE WHEN substr(date_time,1,4)='2012' THEN 1 ELSE 0 END) AS y2012,
    SUM(CASE WHEN substr(date_time,1,4)='2013' THEN 1 ELSE 0 END) AS y2013,
    SUM(CASE WHEN substr(date_time,1,4)='2014' THEN 1 ELSE 0 END) AS y2014,
    SUM(CASE WHEN substr(date_time,1,4)='2015' THEN 1 ELSE 0 END) AS y2015,
    SUM(CASE WHEN substr(date_time,1,4)='2016' THEN 1 ELSE 0 END) AS y2016,
    SUM(CASE WHEN substr(date_time,1,4)='2017' THEN 1 ELSE 0 END) AS y2017
FROM (SELECT DISTINCT date_time FROM traffic)
GROUP BY month
ORDER BY month;
