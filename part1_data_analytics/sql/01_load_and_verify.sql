-- =====================================================================
-- Capstone Part 1 - Task 1.1 : Load the dataset into SQLite and verify
-- Dataset : Metro Interstate Traffic Volume (westbound I-94, Minneapolis-St Paul)
-- Author  : Sora
-- =====================================================================
--
-- HOW THE DATA IS LOADED
-- ----------------------
-- Option A - sqlite3 command line (comma-delimited import):
--
--     sqlite3 traffic.db
--     sqlite> .mode csv
--     sqlite> .import data/raw/Metro_Interstate_Traffic_Volume.csv traffic
--
-- Option B - reproducible loader used in this project:
--
--     python part1_data_analytics/run_sql_analysis.py
--
-- Option B is used for the submitted results because it rebuilds the
-- database from the raw CSV every run, guaranteeing reproducibility, and
-- it applies the explicit typed schema below rather than letting .import
-- store every column as TEXT.
-- ---------------------------------------------------------------------

DROP TABLE IF EXISTS traffic;

CREATE TABLE traffic (
    holiday             TEXT,       -- 'None' or the name of a US federal holiday
    temp                REAL,       -- air temperature in Kelvin
    rain_1h             REAL,       -- rain in the hour, mm
    snow_1h             REAL,       -- snow in the hour, mm
    clouds_all          INTEGER,    -- cloud cover, percent
    weather_main        TEXT,       -- short weather category
    weather_description TEXT,       -- longer weather description
    date_time           TEXT,       -- 'YYYY-MM-DD HH:MM:SS' local (CST) time
    traffic_volume      INTEGER     -- hourly westbound I-94 vehicle count
);

-- Indexes that support the analytical queries that follow.
CREATE INDEX IF NOT EXISTS idx_traffic_date_time ON traffic(date_time);
CREATE INDEX IF NOT EXISTS idx_traffic_weather   ON traffic(weather_main);
CREATE INDEX IF NOT EXISTS idx_traffic_holiday   ON traffic(holiday);

-- =====================================================================
-- VERIFICATION QUERIES
-- =====================================================================

-- V1. Row count. Expected: 48,204 data rows (48,205 CSV lines incl. header).
SELECT 'V1_row_count' AS check_name, COUNT(*) AS value FROM traffic;

-- V2. Column count and declared types.
PRAGMA table_info(traffic);

-- V3. First five rows, to confirm the columns landed in the right order.
SELECT * FROM traffic ORDER BY date_time LIMIT 5;

-- V4. Date coverage: the dataset should span late 2012 to late 2018.
SELECT
    MIN(date_time) AS first_timestamp,
    MAX(date_time) AS last_timestamp,
    COUNT(DISTINCT substr(date_time, 1, 4)) AS distinct_years
FROM traffic;

-- V5. NULL check across every column. All should be zero: this dataset
--     encodes "no holiday" as the literal string 'None' rather than NULL.
SELECT
    SUM(CASE WHEN holiday             IS NULL THEN 1 ELSE 0 END) AS null_holiday,
    SUM(CASE WHEN temp                IS NULL THEN 1 ELSE 0 END) AS null_temp,
    SUM(CASE WHEN rain_1h             IS NULL THEN 1 ELSE 0 END) AS null_rain_1h,
    SUM(CASE WHEN snow_1h             IS NULL THEN 1 ELSE 0 END) AS null_snow_1h,
    SUM(CASE WHEN clouds_all          IS NULL THEN 1 ELSE 0 END) AS null_clouds_all,
    SUM(CASE WHEN weather_main        IS NULL THEN 1 ELSE 0 END) AS null_weather_main,
    SUM(CASE WHEN weather_description IS NULL THEN 1 ELSE 0 END) AS null_weather_description,
    SUM(CASE WHEN date_time           IS NULL THEN 1 ELSE 0 END) AS null_date_time,
    SUM(CASE WHEN traffic_volume      IS NULL THEN 1 ELSE 0 END) AS null_traffic_volume
FROM traffic;

-- V6. Range sanity check on the numeric columns. This surfaces the known
--     sensor faults: temp has 0 K readings and rain_1h has a 9,831.3 mm
--     reading, both physically impossible. They are documented here and
--     repaired in the Part 2 pipeline.
SELECT
    MIN(temp)           AS min_temp_kelvin,
    MAX(temp)           AS max_temp_kelvin,
    MIN(rain_1h)        AS min_rain_mm,
    MAX(rain_1h)        AS max_rain_mm,
    MIN(snow_1h)        AS min_snow_mm,
    MAX(snow_1h)        AS max_snow_mm,
    MIN(clouds_all)     AS min_clouds_pct,
    MAX(clouds_all)     AS max_clouds_pct,
    MIN(traffic_volume) AS min_traffic_volume,
    MAX(traffic_volume) AS max_traffic_volume
FROM traffic;

-- V7. Duplicate timestamps. The raw feed contains repeated hours where the
--     weather station reported more than one observation for the same hour.
SELECT
    COUNT(*)                    AS total_rows,
    COUNT(DISTINCT date_time)   AS distinct_timestamps,
    COUNT(*) - COUNT(DISTINCT date_time) AS duplicate_timestamp_rows
FROM traffic;

-- V8. Fully identical duplicate rows.
SELECT COUNT(*) AS exact_duplicate_rows
FROM (
    SELECT holiday, temp, rain_1h, snow_1h, clouds_all,
           weather_main, weather_description, date_time, traffic_volume,
           COUNT(*) AS n
    FROM traffic
    GROUP BY holiday, temp, rain_1h, snow_1h, clouds_all,
             weather_main, weather_description, date_time, traffic_volume
    HAVING COUNT(*) > 1
);

-- V9. Distinct weather categories and how often each occurs.
SELECT weather_main, COUNT(*) AS hours
FROM traffic
GROUP BY weather_main
ORDER BY hours DESC;

-- V10. Hours recorded per calendar year. This exposes the large gap in
--      2014-2015 where the traffic sensor was offline, which matters for
--      the year-on-year comparison in Task 1.2.
SELECT
    substr(date_time, 1, 4) AS year,
    COUNT(*)                AS hours_recorded,
    ROUND(COUNT(*) * 100.0 / 8760, 1) AS pct_of_full_year
FROM traffic
GROUP BY year
ORDER BY year;
