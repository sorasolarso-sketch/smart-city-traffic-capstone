-- =====================================================================
-- Capstone Part 1 - Task 1.3 : Temperature around holidays, 2015-2017
--                              New Year's Day and Labor Day
-- =====================================================================
--
-- METHODOLOGICAL NOTE - WHY THE `holiday` COLUMN CANNOT BE USED ALONE
-- -------------------------------------------------------------------
-- Only 61 of the 48,204 rows carry a holiday name, and inspection shows
-- the label is attached to a SINGLE hour - 00:00:00 - on each holiday
-- rather than to all 24 hours of the day. Two consequences follow:
--
--   1. Filtering on `holiday <> 'None'` returns one temperature reading
--      per holiday, taken at midnight. A midnight reading is close to the
--      daily minimum and is not representative of the day.
--   2. New Year's Day 2015 carries no label at all, because the traffic
--      sensor was offline across that period, and New Year's Day 2017 is
--      labelled on 2017-01-02 (the observed public holiday) rather than
--      on 1 January.
--
-- The queries below therefore select holidays by CALENDAR DATE and
-- aggregate across all available hours of that date. Labor Day is the
-- first Monday in September: 2015-09-07, 2016-09-05, 2017-09-04.
-- Query H1 reproduces the flag-only view so the limitation is evidenced
-- rather than merely asserted.
-- ---------------------------------------------------------------------

-- ---------------------------------------------------------------------
-- H1. What the `holiday` flag alone actually returns (the limitation)
-- ---------------------------------------------------------------------
SELECT
    holiday,
    date_time,
    ROUND(temp, 2)        AS temp_kelvin,
    ROUND(temp - 273.15, 2) AS temp_celsius,
    traffic_volume
FROM (SELECT DISTINCT holiday, date_time, temp, traffic_volume FROM traffic)
WHERE holiday IN ('New Years Day', 'Labor Day')
ORDER BY date_time;

-- ---------------------------------------------------------------------
-- H2. Hours of data actually available on each holiday date
--     (context for how much weight each year's figure can carry)
-- ---------------------------------------------------------------------
WITH hourly AS (
    SELECT date_time,
           MIN(temp)           AS temp,
           MIN(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
),
holiday_dates AS (
    SELECT '2015-01-01' AS d, 'New Year''s Day' AS holiday_name, 2015 AS year UNION ALL
    SELECT '2016-01-01', 'New Year''s Day', 2016 UNION ALL
    SELECT '2017-01-01', 'New Year''s Day', 2017 UNION ALL
    SELECT '2015-09-07', 'Labor Day',       2015 UNION ALL
    SELECT '2016-09-05', 'Labor Day',       2016 UNION ALL
    SELECT '2017-09-04', 'Labor Day',       2017
)
SELECT
    hd.holiday_name,
    hd.year,
    hd.d                       AS holiday_date,
    COUNT(h.date_time)         AS hours_available,
    CASE WHEN COUNT(h.date_time) = 0 THEN 'NO DATA - sensor outage'
         WHEN COUNT(h.date_time) < 24 THEN 'partial day'
         ELSE 'full day' END   AS coverage
FROM holiday_dates hd
LEFT JOIN hourly h
       ON substr(h.date_time, 1, 10) = hd.d
GROUP BY hd.holiday_name, hd.year, hd.d
ORDER BY hd.holiday_name, hd.year;

-- ---------------------------------------------------------------------
-- H3. Temperature on each holiday, all available hours of the day
-- ---------------------------------------------------------------------
WITH hourly AS (
    SELECT date_time,
           MIN(temp)           AS temp,
           MIN(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
),
holiday_dates AS (
    SELECT '2015-01-01' AS d, 'New Year''s Day' AS holiday_name, 2015 AS year UNION ALL
    SELECT '2016-01-01', 'New Year''s Day', 2016 UNION ALL
    SELECT '2017-01-01', 'New Year''s Day', 2017 UNION ALL
    SELECT '2015-09-07', 'Labor Day',       2015 UNION ALL
    SELECT '2016-09-05', 'Labor Day',       2016 UNION ALL
    SELECT '2017-09-04', 'Labor Day',       2017
)
SELECT
    hd.holiday_name,
    hd.year,
    COUNT(*)                              AS hours,
    ROUND(AVG(h.temp), 2)                 AS avg_temp_kelvin,
    ROUND(AVG(h.temp) - 273.15, 2)        AS avg_temp_celsius,
    ROUND(MIN(h.temp) - 273.15, 2)        AS min_temp_celsius,
    ROUND(MAX(h.temp) - 273.15, 2)        AS max_temp_celsius,
    ROUND(MAX(h.temp) - MIN(h.temp), 2)   AS temp_range_kelvin,
    ROUND(AVG(h.traffic_volume), 1)       AS avg_traffic_volume
FROM holiday_dates hd
JOIN hourly h ON substr(h.date_time, 1, 10) = hd.d
GROUP BY hd.holiday_name, hd.year
ORDER BY hd.holiday_name, hd.year;

-- ---------------------------------------------------------------------
-- H4. Year-on-year change in holiday temperature and traffic
-- ---------------------------------------------------------------------
WITH hourly AS (
    SELECT date_time, MIN(temp) AS temp, MIN(traffic_volume) AS traffic_volume
    FROM traffic GROUP BY date_time
),
holiday_dates AS (
    SELECT '2015-01-01' AS d, 'New Year''s Day' AS holiday_name, 2015 AS year UNION ALL
    SELECT '2016-01-01', 'New Year''s Day', 2016 UNION ALL
    SELECT '2017-01-01', 'New Year''s Day', 2017 UNION ALL
    SELECT '2015-09-07', 'Labor Day',       2015 UNION ALL
    SELECT '2016-09-05', 'Labor Day',       2016 UNION ALL
    SELECT '2017-09-04', 'Labor Day',       2017
),
agg AS (
    SELECT hd.holiday_name, hd.year,
           AVG(h.temp) AS avg_temp,
           AVG(h.traffic_volume) AS avg_vol
    FROM holiday_dates hd
    JOIN hourly h ON substr(h.date_time, 1, 10) = hd.d
    GROUP BY hd.holiday_name, hd.year
)
SELECT
    holiday_name,
    year,
    ROUND(avg_temp - 273.15, 2) AS avg_temp_celsius,
    ROUND(avg_temp - LAG(avg_temp) OVER (PARTITION BY holiday_name ORDER BY year), 2)
        AS temp_change_vs_prior_year_K,
    ROUND(avg_vol, 1) AS avg_traffic_volume,
    ROUND(avg_vol - LAG(avg_vol) OVER (PARTITION BY holiday_name ORDER BY year), 1)
        AS traffic_change_vs_prior_year
FROM agg
ORDER BY holiday_name, year;

-- ---------------------------------------------------------------------
-- H5. Holiday versus ordinary-day baseline
--
--     Puts the holiday figures in context: a holiday temperature is only
--     interesting relative to what that time of year normally looks like,
--     and the traffic comparison shows the size of the holiday effect.
-- ---------------------------------------------------------------------
WITH hourly AS (
    SELECT date_time, MIN(temp) AS temp, MIN(traffic_volume) AS traffic_volume
    FROM traffic GROUP BY date_time
),
tagged AS (
    SELECT
        CAST(substr(date_time,1,4) AS INTEGER) AS year,
        substr(date_time,1,10)                 AS d,
        substr(date_time,6,2)                  AS month,
        temp, traffic_volume,
        CASE WHEN substr(date_time,1,10) IN
                  ('2015-01-01','2016-01-01','2017-01-01') THEN 'New Year''s Day'
             WHEN substr(date_time,1,10) IN
                  ('2015-09-07','2016-09-05','2017-09-04') THEN 'Labor Day'
             ELSE 'ordinary day' END           AS day_type
    FROM hourly
    WHERE CAST(substr(date_time,1,4) AS INTEGER) BETWEEN 2015 AND 2017
)
SELECT
    year,
    month,
    day_type,
    COUNT(*)                        AS hours,
    ROUND(AVG(temp) - 273.15, 2)    AS avg_temp_celsius,
    ROUND(AVG(traffic_volume), 1)   AS avg_traffic_volume
FROM tagged
WHERE month IN ('01', '09')
GROUP BY year, month, day_type
ORDER BY month, year, day_type;
