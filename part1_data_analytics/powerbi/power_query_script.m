// =====================================================================
// Capstone Part 1 - Task 4.1 : Power Query (M) preparation script
//
// HOW TO USE
//   1. Power BI Desktop > Home > Transform data (opens Power Query Editor)
//   2. Home > Advanced Editor
//   3. Replace the contents with this script
//   4. Edit SourcePath below to point at your copy of the raw CSV
//   5. Done > Close & Apply
//
// This reproduces, step for step, the transformations applied by
// `part1_data_analytics/powerbi_prep.py`. Each step is named so that the
// Applied Steps pane reads as an auditable transformation history.
// =====================================================================
let
    // -----------------------------------------------------------------
    // Change this to the location of the raw file on your machine.
    // -----------------------------------------------------------------
    SourcePath = "C:\capstone\data\raw\Metro_Interstate_Traffic_Volume.csv",

    // Step 1 - Load the comma-delimited file (9 columns, UTF-8)
    Source = Csv.Document(
        File.Contents(SourcePath),
        [Delimiter = ",", Columns = 9, Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
    ),

    // Step 2 - Promote the first row to column headers
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),

    // Step 3 - Set explicit data types.
    //          Loading everything as text and casting deliberately avoids
    //          Power Query's type guessing, which infers `holiday` as a
    //          logical column on some locales because most values are the
    //          literal word "None".
    TypedColumns = Table.TransformColumnTypes(
        PromotedHeaders,
        {
            {"holiday",             type text},
            {"temp",                type number},
            {"rain_1h",             type number},
            {"snow_1h",             type number},
            {"clouds_all",          Int64.Type},
            {"weather_main",        type text},
            {"weather_description", type text},
            {"date_time",           type datetime},
            {"traffic_volume",      Int64.Type}
        }
    ),

    // Step 4 - Standardise the categorical text columns
    TrimmedText = Table.TransformColumns(
        TypedColumns,
        {
            {"weather_main",        each Text.Proper(Text.Trim(_)), type text},
            {"weather_description", each Text.Lower(Text.Trim(_)),  type text},
            {"holiday",             each Text.Trim(_),              type text}
        }
    ),

    // Step 5 - Rank weather severity, so that when the weather feed
    //          reports the same hour more than once the most severe
    //          condition is the one kept in step 7.
    AddedSeverity = Table.AddColumn(
        TrimmedText, "WeatherSeverity",
        each
            let m = [weather_main] in
            if m = "Clear"        then 0
            else if m = "Clouds"  then 1
            else if m = "Mist"    then 2
            else if m = "Haze"    then 3
            else if m = "Fog"     then 4
            else if m = "Smoke"   then 5
            else if m = "Drizzle" then 6
            else if m = "Rain"    then 7
            else if m = "Snow"    then 8
            else if m = "Squall"  then 9
            else 10,
        Int64.Type
    ),

    // Step 6 - Sort so the most severe record for each hour sorts last
    SortedForDedup = Table.Sort(
        AddedSeverity,
        {{"date_time", Order.Ascending}, {"WeatherSeverity", Order.Descending}}
    ),

    // Step 7 - Remove duplicate timestamps.
    //          7,629 rows are removed. These are NOT data errors: the
    //          weather provider emits one record per observed condition,
    //          so a single hour can appear two or three times carrying an
    //          identical traffic count. Keeping them would inflate every
    //          SUM over traffic_volume by 17-21% per year.
    RemovedDuplicateHours = Table.Distinct(SortedForDedup, {"date_time"}),

    // Step 8 - Repair impossible temperature readings (0 Kelvin).
    //          10 rows. 0 K is absolute zero and cannot be a real reading.
    MedianTemp = List.Median(
        Table.SelectRows(RemovedDuplicateHours, each [temp] > 100)[temp]
    ),
    FixedTemperature = Table.ReplaceValue(
        RemovedDuplicateHours,
        each [temp],
        each if [temp] < 100 then MedianTemp else [temp],
        Replacer.ReplaceValue, {"temp"}
    ),

    // Step 9 - Repair impossible rainfall readings (> 9,000 mm in an hour).
    //          1 row, recording 9,831.3 mm. The world record for hourly
    //          rainfall is around 305 mm.
    MedianRain = List.Median(
        Table.SelectRows(FixedTemperature, each [rain_1h] <= 9000)[rain_1h]
    ),
    FixedRainfall = Table.ReplaceValue(
        FixedTemperature,
        each [rain_1h],
        each if [rain_1h] > 9000 then MedianRain else [rain_1h],
        Replacer.ReplaceValue, {"rain_1h"}
    ),

    // Step 10 - Extract Hour from DateTime (required by Task 4.1)
    AddedHour = Table.AddColumn(
        FixedRainfall, "Hour", each Time.Hour(DateTime.Time([date_time])), Int64.Type
    ),

    // Step 11 - Additional date parts used by the dashboard visuals
    AddedDate      = Table.AddColumn(AddedHour,   "Date",      each DateTime.Date([date_time]), type date),
    AddedYear      = Table.AddColumn(AddedDate,   "Year",      each Date.Year([date_time]), Int64.Type),
    AddedMonth     = Table.AddColumn(AddedYear,   "Month",     each Date.Month([date_time]), Int64.Type),
    AddedMonthName = Table.AddColumn(AddedMonth,  "MonthName", each Date.ToText(Date.From([date_time]), [Format="MMM", Culture="en-US"]), type text),
    AddedDayName   = Table.AddColumn(AddedMonthName, "DayOfWeek", each Date.DayOfWeekName([date_time]), type text),
    AddedIsWeekend = Table.AddColumn(
        AddedDayName, "IsWeekend",
        each Date.DayOfWeek([date_time], Day.Monday) >= 5, type logical
    ),

    // Step 12 - Temperature in Celsius (required by Task 4.1)
    AddedCelsius = Table.AddColumn(
        AddedIsWeekend, "TempCelsius",
        each Number.Round([temp] - 273.15, 2), type number
    ),

    // Step 13 - Traffic Category (required by Task 4.1)
    //           < 4,500 Low | 4,500-5,500 Medium | > 5,500 High
    AddedTrafficCategory = Table.AddColumn(
        AddedCelsius, "TrafficCategory",
        each
            if      [traffic_volume] < 4500 then "Low"
            else if [traffic_volume] <= 5500 then "Medium"
            else "High",
        type text
    ),

    // Step 14 - Holiday flag as a proper boolean
    AddedIsHoliday = Table.AddColumn(
        AddedTrafficCategory, "IsHoliday",
        each [holiday] <> "None", type logical
    ),

    // Step 15 - Drop the helper column
    RemovedHelper = Table.RemoveColumns(AddedIsHoliday, {"WeatherSeverity"}),

    // Step 16 - Final column order
    Reordered = Table.ReorderColumns(
        RemovedHelper,
        {
            "date_time", "Date", "Year", "Month", "MonthName", "Hour",
            "DayOfWeek", "IsWeekend", "holiday", "IsHoliday",
            "temp", "TempCelsius", "rain_1h", "snow_1h", "clouds_all",
            "weather_main", "weather_description",
            "traffic_volume", "TrafficCategory"
        }
    )
in
    Reordered
