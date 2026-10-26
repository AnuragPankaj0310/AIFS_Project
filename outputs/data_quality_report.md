# Data Quality Report
_Generated: 2026-10-26 15:43:25_

## Summary

### carbon_emissions / df_fuel_ckan.csv

- **Quality:** ✅ Low missing data
- **Missing values:** 0.0% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

### electricity_demand / time_series.xlsx

- **Quality:** ⚠️ Moderate missing data
- **Missing values:** 12.23% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

### electricity_demand / time_series_15min_singleindex.csv

- **Quality:** ⚠️ Moderate missing data
- **Missing values:** 12.23% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

### electricity_demand / time_series_30min_singleindex.csv

- **Quality:** ✅ Low missing data
- **Missing values:** 3.18% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

### electricity_demand / time_series_60min_singleindex.csv

- **Quality:** ⚠️ Moderate missing data
- **Missing values:** 8.74% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

### solar_generation / Plant_1_Generation_Data.csv

- **Quality:** ✅ Low missing data
- **Missing values:** 0.0% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

### solar_generation / Plant_1_Weather_Sensor_Data.csv

- **Quality:** ✅ Low missing data
- **Missing values:** 0.0% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

### solar_generation / Plant_2_Generation_Data.csv

- **Quality:** ✅ Low missing data
- **Missing values:** 0.0% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

### solar_generation / Plant_2_Weather_Sensor_Data.csv

- **Quality:** ✅ Low missing data
- **Missing values:** 0.0% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

### weather / datapackage.json

- **Quality:** ✅ Low missing data
- **Missing values:** 0.0% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

### weather / weather_data.csv

- **Quality:** ✅ Low missing data
- **Missing values:** 0.0% (of first 50k rows sampled)
- **Duplicate rows:** 0
- **Date range:** N/A → N/A

## Integration Notes

- **Solar Generation** (Plant 1 & 2): India, May–June 2020, 15-min resolution.
- **Carbon Emissions** (df_fuel_ckan): UK National Grid, 30-min, 2009–present.
  Contains half-hourly CARBON_INTENSITY (gCO₂/kWh) — suitable for carbon calc.
- **Electricity Demand** (time_series_*): Europe ENTSOE, hourly, 2015–present.
  Contains actual & forecast loads, solar/wind generation, day-ahead prices per country.
- **Weather** (weather_data): Europe, hourly, 1980–2019.
  Country-level temperature + radiation (direct/diffuse horizontal) in W/m².

### Geographic/Temporal Alignment
- Solar plant data (India) does NOT align geographically with European weather/demand.
- Carbon intensity (UK) is used as a general proxy; labeled clearly.
- European electricity demand and weather datasets share hourly UTC timestamps.
  → These two CAN be merged for European demand + weather analysis.
- Solar plant weather sensor data IS co-located with generation data.
  → Plant weather sensors are used for solar forecasting features.