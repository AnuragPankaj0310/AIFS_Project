# Data Directory – AIFS Project

> **Building-to-Grid Demand Response Optimization via Predictive AI and Dynamic Load Shifting**

This directory contains all preprocessed datasets used by the AIFS pipeline.
Raw source files live in the top-level domain folders (see §2 below).

---

## Directory Structure

```
data/
└── processed/                        ← All ETL outputs (ready for modelling)
    ├── solar_generation_plant1.csv   ← Cleaned solar AC/DC power (Plant 1)
    ├── solar_generation_plant2.csv   ← Cleaned solar AC/DC power (Plant 2)
    ├── solar_weather_plant1.csv      ← Co-located weather sensors (Plant 1)
    ├── solar_weather_plant2.csv      ← Co-located weather sensors (Plant 2)
    ├── solar_integrated_plant1.csv   ← Solar + weather merged (Plant 1)
    ├── solar_integrated_plant2.csv   ← Solar + weather merged (Plant 2)
    ├── solar_features_plant1.csv     ← Feature-engineered dataset for forecasting (Plant 1)
    ├── solar_features_plant2.csv     ← Feature-engineered dataset for forecasting (Plant 2)
    ├── electricity_demand_DE.csv     ← Cleaned hourly demand for Germany (DE)
    ├── weather_DE.csv                ← Hourly weather for Germany (DE)
    ├── demand_weather_DE.csv         ← Demand + weather merged (DE)
    ├── demand_carbon_DE.csv          ← Demand + weather + carbon intensity (DE)
    ├── demand_features_DE.csv        ← Feature-engineered demand dataset (DE)
    └── carbon_emissions.csv          ← UK National Grid half-hourly carbon intensity
```

**Raw data sources** (original, untouched) are stored in the top-level folders:

```
AIFS PROJECT/
├── Solar Panel Electricity Generation/   ← Plant 1 & 2 CSV (Kaggle)
├── Weather and solar energy/             ← Co-located weather sensor CSVs
├── Electricity consumption and demand/   ← ENTSOE time_series CSVs + XLSX
└── Carbon Emissions Data/                ← UK ESO df_fuel_ckan.csv
```

---

## Processed Files Reference

### Solar Generation

| File | Rows | Columns | Time Range | Resolution |
|---|---|---|---|---|
| `solar_generation_plant1.csv` | 3,264 | 7 | 2020-05-15 → 2020-06-17 | 15-min |
| `solar_generation_plant2.csv` | 3,264 | 7 | 2020-05-15 → 2020-06-17 | 15-min |
| `solar_weather_plant1.csv` | 3,264 | 6 | 2020-05-15 → 2020-06-17 | 15-min |
| `solar_weather_plant2.csv` | 3,264 | 6 | 2020-05-15 → 2020-06-17 | 15-min |
| `solar_integrated_plant1.csv` | 3,264 | 10 | 2020-05-15 → 2020-06-17 | 15-min |
| `solar_integrated_plant2.csv` | 3,264 | 10 | 2020-05-15 → 2020-06-17 | 15-min |
| `solar_features_plant1.csv` | 3,264 | 39 | 2020-05-15 → 2020-06-17 | 15-min |
| `solar_features_plant2.csv` | 3,264 | 39 | 2020-05-15 → 2020-06-17 | 15-min |

**Key columns in `solar_generation_*.csv`:**
- `DATE_TIME` – timestamp (IST, 15-min intervals)
- `DC_POWER` – DC power output (kW)
- `AC_POWER` – AC power output after inverter (kW) ← **forecast target**
- `DAILY_YIELD` – cumulative energy yield for the day (kWh)
- `TOTAL_YIELD` – cumulative total yield (kWh)
- `PLANT_ID`, `SOURCE_KEY` – plant and inverter identifiers

**Key columns in `solar_weather_*.csv`:**
- `AMBIENT_TEMPERATURE` – ambient air temperature (°C)
- `MODULE_TEMPERATURE` – PV module surface temperature (°C)
- `IRRADIATION` – solar irradiation (W/m²)

**Key columns in `solar_features_*.csv`** (adds time + lag features):
- All above columns plus:
- `hour`, `minute`, `dayofweek`, `month` – calendar features
- `hour_sin`, `hour_cos`, `month_sin`, `month_cos` – cyclical encodings
- `AC_POWER_lag_1` … `AC_POWER_lag_96` – lag features (up to 24h back)
- `AC_POWER_roll_4_mean`, `_roll_12_mean`, `_roll_24_mean` – rolling stats

---

### Electricity Demand (Germany – DE)

| File | Rows (sampled) | Columns | Time Range | Resolution |
|---|---|---|---|---|
| `electricity_demand_DE.csv` | 47,000+ | 43 | 2015-01-01 → 2020 | Hourly |
| `weather_DE.csv` | 350,000+ | 5 | 1980-01-01 → 2019 | Hourly |
| `demand_weather_DE.csv` | 47,000+ | 46 | 2015-01-01 → 2020 | Hourly |
| `demand_carbon_DE.csv` | 47,000+ | 48 | 2015-01-01 → 2020 | Hourly |
| `demand_features_DE.csv` | 47,000+ | 74 | 2015-01-01 → 2020 | Hourly |

**Key columns in `electricity_demand_DE.csv`:**
- `timestamp` – UTC hourly timestamp
- `load_actual_entsoe_transparency` – actual grid load (MW) ← **forecast target**
- `load_forecast_entsoe_transparency` – official ENTSOE day-ahead forecast (MW)
- `solar_generation_actual` – solar generation in grid (MW)
- `wind_onshore_generation_actual` – wind generation (MW)
- `price_day_ahead` – day-ahead electricity price (EUR/MWh)

**Key columns in `weather_DE.csv`:**
- `temperature` – 2m air temperature (°C)
- `radiation_direct_horizontal` – direct horizontal solar irradiance (W/m²)
- `radiation_diffuse_horizontal` – diffuse horizontal solar irradiance (W/m²)
- `country` – country code (filtered to `DE`)

**Key columns in `demand_features_DE.csv`** (adds time + lag features):
- All above columns plus:
- `hour`, `dayofweek`, `month`, `is_weekend` – calendar features
- `hour_sin`, `hour_cos`, `month_sin`, `month_cos` – cyclical encodings
- `load_lag_1h`, `load_lag_24h`, `load_lag_168h` – lag features
- `load_roll_6h_mean`, `load_roll_24h_mean`, `load_roll_168h_mean` – rolling stats

---

### Carbon Emissions

| File | Rows (sampled) | Columns | Time Range | Resolution |
|---|---|---|---|---|
| `carbon_emissions.csv` | 335,000+ | 36 | 2009-01-01 → present | 30-min |

**Key columns:**
- `DATETIME` – UTC half-hourly timestamp
- `CARBON_INTENSITY` – grid carbon intensity (gCO₂/kWh)
- `GAS`, `COAL`, `NUCLEAR`, `WIND`, `SOLAR`, `BIOMASS`, `HYDRO`, `IMPORTS`, `OTHER` – fuel mix fractions (%)

> ⚠️ **Geographic note**: Carbon intensity data is from the **UK National Grid** and is used as a general proxy for carbon calculations. It is **NOT** geographically representative of the German or European grid. For production use, replace with country-specific ENTSO-E or Ember data.

---

## Data Pipeline

The preprocessing pipeline transforms raw data into model-ready features through these steps:

```
Raw Sources
    │
    ▼
src/data/discovery.py       ← Dataset inventory & quality report
    │
    ▼
src/data/preprocessing.py   ← ETL: parse timestamps, handle missing, clip outliers
    │
    ▼
src/data/integration.py     ← Merge solar+weather, demand+weather+carbon
    │
    ▼
src/data/feature_engineering.py ← Time features, lag features, rolling stats,
                                   chronological train/val/test split (70/15/15)
    │
    ▼
data/processed/*.csv        ← Ready for modelling
```

### Missing Data Handling

| Dataset | Missing % | Strategy |
|---|---|---|
| Solar generation | 0% | None needed |
| Solar weather sensors | 0% | None needed |
| Carbon emissions | 0% | None needed |
| Electricity demand (60-min) | ~8.7% | Forward-fill → backward-fill → median imputation |
| Electricity demand (30-min) | ~3.2% | Forward-fill → backward-fill → median imputation |
| European weather | ~0% | None needed |

---

## Data Sources & Citations

| Dataset | Source | License | Citation |
|---|---|---|---|
| Solar Power Generation (Plant 1 & 2) | Kaggle – Anil Karanal | CC0 | [Kaggle Dataset](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) |
| European Weather Data | Open Power System Data | [OPSD License](https://open-power-system-data.org) | Ohlendorf et al. (2020). DOI: [10.25832/weather_data/2020-09-16](https://doi.org/10.25832/weather_data/2020-09-16) |
| European Electricity Demand | Open Power System Data (ENTSO-E) | [OPSD License](https://open-power-system-data.org) | OPSD (2020). DOI: [10.25832/time_series/2020-10-06](https://doi.org/10.25832/time_series/2020-10-06) |
| UK Carbon Intensity | National Grid ESO | [OGL v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/) | [Carbon Intensity API](https://carbonintensity.org.uk) |

---

## Reproducing the Processed Data

If you need to regenerate all processed files from scratch:

```bash
# Step 1: Inventory all raw datasets
python -m src.data.discovery

# Step 2: Preprocessing & ETL
python -m src.data.preprocessing

# Step 3: Integration (merge multi-source datasets)
python -m src.data.integration

# Step 4: Feature engineering + chronological splits
python -m src.data.feature_engineering
```

> **Note:** Steps 2–4 read from the top-level raw source folders and write to `data/processed/`.
> These steps are idempotent — safe to re-run.

---

## Key Design Decisions

1. **No data leakage**: All lag/rolling features are computed with `shift()` so future values are never used. Chronological 70/15/15 splits prevent any temporal leakage.
2. **Geographic separation**: Indian solar data (Plant 1 & 2) is used **only** for solar forecasting and is **never** merged with European demand/weather data.
3. **Carbon intensity as proxy**: UK National Grid carbon intensity is the only available half-hourly carbon dataset. It is clearly labeled and documented everywhere it is used.
4. **Cyclical encoding**: Hour-of-day and month-of-year are encoded as sine/cosine pairs to preserve their circular nature for ML models.
