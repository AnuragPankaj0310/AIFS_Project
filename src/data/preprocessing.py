"""
Data Preprocessing Module
==========================
Cleans, parses, and normalises each raw dataset.

Key decisions documented inline:
  - Solar generation: India plants, 15-min resolution, May–June 2020.
  - Carbon emissions: UK National Grid, 30-min, 2009–present.
  - Electricity demand: Europe ENTSOE, hourly, 2015–present.
  - Weather: Europe, hourly, 1980–2019. Chosen country = DE (Germany) as
    largest European energy market with good data coverage.

Run directly:
    python -m src.data.preprocessing
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("preprocessing")

BASE_DIR = Path(__file__).resolve().parents[2]
RAW = BASE_DIR
PROCESSED = BASE_DIR / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "preprocessing.log", mode="a"),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# SOLAR GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_solar_generation(plant: int = 1) -> pd.DataFrame:
    """
    Loads and cleans solar generation data for Plant 1 or 2.

    Decisions:
      - DC_POWER and AC_POWER in kW (15-min intervals).
      - DAILY_YIELD and TOTAL_YIELD in kWh.
      - Aggregate multi-inverter readings to plant-level totals per timestamp.
      - Negative values replaced by 0 (measurement artefacts at night).
      - Missing timestamp gaps filled with NaN, not interpolated (to avoid
        leaking future generation into model features).
    """
    gen_path = RAW / "Solar Panel Electricity Generation" / f"Plant_{plant}_Generation_Data.csv"
    log.info("Loading solar generation: %s", gen_path.name)
    df = pd.read_csv(gen_path, low_memory=False)

    # Parse timestamps
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    df = df.dropna(subset=["DATE_TIME"])

    # Numeric coercion
    for col in ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    # Aggregate per timestamp (sum across inverters/source keys)
    agg = (
        df.groupby("DATE_TIME")
        .agg(
            DC_POWER=("DC_POWER", "sum"),
            AC_POWER=("AC_POWER", "sum"),
            DAILY_YIELD=("DAILY_YIELD", "sum"),
            TOTAL_YIELD=("TOTAL_YIELD", "sum"),
            INVERTER_COUNT=("SOURCE_KEY", "count"),
        )
        .reset_index()
        .sort_values("DATE_TIME")
    )

    agg["PLANT_ID"] = plant

    # Add complete 15-min index (expose gaps as NaN)
    full_idx = pd.date_range(agg["DATE_TIME"].min(), agg["DATE_TIME"].max(), freq="15min")
    agg = agg.set_index("DATE_TIME").reindex(full_idx).rename_axis("DATE_TIME").reset_index()
    agg["PLANT_ID"] = agg["PLANT_ID"].fillna(plant)

    # UNIT NOTE: DC_POWER/AC_POWER in kW, DAILY/TOTAL_YIELD in kWh
    log.info("Solar gen Plant %d: %d rows, %s → %s", plant, len(agg),
             agg["DATE_TIME"].min(), agg["DATE_TIME"].max())

    out_path = PROCESSED / f"solar_generation_plant{plant}.csv"
    agg.to_csv(out_path, index=False)
    log.info("Saved → %s", out_path)
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# SOLAR WEATHER SENSOR
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_solar_weather(plant: int = 1) -> pd.DataFrame:
    """
    Loads plant-level weather sensor data (co-located with solar plant).

    Variables:
      - AMBIENT_TEMPERATURE  (°C)
      - MODULE_TEMPERATURE   (°C)
      - IRRADIATION          (W/m²) – direct sensor measurement
    """
    ws_path = RAW / "Solar Panel Electricity Generation" / f"Plant_{plant}_Weather_Sensor_Data.csv"
    log.info("Loading solar weather sensor: %s", ws_path.name)
    df = pd.read_csv(ws_path, low_memory=False)

    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    df = df.dropna(subset=["DATE_TIME"])

    for col in ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # One row per timestamp (single weather sensor per plant)
    df = df.sort_values("DATE_TIME").drop_duplicates(subset="DATE_TIME")
    df["PLANT_ID"] = plant

    full_idx = pd.date_range(df["DATE_TIME"].min(), df["DATE_TIME"].max(), freq="15min")
    df = df.set_index("DATE_TIME").reindex(full_idx).rename_axis("DATE_TIME").reset_index()
    # Interpolate weather gaps (linear, short gaps only)
    for col in ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]:
        df[col] = df[col].interpolate(method="linear", limit=4)
    df["PLANT_ID"] = df["PLANT_ID"].fillna(plant)

    out_path = PROCESSED / f"solar_weather_plant{plant}.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved solar weather -> %s", out_path)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CARBON EMISSIONS (UK)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_carbon_emissions() -> pd.DataFrame:
    """
    Loads UK National Grid carbon intensity data.

    Key columns used:
      - DATETIME          – half-hourly UTC
      - CARBON_INTENSITY  – gCO₂/kWh (actual carbon intensity)
      - GENERATION        – MW (total generation)
      - fuel columns      – MW per fuel type
      - *_perc columns    – percentage share

    NOTE: This is UK grid data. Not used as a direct proxy for India or
    Europe without explicit labeling. Used for:
      (a) Time-of-use carbon intensity for demand response analysis.
      (b) Fuel-mix analysis and visualization.
    """
    path = RAW / "Carbon Emissions Data" / "df_fuel_ckan.csv"
    log.info("Loading carbon emissions: %s", path.name)

    # Read full file
    df = pd.read_csv(path, low_memory=False)
    log.info("Rows loaded: %d", len(df))

    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    df = df.dropna(subset=["DATETIME"]).sort_values("DATETIME")

    numeric_cols = [c for c in df.columns if c != "DATETIME"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clip obviously wrong values (negative generation / intensity)
    df["GENERATION"] = df["GENERATION"].clip(lower=0)
    df["CARBON_INTENSITY"] = df["CARBON_INTENSITY"].clip(lower=0)

    # Add hour-of-day and date for merging / features
    df["hour"] = df["DATETIME"].dt.hour
    df["date"] = df["DATETIME"].dt.date

    out_path = PROCESSED / "carbon_emissions.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved carbon emissions → %s | rows=%d", out_path, len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# ELECTRICITY DEMAND (Europe ENTSOE)
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_electricity_demand(country: str = "DE", resolution: str = "60min") -> pd.DataFrame:
    """
    Loads and preprocesses ENTSOE electricity time-series for a given country.

    Assumptions:
      - Uses time_series_60min_singleindex.csv (hourly) by default.
      - Columns of interest: {country}_load_actual, {country}_load_forecast,
        {country}_solar_generation_actual, {country}_price_day_ahead.
      - Germany (DE) is chosen as the default: largest market, best coverage.

    Units:
      - Load: MW
      - Solar generation: MW
      - Price: EUR/MWh
    """
    fname = f"time_series_{resolution}_singleindex.csv"
    path = RAW / "Electricity consumption and demand" / fname
    log.info("Loading electricity demand (%s, %s): %s", country, resolution, fname)

    # Read only relevant columns to avoid OOM on 130MB file
    # First peek at headers
    header = pd.read_csv(path, nrows=0)
    target_cols_prefix = [country]
    relevant = ["utc_timestamp"] + [c for c in header.columns if c.startswith(f"{country}_")]
    log.info("Relevant columns for %s: %s", country, relevant)

    df = pd.read_csv(path, usecols=relevant, low_memory=False)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["utc_timestamp"]).sort_values("utc_timestamp")

    for col in df.columns:
        if col != "utc_timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rename for convenience
    rename_map = {}
    for col in df.columns:
        new = col.replace(f"{country}_", "")
        rename_map[col] = new
    df = df.rename(columns=rename_map)
    df = df.rename(columns={"utc_timestamp": "timestamp"})

    # Forward-fill short gaps (up to 3 hours)
    for col in df.columns:
        if col != "timestamp":
            df[col] = df[col].ffill(limit=3)

    df["country"] = country

    out_path = PROCESSED / f"electricity_demand_{country}.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved electricity demand → %s | rows=%d", out_path, len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# EUROPEAN WEATHER DATA
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_weather(country: str = "DE") -> pd.DataFrame:
    """
    Loads Open Power System Data weather for a given European country.

    Variables (all hourly):
      - {country}_temperature              (°C)
      - {country}_radiation_direct_horizontal  (W/m²)
      - {country}_radiation_diffuse_horizontal (W/m²)

    NOTE: This covers 1980–2019. For alignment with ENTSOE demand (2015–),
    we filter to 2015–2019. The solar plant data (India 2020) is geographically
    incompatible, so weather is NOT merged with it; instead the plant's own
    weather sensors are used.
    """
    path = RAW / "Weather and solar energy" / "weather_data.csv"
    log.info("Loading European weather for %s", country)

    header = pd.read_csv(path, nrows=0)
    target_cols = ["utc_timestamp"] + [c for c in header.columns if c.startswith(f"{country}_")]
    log.info("Weather columns for %s: %s", country, target_cols)

    df = pd.read_csv(path, usecols=target_cols, low_memory=False)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["utc_timestamp"]).sort_values("utc_timestamp")

    for col in df.columns:
        if col != "utc_timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rename_map = {c: c.replace(f"{country}_", "") for c in df.columns if c.startswith(f"{country}_")}
    rename_map["utc_timestamp"] = "timestamp"
    df = df.rename(columns=rename_map)
    df["country"] = country

    out_path = PROCESSED / f"weather_{country}.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved weather → %s | rows=%d", out_path, len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ──────────────────────────────────────────────────────────────────────────────
def run_all(country: str = "DE"):
    log.info("=== Starting full preprocessing pipeline (country=%s) ===", country)

    log.info("--- Solar Generation ---")
    preprocess_solar_generation(plant=1)
    preprocess_solar_generation(plant=2)

    log.info("--- Solar Weather Sensors ---")
    preprocess_solar_weather(plant=1)
    preprocess_solar_weather(plant=2)

    log.info("--- Carbon Emissions ---")
    preprocess_carbon_emissions()

    log.info("--- Electricity Demand ---")
    preprocess_electricity_demand(country=country, resolution="60min")

    log.info("--- European Weather ---")
    preprocess_weather(country=country)

    log.info("=== Preprocessing complete ===")


if __name__ == "__main__":
    run_all()
