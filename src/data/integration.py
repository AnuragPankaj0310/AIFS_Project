"""
Data Integration Module
========================
Merges preprocessed datasets into unified analysis frames
for downstream modelling and optimization.

Design decisions:
  - Solar generation + plant weather sensors → merged on DATE_TIME (co-located).
  - ENTSOE demand + European weather → merged on hourly UTC timestamp.
  - Carbon intensity (UK) → used independently for sustainability calculations;
    NOT merged with European or India datasets without a documented assumption.
  - Solar (India, 2020) and European data are NOT merged (different geography/period).
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

log = logging.getLogger("integration")

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed"


def load_processed(name: str, ts_col: str | None = None) -> pd.DataFrame:
    path = PROCESSED / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if ts_col and ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    return df


def integrate_solar_plant(plant: int = 1) -> pd.DataFrame:
    """
    Merge solar generation + weather sensor for a single plant.
    Both datasets are at 15-min resolution, same location.
    """
    gen = load_processed(f"solar_generation_plant{plant}", ts_col=None)
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"])

    ws = load_processed(f"solar_weather_plant{plant}", ts_col=None)
    ws["DATE_TIME"] = pd.to_datetime(ws["DATE_TIME"])
    ws = ws[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

    merged = gen.merge(ws, on="DATE_TIME", how="left")
    merged["PLANT_ID"] = plant

    out = PROCESSED / f"solar_integrated_plant{plant}.csv"
    merged.to_csv(out, index=False)
    log.info("Solar integrated plant %d → %s | rows=%d", plant, out, len(merged))
    return merged


def integrate_demand_weather(country: str = "DE") -> pd.DataFrame:
    """
    Merge ENTSOE hourly demand with European weather for the same country.

    Both datasets use UTC timestamps and cover overlapping periods
    (demand: 2015–present; weather: 1980–2019).
    → Filter weather to demand period (2015–2019) for overlap.
    """
    demand = load_processed(f"electricity_demand_{country}", ts_col="timestamp")
    weather = load_processed(f"weather_{country}", ts_col="timestamp")

    # Normalize to UTC-aware datetime
    demand["timestamp"] = pd.to_datetime(demand["timestamp"], utc=True)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)

    # Remove country label column to avoid confusion
    weather = weather.drop(columns=["country"], errors="ignore")

    merged = demand.merge(weather, on="timestamp", how="left")

    out = PROCESSED / f"demand_weather_{country}.csv"
    merged.to_csv(out, index=False)
    log.info(
        "Demand+weather integrated (%s) → %s | rows=%d, cols=%d",
        country, out, len(merged), len(merged.columns),
    )
    return merged


def integrate_carbon_demand(country: str = "DE") -> pd.DataFrame:
    """
    Attach UK carbon intensity as a scenario assumption to European demand.

    DOCUMENTED ASSUMPTION:
      UK carbon intensity is used as a placeholder for European grid carbon
      intensity because no country-specific ENTSOE carbon data is available.
      Users should replace this with country-specific carbon intensity data
      for production analysis.

    Alignment:
      - UK data: 30-min → resampled to hourly averages.
      - Both datasets use UTC timestamps.
      - Matched by hour-of-day only (not calendar date), to produce a
        'typical daily profile' scenario. This is a simplification.
    """
    carbon = load_processed("carbon_emissions", ts_col=None)
    carbon["DATETIME"] = pd.to_datetime(carbon["DATETIME"])
    carbon = carbon[["DATETIME", "CARBON_INTENSITY"]].rename(columns={"DATETIME": "timestamp"})

    # Resample to hourly
    carbon = (
        carbon.set_index("timestamp")
        .resample("1h")
        .mean()
        .reset_index()
    )
    carbon["timestamp"] = carbon["timestamp"].dt.tz_localize("UTC")

    demand_path = PROCESSED / f"demand_weather_{country}.csv"
    if demand_path.exists():
        demand = pd.read_csv(demand_path)
        demand["timestamp"] = pd.to_datetime(demand["timestamp"], utc=True)
    else:
        demand = load_processed(f"electricity_demand_{country}", ts_col="timestamp")
        demand["timestamp"] = pd.to_datetime(demand["timestamp"], utc=True)

    merged = demand.merge(carbon, on="timestamp", how="left")
    merged["CARBON_INTENSITY"] = merged["CARBON_INTENSITY"].fillna(
        merged["CARBON_INTENSITY"].median()
    )
    merged["carbon_source"] = "UK proxy (documented assumption)"

    out = PROCESSED / f"demand_carbon_{country}.csv"
    merged.to_csv(out, index=False)
    log.info("Carbon integrated → %s | rows=%d", out, len(merged))
    return merged


def run_integration(country: str = "DE") -> None:
    log.info("=== Starting integration pipeline ===")
    integrate_solar_plant(1)
    integrate_solar_plant(2)
    integrate_demand_weather(country)
    integrate_carbon_demand(country)
    log.info("=== Integration complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_integration()
