"""
Feature Engineering Module
===========================
Creates model-ready features from preprocessed datasets.

Features are designed to prevent data leakage:
  - Only use information available at forecast time.
  - Lag/rolling features use only past observations.
  - Timestamp features (hour, day-of-week, month) are always available.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("feature_engineering")

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed"


# ── Time-based features ────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame, ts_col: str = "DATE_TIME") -> pd.DataFrame:
    """Add calendar/cyclical time features."""
    ts = df[ts_col]

    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.day_of_week  # 0=Mon
    df["day_of_year"] = ts.dt.day_of_year
    df["month"] = ts.dt.month
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding (prevents ordinal discontinuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


# ── Lag / rolling features ────────────────────────────────────────────────────
def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
    freq_periods: int = 1,
) -> pd.DataFrame:
    """
    Add lag features.

    Parameters
    ----------
    lags : periods to lag (in dataframe rows, NOT minutes/hours).
    freq_periods : rows per hour (e.g., 4 for 15-min data; 1 for hourly).
    """
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
) -> pd.DataFrame:
    """Add rolling mean/std features (window in rows)."""
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = (
            df[target_col].shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        )
        df[f"{target_col}_roll_std_{w}"] = (
            df[target_col].shift(1).rolling(w, min_periods=max(1, w // 2)).std()
        )
    return df


# ── Solar-specific features ───────────────────────────────────────────────────
def add_solar_features(df: pd.DataFrame, ts_col: str = "DATE_TIME") -> pd.DataFrame:
    """Create solar-specific engineered features."""
    df = add_time_features(df, ts_col)

    # Daylight indicator (rough: hour 5–20)
    df["is_daylight"] = ((df["hour"] >= 5) & (df["hour"] <= 20)).astype(int)

    # Solar irradiation interactions (if available)
    if "IRRADIATION" in df.columns:
        df["irradiation_sq"] = df["IRRADIATION"] ** 2
        if "AMBIENT_TEMPERATURE" in df.columns:
            df["irrad_x_temp"] = df["IRRADIATION"] * df["AMBIENT_TEMPERATURE"]

    # Lagged AC_POWER (15-min intervals)
    if "AC_POWER" in df.columns:
        # 1-step, 4-step (1h), 8-step (2h), 96-step (1 day)
        df = add_lag_features(df, "AC_POWER", lags=[1, 2, 4, 8, 96])
        df = add_rolling_features(df, "AC_POWER", windows=[4, 8, 16, 96])

    return df


# ── Demand-specific features ──────────────────────────────────────────────────
def add_demand_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Create demand-specific engineered features (hourly)."""
    df = add_time_features(df, ts_col)

    # Lag 1h, 2h, 24h, 48h, 168h (1 week)
    target = "load_actual_entsoe_transparency"
    if target not in df.columns:
        # try alternative names
        candidates = [c for c in df.columns if "load" in c.lower() and "actual" in c.lower()]
        if candidates:
            target = candidates[0]

    if target in df.columns:
        df = add_lag_features(df, target, lags=[1, 2, 24, 48, 168])
        df = add_rolling_features(df, target, windows=[3, 6, 24, 168])

    # Weather features (if merged)
    if "temperature" in df.columns:
        df["temp_sq"] = df["temperature"] ** 2  # Heating/cooling degree effect
        df["radiation_total"] = (
            df.get("radiation_direct_horizontal", 0)
            + df.get("radiation_diffuse_horizontal", 0)
        )

    return df


# ── Chronological split ───────────────────────────────────────────────────────
def chronological_split(
    df: pd.DataFrame,
    ts_col: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    # Remaining fraction is test
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train / validation / test sets.

    IMPORTANT: No random shuffling. Uses chronological order only.
    This prevents data leakage for time-series models.
    """
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    log.info(
        "Chronological split: train=%d, val=%d, test=%d (total=%d)",
        len(train), len(val), len(test), n,
    )
    return train, val, test


# ── Feature sets ──────────────────────────────────────────────────────────────
SOLAR_FEATURE_COLS = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_daylight", "is_weekend",
    "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
    "irradiation_sq", "irrad_x_temp",
    "AC_POWER_lag_1", "AC_POWER_lag_2", "AC_POWER_lag_4", "AC_POWER_lag_8",
    "AC_POWER_lag_96",
    "AC_POWER_roll_mean_4", "AC_POWER_roll_mean_8", "AC_POWER_roll_mean_96",
    "AC_POWER_roll_std_4",
]

DEMAND_FEATURE_COLS_BASE = [
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_daylight", "is_weekend",
]


def get_available_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    """Return only feature columns that exist in the DataFrame."""
    return [c for c in feature_list if c in df.columns]


# ── Main engineering pipeline ─────────────────────────────────────────────────
def engineer_solar_dataset(plant: int = 1) -> pd.DataFrame:
    """Load preprocessed solar data and apply full feature engineering."""
    gen = pd.read_csv(PROCESSED / f"solar_generation_plant{plant}.csv")
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"])

    ws = pd.read_csv(PROCESSED / f"solar_weather_plant{plant}.csv")
    ws["DATE_TIME"] = pd.to_datetime(ws["DATE_TIME"])
    ws = ws[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

    # Merge on timestamp
    df = gen.merge(ws, on="DATE_TIME", how="left")
    df = add_solar_features(df, ts_col="DATE_TIME")

    out = PROCESSED / f"solar_features_plant{plant}.csv"
    df.to_csv(out, index=False)
    log.info("Solar features saved → %s | shape=%s", out, df.shape)
    return df


def engineer_demand_dataset(country: str = "DE") -> pd.DataFrame:
    """Load preprocessed demand + weather and apply feature engineering."""
    demand = pd.read_csv(PROCESSED / f"electricity_demand_{country}.csv")
    demand["timestamp"] = pd.to_datetime(demand["timestamp"], utc=True)

    weather = pd.read_csv(PROCESSED / f"weather_{country}.csv")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)
    weather_cols = ["timestamp"] + [c for c in weather.columns if c not in ["timestamp", "country"]]

    df = demand.merge(weather[weather_cols], on="timestamp", how="left")
    df = add_demand_features(df, ts_col="timestamp")

    out = PROCESSED / f"demand_features_{country}.csv"
    df.to_csv(out, index=False)
    log.info("Demand features saved → %s | shape=%s", out, df.shape)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engineer_solar_dataset(1)
    engineer_solar_dataset(2)
    engineer_demand_dataset("DE")
    log.info("Feature engineering complete.")
