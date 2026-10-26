"""
Data Validation Module
======================
Validates loaded DataFrames for:
  - Required columns
  - Expected data types
  - Timestamp continuity
  - Value range checks
  - Missing-value thresholds
"""

import logging
from typing import Any
import pandas as pd
import numpy as np

log = logging.getLogger("validation")


class ValidationError(Exception):
    pass


class DataValidator:
    """Validates a DataFrame against a schema and business rules."""

    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self.df = df
        self.name = name
        self.issues: list[str] = []

    # ── Column presence ──────────────────────────────────────────────────────
    def require_columns(self, cols: list[str]) -> "DataValidator":
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            msg = f"[{self.name}] Missing required columns: {missing}"
            self.issues.append(msg)
            log.warning(msg)
        return self

    # ── Type checks ──────────────────────────────────────────────────────────
    def require_numeric(self, cols: list[str]) -> "DataValidator":
        for col in cols:
            if col in self.df.columns and not pd.api.types.is_numeric_dtype(self.df[col]):
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                    log.info("[%s] Coerced %s to numeric.", self.name, col)
                except Exception:
                    self.issues.append(f"[{self.name}] Column {col} could not be coerced to numeric.")
        return self

    # ── Timestamp checks ─────────────────────────────────────────────────────
    def require_datetime(self, col: str) -> "DataValidator":
        if col not in self.df.columns:
            self.issues.append(f"[{self.name}] Timestamp column '{col}' not found.")
            return self
        if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                nulls = self.df[col].isnull().sum()
                if nulls > 0:
                    log.warning("[%s] %d rows have unparseable timestamps.", self.name, nulls)
            except Exception as e:
                self.issues.append(f"[{self.name}] Cannot parse timestamp '{col}': {e}")
        return self

    # ── Missing-value threshold ──────────────────────────────────────────────
    def missing_threshold(self, col: str, max_pct: float = 50.0) -> "DataValidator":
        if col not in self.df.columns:
            return self
        pct = self.df[col].isnull().mean() * 100
        if pct > max_pct:
            msg = f"[{self.name}] Column '{col}' has {pct:.1f}% missing (threshold {max_pct}%)."
            self.issues.append(msg)
            log.warning(msg)
        return self

    # ── Value range ──────────────────────────────────────────────────────────
    def value_range(self, col: str, lo: float, hi: float) -> "DataValidator":
        if col not in self.df.columns:
            return self
        num = pd.to_numeric(self.df[col], errors="coerce")
        out_of_range = ((num < lo) | (num > hi)).sum()
        if out_of_range:
            msg = f"[{self.name}] {out_of_range} rows in '{col}' outside [{lo}, {hi}]."
            self.issues.append(msg)
            log.warning(msg)
        return self

    # ── Non-negative ─────────────────────────────────────────────────────────
    def non_negative(self, cols: list[str]) -> "DataValidator":
        for col in cols:
            if col not in self.df.columns:
                continue
            num = pd.to_numeric(self.df[col], errors="coerce")
            neg = (num < 0).sum()
            if neg:
                msg = f"[{self.name}] {neg} negative values in '{col}'."
                self.issues.append(msg)
                log.warning(msg)
        return self

    # ── Duplicate rows ───────────────────────────────────────────────────────
    def no_duplicates(self, subset: list[str] | None = None) -> "DataValidator":
        dups = self.df.duplicated(subset=subset).sum()
        if dups:
            msg = f"[{self.name}] {dups} duplicate rows found."
            self.issues.append(msg)
            log.warning(msg)
        return self

    # ── Monotonic timestamps ─────────────────────────────────────────────────
    def monotonic_timestamps(self, col: str) -> "DataValidator":
        if col not in self.df.columns:
            return self
        ts = self.df[col].dropna()
        if not ts.is_monotonic_increasing:
            msg = f"[{self.name}] Timestamps in '{col}' are NOT monotonically increasing."
            self.issues.append(msg)
            log.warning(msg)
        return self

    # ── Result ───────────────────────────────────────────────────────────────
    def report(self, raise_on_error: bool = False) -> dict[str, Any]:
        result = {
            "dataset": self.name,
            "issues": self.issues,
            "passed": len(self.issues) == 0,
        }
        if self.issues:
            log.warning("[%s] Validation found %d issue(s).", self.name, len(self.issues))
        else:
            log.info("[%s] Validation passed.", self.name)
        if raise_on_error and not result["passed"]:
            raise ValidationError(f"{self.name}: " + "; ".join(self.issues))
        return result


# ── Convenience validators ─────────────────────────────────────────────────────
def validate_solar_generation(df: pd.DataFrame, name: str = "solar_gen") -> dict:
    return (
        DataValidator(df, name)
        .require_columns(["DATE_TIME", "DC_POWER", "AC_POWER", "DAILY_YIELD"])
        .require_datetime("DATE_TIME")
        .require_numeric(["DC_POWER", "AC_POWER", "DAILY_YIELD"])
        .non_negative(["DC_POWER", "AC_POWER", "DAILY_YIELD"])
        .no_duplicates(["DATE_TIME", "SOURCE_KEY"])
        .report()
    )


def validate_solar_weather(df: pd.DataFrame, name: str = "solar_weather") -> dict:
    return (
        DataValidator(df, name)
        .require_columns(["DATE_TIME", "AMBIENT_TEMPERATURE", "IRRADIATION"])
        .require_datetime("DATE_TIME")
        .require_numeric(["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"])
        .non_negative(["IRRADIATION"])
        .report()
    )


def validate_carbon_emissions(df: pd.DataFrame, name: str = "carbon") -> dict:
    return (
        DataValidator(df, name)
        .require_columns(["DATETIME", "CARBON_INTENSITY"])
        .require_datetime("DATETIME")
        .require_numeric(["CARBON_INTENSITY", "GENERATION"])
        .non_negative(["CARBON_INTENSITY", "GENERATION"])
        .report()
    )


def validate_electricity_demand(df: pd.DataFrame, name: str = "elec_demand") -> dict:
    v = DataValidator(df, name).require_columns(["utc_timestamp"]).require_datetime("utc_timestamp")
    return v.report()


def validate_weather_data(df: pd.DataFrame, name: str = "weather") -> dict:
    v = DataValidator(df, name).require_columns(["utc_timestamp"]).require_datetime("utc_timestamp")
    return v.report()
