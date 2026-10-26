"""
Dataset Discovery Module
========================
Inspects all four raw data folders and generates:
  - outputs/dataset_inventory.csv
  - outputs/dataset_inventory.md
  - outputs/data_quality_report.md

Run directly:
    python -m src.data.discovery
"""

import os
import sys
import csv
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/discovery.log", mode="a"),
    ],
)
log = logging.getLogger("discovery")

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FOLDERS = {
    "Carbon Emissions Data": "carbon_emissions",
    "Electricity consumption and demand": "electricity_demand",
    "Solar Panel Electricity Generation": "solar_generation",
    "Weather and solar energy": "weather",
}
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helper: read any supported format ──────────────────────────────────────────
def _load_file(path: Path, nrows: int = 50_000) -> pd.DataFrame | None:
    """Load a file into a DataFrame (limited rows for inspection)."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, nrows=nrows, low_memory=False)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path, nrows=nrows)
        elif suffix == ".parquet":
            return pd.read_parquet(path).head(nrows)
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data[:nrows])
            elif isinstance(data, dict):
                return pd.DataFrame([data])
        else:
            log.warning("Unsupported format: %s", path.name)
            return None
    except Exception as exc:
        log.error("Failed to load %s: %s", path.name, exc)
        return None


def _full_row_count(path: Path) -> int:
    """Count rows without loading the full file into memory."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in f) - 1  # subtract header
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
            return len(df)
        elif suffix == ".parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            return pf.metadata.num_rows
        else:
            return -1
    except Exception:
        return -1


def _detect_timestamp_col(df: pd.DataFrame) -> str | None:
    """Return the first column name that looks like a timestamp."""
    ts_keywords = ["timestamp", "datetime", "date", "time", "utc", "cet"]
    for col in df.columns:
        if any(kw in col.lower() for kw in ts_keywords):
            return col
    # Fallback: first object/string column that can be parsed
    for col in df.select_dtypes(include=["object"]).columns[:3]:
        try:
            pd.to_datetime(df[col].dropna().head(5), )
            return col
        except Exception:
            pass
    return None


def _date_range(df: pd.DataFrame, ts_col: str | None) -> tuple[str, str]:
    if ts_col is None:
        return "N/A", "N/A"
    try:
        parsed = pd.to_datetime(df[ts_col], errors="coerce")
        return str(parsed.min()), str(parsed.max())
    except Exception:
        return "N/A", "N/A"


def _missing_pct(df: pd.DataFrame) -> float:
    total = df.size
    if total == 0:
        return 0.0
    return round(df.isnull().sum().sum() / total * 100, 2)


def _dup_count(df: pd.DataFrame) -> int:
    try:
        # Only check hashable (non-list/dict) columns
        hashable_cols = []
        for col in df.columns:
            sample = df[col].dropna().head(1)
            if len(sample) > 0:
                try:
                    hash(sample.iloc[0])
                    hashable_cols.append(col)
                except TypeError:
                    pass
        if not hashable_cols:
            return 0
        return int(df[hashable_cols].duplicated().sum())
    except Exception:
        return -1


# ── Core inspection ────────────────────────────────────────────────────────────
def inspect_file(path: Path, folder_label: str) -> dict:
    """Produce a metadata dictionary for one file."""
    log.info("Inspecting %s", path.name)
    size_mb = round(path.stat().st_size / 1_048_576, 2)
    df = _load_file(path)

    if df is None:
        return {
            "folder": folder_label,
            "filename": path.name,
            "format": path.suffix.upper().lstrip("."),
            "size_mb": size_mb,
            "full_rows": -1,
            "columns": 0,
            "column_names": "N/A",
            "dtypes": "N/A",
            "timestamp_col": "N/A",
            "ts_format": "N/A",
            "date_start": "N/A",
            "date_end": "N/A",
            "missing_pct": "N/A",
            "duplicate_rows": "N/A",
            "notes": "Could not load file",
        }

    ts_col = _detect_timestamp_col(df)
    date_start, date_end = _date_range(df, ts_col)
    full_rows = _full_row_count(path)
    col_dtypes = {c: str(t) for c, t in df.dtypes.items()}

    return {
        "folder": folder_label,
        "filename": path.name,
        "format": path.suffix.upper().lstrip("."),
        "size_mb": size_mb,
        "full_rows": full_rows,
        "columns": len(df.columns),
        "column_names": " | ".join(df.columns.tolist()),
        "dtypes": " | ".join(f"{c}:{t}" for c, t in col_dtypes.items()),
        "timestamp_col": ts_col or "None detected",
        "ts_format": "ISO 8601 / inferred",
        "date_start": date_start,
        "date_end": date_end,
        "missing_pct": _missing_pct(df),
        "duplicate_rows": _dup_count(df),
        "notes": "",
    }


def inspect_all() -> list[dict]:
    records = []
    for folder_name, label in DATA_FOLDERS.items():
        folder_path = BASE_DIR / folder_name
        if not folder_path.exists():
            log.warning("Folder not found: %s", folder_path)
            continue
        supported = {".csv", ".xlsx", ".xls", ".parquet", ".json"}
        files = [
            p
            for p in sorted(folder_path.iterdir())
            if p.is_file() and p.suffix.lower() in supported
        ]
        if not files:
            log.warning("No supported files in %s", folder_name)
        for fp in files:
            records.append(inspect_file(fp, label))
    return records


# ── Report generators ──────────────────────────────────────────────────────────
def save_inventory_csv(records: list[dict]) -> Path:
    out = OUTPUT_DIR / "dataset_inventory.csv"
    if records:
        pd.DataFrame(records).to_csv(out, index=False)
    log.info("Saved inventory CSV → %s", out)
    return out


def save_inventory_md(records: list[dict]) -> Path:
    out = OUTPUT_DIR / "dataset_inventory.md"
    lines = [
        "# Dataset Inventory",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
    ]
    for r in records:
        lines += [
            f"## {r['folder']} / {r['filename']}",
            "",
            f"| Property | Value |",
            f"|---|---|",
            f"| Format | {r['format']} |",
            f"| Size | {r['size_mb']} MB |",
            f"| Rows | {r['full_rows']:,} |",
            f"| Columns | {r['columns']} |",
            f"| Timestamp Column | {r['timestamp_col']} |",
            f"| Date Range | {r['date_start']} → {r['date_end']} |",
            f"| Missing % | {r['missing_pct']}% |",
            f"| Duplicate Rows | {r['duplicate_rows']} |",
            "",
            f"**Columns:** `{r['column_names']}`",
            "",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved inventory MD → %s", out)
    return out


def save_quality_report(records: list[dict]) -> Path:
    out = OUTPUT_DIR / "data_quality_report.md"
    lines = [
        "# Data Quality Report",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Summary",
        "",
    ]
    for r in records:
        mp = r["missing_pct"]
        dup = r["duplicate_rows"]
        if mp == "N/A":
            quality = "Unknown (load error)"
        elif float(mp) > 30:
            quality = "⚠️ HIGH missing data"
        elif float(mp) > 5:
            quality = "⚠️ Moderate missing data"
        else:
            quality = "✅ Low missing data"

        lines += [
            f"### {r['folder']} / {r['filename']}",
            "",
            f"- **Quality:** {quality}",
            f"- **Missing values:** {mp}% (of first 50k rows sampled)",
            f"- **Duplicate rows:** {dup}",
            f"- **Date range:** {r['date_start']} → {r['date_end']}",
            "",
        ]

    lines += [
        "## Integration Notes",
        "",
        "- **Solar Generation** (Plant 1 & 2): India, May–June 2020, 15-min resolution.",
        "- **Carbon Emissions** (df_fuel_ckan): UK National Grid, 30-min, 2009–present.",
        "  Contains half-hourly CARBON_INTENSITY (gCO₂/kWh) — suitable for carbon calc.",
        "- **Electricity Demand** (time_series_*): Europe ENTSOE, hourly, 2015–present.",
        "  Contains actual & forecast loads, solar/wind generation, day-ahead prices per country.",
        "- **Weather** (weather_data): Europe, hourly, 1980–2019.",
        "  Country-level temperature + radiation (direct/diffuse horizontal) in W/m².",
        "",
        "### Geographic/Temporal Alignment",
        "- Solar plant data (India) does NOT align geographically with European weather/demand.",
        "- Carbon intensity (UK) is used as a general proxy; labeled clearly.",
        "- European electricity demand and weather datasets share hourly UTC timestamps.",
        "  → These two CAN be merged for European demand + weather analysis.",
        "- Solar plant weather sensor data IS co-located with generation data.",
        "  → Plant weather sensors are used for solar forecasting features.",
    ]

    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved quality report → %s", out)
    return out


def print_summary(records: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 70)
    for r in records:
        print(f"\n📁 {r['folder']} / {r['filename']}")
        print(f"   Format  : {r['format']}  |  Size: {r['size_mb']} MB")
        rows_str = f"{r['full_rows']:,}" if isinstance(r["full_rows"], int) and r["full_rows"] >= 0 else "?"
        print(f"   Rows    : {rows_str}  |  Columns: {r['columns']}")
        print(f"   Date    : {r['date_start']} → {r['date_end']}")
        print(f"   Missing : {r['missing_pct']}%  |  Duplicates: {r['duplicate_rows']}")
        cols_preview = r["column_names"].split(" | ")[:6]
        print(f"   Cols    : {', '.join(cols_preview)}" + (" ..." if len(r["column_names"].split(" | ")) > 6 else ""))
    print("\n" + "=" * 70 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────
def run() -> list[dict]:
    log.info("Starting dataset discovery from base: %s", BASE_DIR)
    records = inspect_all()
    save_inventory_csv(records)
    save_inventory_md(records)
    save_quality_report(records)
    print_summary(records)
    log.info("Discovery complete. %d files inspected.", len(records))
    # Note: files were saved even if summary print had unicode errors
    return records


if __name__ == "__main__":
    run()
