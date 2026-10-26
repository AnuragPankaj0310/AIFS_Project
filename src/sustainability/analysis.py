"""
Sustainability Analysis Module
================================
Calculates carbon emissions, electricity costs, and renewable utilization
for baseline and optimized scenarios.

All assumptions are documented. Simulated inputs are labeled clearly.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("sustainability")

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_electricity_cost(
    grid_import_kw: np.ndarray,
    tariff_eur_per_kwh: np.ndarray,
    dt_hours: float = 1.0,
) -> float:
    """
    Total electricity cost (€).

    Formula: Cost = Σ_t [grid_import(t) × tariff(t) × dt]
    Units:   kW × EUR/kWh × h = EUR
    """
    return float(np.sum(grid_import_kw * tariff_eur_per_kwh * dt_hours))


def calculate_carbon_emissions(
    grid_import_kw: np.ndarray,
    carbon_intensity_gco2_kwh: np.ndarray,
    dt_hours: float = 1.0,
) -> dict:
    """
    Grid-related carbon emissions.

    Formula: Emissions = Σ_t [grid_import(t) × carbon_intensity(t) × dt]
    Units:   kW × gCO₂/kWh × h = gCO₂ → converted to kg

    Returns dict with grams and kilograms.
    """
    total_g = float(np.sum(grid_import_kw * carbon_intensity_gco2_kwh * dt_hours))
    return {"gco2": total_g, "kgco2": total_g / 1000}


def calculate_peak_demand(grid_import_kw: np.ndarray) -> float:
    """Maximum grid import over the evaluation period (kW)."""
    return float(np.max(grid_import_kw))


def calculate_renewable_utilization(
    solar_used_kw: np.ndarray,
    solar_generation_kw: np.ndarray,
    dt_hours: float = 1.0,
) -> dict:
    """
    Fraction of available solar generation used on-site.

    solar_self_consumption_ratio = Σ solar_used / Σ solar_available
    solar_self_sufficiency_ratio = Σ solar_used / Σ building_load  (not calculated here)
    """
    total_solar = float(np.sum(solar_generation_kw * dt_hours))
    total_used = float(np.sum(solar_used_kw * dt_hours))
    ratio = total_used / total_solar if total_solar > 0 else 0.0
    return {
        "solar_available_kwh": round(total_solar, 2),
        "solar_used_kwh": round(total_used, 2),
        "self_consumption_ratio": round(ratio, 4),
        "self_consumption_pct": round(ratio * 100, 1),
    }


def compare_sustainability(
    baseline_grid: np.ndarray,
    optimized_grid: np.ndarray,
    baseline_solar_used: np.ndarray,
    optimized_solar_used: np.ndarray,
    solar_generation: np.ndarray,
    tariff: np.ndarray,
    carbon_intensity: np.ndarray,
    dt_hours: float = 1.0,
    label_baseline: str = "Baseline",
    label_optimized: str = "Optimized",
) -> pd.DataFrame:
    """
    Compute and compare sustainability metrics for two scenarios.
    Returns a DataFrame with absolute differences and percentage savings.
    """
    base_cost = calculate_electricity_cost(baseline_grid, tariff, dt_hours)
    opt_cost = calculate_electricity_cost(optimized_grid, tariff, dt_hours)
    base_co2 = calculate_carbon_emissions(baseline_grid, carbon_intensity, dt_hours)
    opt_co2 = calculate_carbon_emissions(optimized_grid, carbon_intensity, dt_hours)
    base_peak = calculate_peak_demand(baseline_grid)
    opt_peak = calculate_peak_demand(optimized_grid)
    base_re = calculate_renewable_utilization(baseline_solar_used, solar_generation, dt_hours)
    opt_re = calculate_renewable_utilization(optimized_solar_used, solar_generation, dt_hours)

    rows = []

    def _row(metric, base_val, opt_val, unit=""):
        saving = base_val - opt_val
        pct = (saving / base_val * 100) if base_val != 0 else 0.0
        rows.append({
            "Metric": metric,
            "Unit": unit,
            label_baseline: round(base_val, 3),
            label_optimized: round(opt_val, 3),
            "Saving": round(saving, 3),
            "Saving_%": round(pct, 1),
        })

    _row("Electricity Cost", base_cost, opt_cost, "EUR")
    _row("Carbon Emissions", base_co2["kgco2"], opt_co2["kgco2"], "kg CO₂")
    _row("Peak Grid Demand", base_peak, opt_peak, "kW")
    _row("Solar Self-Consumption",
         base_re["solar_used_kwh"], opt_re["solar_used_kwh"], "kWh")

    df = pd.DataFrame(rows)
    out = REPORTS_DIR / "sustainability_comparison.csv"
    df.to_csv(out, index=False)
    log.info("Sustainability comparison saved → %s", out)
    return df


def generate_sustainability_report(
    comparison_df: pd.DataFrame,
    assumptions: dict,
    output_path: Path | None = None,
) -> str:
    """Generate a Markdown sustainability report."""
    if output_path is None:
        output_path = REPORTS_DIR / "sustainability_report.md"

    lines = [
        "# Sustainability Analysis Report",
        "",
        "## Results Summary",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## Assumptions & Limitations",
        "",
    ]
    for key, val in assumptions.items():
        lines.append(f"- **{key}**: {val}")

    lines += [
        "",
        "## Methodology",
        "",
        "### Electricity Cost",
        "```",
        "Total Cost = Σ_t [ grid_import(t) × tariff(t) × Δt ]",
        "Units: kW × EUR/kWh × h = EUR",
        "```",
        "",
        "### Carbon Emissions",
        "```",
        "Emissions = Σ_t [ grid_import(t) × carbon_intensity(t) × Δt ]",
        "Units: kW × gCO₂/kWh × h = gCO₂ (÷ 1000 = kg CO₂)",
        "```",
        "",
        "### Carbon Intensity Source",
        "- UK National Grid half-hourly data (df_fuel_ckan.csv).",
        "- Used as proxy carbon intensity; NOT representative of any specific European grid.",
        "- For accurate results, replace with country-specific grid carbon intensity data.",
        "",
        "### Renewable Utilization",
        "```",
        "Self-Consumption Ratio = Σ solar_used / Σ solar_available",
        "```",
    ]

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    log.info("Sustainability report saved → %s", output_path)
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Demo
    T = 24
    solar = 60 * np.exp(-0.5 * ((np.arange(T) - 12) / 3) ** 2)
    solar[:6] = 0; solar[20:] = 0
    load = np.ones(T) * 30
    tariff = np.where(np.arange(T) >= 7, 0.30, 0.15)
    ci = np.full(T, 280.0)

    base_grid = np.maximum(0, load - solar)
    opt_grid = base_grid * 0.75  # simulated 25% reduction

    df = compare_sustainability(
        baseline_grid=base_grid,
        optimized_grid=opt_grid,
        baseline_solar_used=np.minimum(solar, load),
        optimized_solar_used=np.minimum(solar, load),
        solar_generation=solar,
        tariff=tariff,
        carbon_intensity=ci,
    )
    print(df.to_string())

    generate_sustainability_report(df, {
        "Carbon intensity source": "UK National Grid (proxy)",
        "Tariff": "Simulated time-of-use",
        "Solar data": "Simulated bell-curve profile",
    })
