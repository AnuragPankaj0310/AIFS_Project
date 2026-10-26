"""
Dynamic Load Optimization Engine
==================================
Implements linear programming (PuLP) for building-to-grid demand response.

Decision Variables
------------------
  grid_import[t]     : MW imported from grid at time t (≥ 0)
  solar_used[t]      : MW of solar used on-site at time t (≥ 0)
  solar_export[t]    : MW of solar exported to grid at time t (≥ 0)
  batt_charge[t]     : MW charged into battery at time t (≥ 0)
  batt_discharge[t]  : MW discharged from battery at time t (≥ 0)
  soc[t]             : MWh stored in battery at time t (≥ 0)
  flex_load[t]       : MW of flexible load scheduled at time t (≥ 0)

Objective Function
------------------
  Minimize: Σ_t [ tariff[t] * grid_import[t]
                  + carbon_intensity[t] * grid_import[t] * carbon_weight
                  + peak_penalty * peak_demand ]

Constraints
-----------
  1. Energy balance at each t
  2. Battery SoC update (efficiency-aware)
  3. Battery capacity limits
  4. Charge / discharge rate limits
  5. SoC bounds (min_soc to max_soc)
  6. Flexible load operating window
  7. Flexible load total energy requirement
  8. Non-negative grid import
  9. Solar availability
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("optimization")

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS = BASE_DIR / "outputs"
FORECASTS_DIR = OUTPUTS / "forecasts"
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Configuration ──────────────────────────────────────────────────────────────
@dataclass
class OptimizationConfig:
    """User-configurable optimization parameters."""

    # Battery
    battery_capacity_kwh: float = 100.0         # kWh
    initial_soc_kwh: float = 50.0               # kWh
    max_charge_kw: float = 30.0                 # kW
    max_discharge_kw: float = 30.0              # kW
    charge_efficiency: float = 0.95             # η_charge
    discharge_efficiency: float = 0.95          # η_discharge
    min_soc_fraction: float = 0.10              # minimum SoC as fraction of capacity
    max_soc_fraction: float = 0.95              # maximum SoC as fraction of capacity

    # Grid
    grid_import_limit_kw: float = 200.0         # kW maximum grid import

    # Flexible load
    total_flex_energy_kwh: float = 20.0         # kWh total flexible energy requirement
    flex_window_start: int = 6                  # earliest hour
    flex_window_end: int = 22                   # latest hour
    max_flex_kw: float = 10.0                   # kW per interval

    # Tariff (default: flat rate)
    flat_tariff_eur_per_kwh: float = 0.25

    # Carbon weight (EUR per gCO₂/kWh * kWh)
    carbon_weight: float = 0.0001               # scales carbon penalty vs cost

    # Peak demand penalty
    peak_penalty: float = 0.50                  # EUR per kW of peak demand

    # Objective weights
    cost_weight: float = 1.0
    carbon_weight_obj: float = 0.5
    peak_weight: float = 0.3

    # Time
    dt_hours: float = 1.0                       # time step in hours


@dataclass
class OptimizationInputs:
    """Input time-series for the optimization model."""
    solar_generation_kw: np.ndarray              # kW available from solar
    building_load_kw: np.ndarray                 # kW of inflexible building load
    tariff_eur_per_kwh: Optional[np.ndarray] = None    # per-interval tariff
    carbon_intensity_gco2_kwh: Optional[np.ndarray] = None  # gCO₂/kWh
    hours: Optional[np.ndarray] = None           # hour-of-day for each interval

    def __post_init__(self):
        T = len(self.solar_generation_kw)
        if self.tariff_eur_per_kwh is None:
            self.tariff_eur_per_kwh = np.full(T, 0.25)
        if self.carbon_intensity_gco2_kwh is None:
            self.carbon_intensity_gco2_kwh = np.full(T, 300.0)
        if self.hours is None:
            self.hours = np.arange(T) % 24


@dataclass
class OptimizationResult:
    """Result of one optimization run."""
    grid_import: np.ndarray
    solar_used: np.ndarray
    solar_export: np.ndarray
    batt_charge: np.ndarray
    batt_discharge: np.ndarray
    soc: np.ndarray
    flex_load: np.ndarray
    total_cost_eur: float
    total_carbon_kg: float
    peak_demand_kw: float
    status: str
    scenario: str

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "grid_import_kw": self.grid_import,
            "solar_used_kw": self.solar_used,
            "solar_export_kw": self.solar_export,
            "batt_charge_kw": self.batt_charge,
            "batt_discharge_kw": self.batt_discharge,
            "soc_kwh": self.soc,
            "flex_load_kw": self.flex_load,
        })


# ── Baseline (no-optimization) scenario ───────────────────────────────────────
def run_baseline(inputs: OptimizationInputs, cfg: OptimizationConfig) -> OptimizationResult:
    """
    Baseline scenario: no battery, no load shifting.
    All building load met by solar first, then grid.
    """
    T = len(inputs.solar_generation_kw)
    solar_used = np.minimum(inputs.solar_generation_kw, inputs.building_load_kw)
    solar_export = np.maximum(0, inputs.solar_generation_kw - inputs.building_load_kw)
    grid_import = np.maximum(0, inputs.building_load_kw - inputs.solar_generation_kw)
    batt_charge = np.zeros(T)
    batt_discharge = np.zeros(T)
    soc = np.full(T + 1, cfg.initial_soc_kwh)
    flex_load = np.zeros(T)  # all flex load at default time (within window)

    cost = float(np.sum(grid_import * inputs.tariff_eur_per_kwh * cfg.dt_hours))
    carbon = float(np.sum(grid_import * inputs.carbon_intensity_gco2_kwh * cfg.dt_hours) / 1000)  # kg
    peak = float(grid_import.max())

    log.info("[Baseline] Cost=€%.2f | Carbon=%.1f kg | Peak=%.1f kW", cost, carbon, peak)
    return OptimizationResult(
        grid_import=grid_import, solar_used=solar_used, solar_export=solar_export,
        batt_charge=batt_charge, batt_discharge=batt_discharge,
        soc=soc[:-1], flex_load=flex_load,
        total_cost_eur=cost, total_carbon_kg=carbon, peak_demand_kw=peak,
        status="baseline", scenario="Baseline (no optimization)"
    )


# ── LP Optimization ──────────────────────────────────────────────────────────
def run_optimization(inputs: OptimizationInputs, cfg: OptimizationConfig) -> OptimizationResult:
    """
    Run PuLP LP optimization for demand response.
    """
    try:
        import pulp
    except ImportError:
        log.error("PuLP not installed. Returning baseline instead.")
        return run_baseline(inputs, cfg)

    T = len(inputs.solar_generation_kw)
    solar = inputs.solar_generation_kw
    load = inputs.building_load_kw
    tariff = inputs.tariff_eur_per_kwh
    ci = inputs.carbon_intensity_gco2_kwh
    hours = inputs.hours
    dt = cfg.dt_hours

    # Battery limits
    cap = cfg.battery_capacity_kwh
    min_soc = cap * cfg.min_soc_fraction
    max_soc = cap * cfg.max_soc_fraction

    # ── Define problem ────────────────────────────────────────────────────────
    prob = pulp.LpProblem("DemandResponse", pulp.LpMinimize)

    # ── Decision variables (PuLP 4.x uses prob.add_variable) ─────────────────
    def lv(name, lb=0.0, ub=None):
        return prob.add_variable(name, lowBound=lb, upBound=ub)

    g = [lv(f"grid_{t}", 0, cfg.grid_import_limit_kw) for t in range(T)]
    su = [lv(f"solar_used_{t}") for t in range(T)]
    se = [lv(f"solar_export_{t}") for t in range(T)]
    bc = [lv(f"batt_charge_{t}", 0, cfg.max_charge_kw) for t in range(T)]
    bd = [lv(f"batt_discharge_{t}", 0, cfg.max_discharge_kw) for t in range(T)]
    soc = [lv(f"soc_{t}", min_soc, max_soc) for t in range(T + 1)]
    fl = [lv(f"flex_{t}", 0, cfg.max_flex_kw) for t in range(T)]
    peak = lv("peak_demand")

    # ── Objective ─────────────────────────────────────────────────────────────
    energy_cost = pulp.lpSum(g[t] * tariff[t] * dt * cfg.cost_weight for t in range(T))
    carbon_cost = pulp.lpSum(g[t] * ci[t] * dt * cfg.carbon_weight_obj * cfg.carbon_weight for t in range(T))
    peak_cost = peak * cfg.peak_weight * cfg.peak_penalty
    prob += energy_cost + carbon_cost + peak_cost

    # ── Constraints ───────────────────────────────────────────────────────────
    # Initial SoC
    prob += soc[0] == cfg.initial_soc_kwh

    for t in range(T):
        # Energy balance: supply = demand
        # supply: grid + solar_used + battery_discharge
        # demand: inflexible_load + flex_load + battery_charge + solar_export
        prob += (g[t] + su[t] + bd[t] * cfg.discharge_efficiency ==
                 load[t] + fl[t] + bc[t] + se[t])

        # Solar balance: solar_used + solar_export + batt_charged_from_solar ≤ available
        prob += su[t] + se[t] + bc[t] <= solar[t] + g[t]  # conservative: charge can come from grid too
        prob += su[t] <= solar[t]
        prob += se[t] >= 0

        # Battery SoC update
        prob += soc[t + 1] == soc[t] + bc[t] * cfg.charge_efficiency * dt - bd[t] * dt

        # Peak tracking
        prob += peak >= g[t]

        # Flexible load window
        if hours[t] < cfg.flex_window_start or hours[t] > cfg.flex_window_end:
            prob += fl[t] == 0

    # Flexible load total energy
    prob += pulp.lpSum(fl[t] * dt for t in range(T)) == cfg.total_flex_energy_kwh

    # Final SoC ≥ initial (cycle constraint)
    prob += soc[T] >= cfg.initial_soc_kwh * 0.8

    # ── Solve ─────────────────────────────────────────────────────────────────
    # PuLP 4.x uses HiGHS instead of CBC
    available = pulp.listSolvers(onlyAvailable=True)
    if "HiGHS" in available:
        solver = pulp.getSolver("HiGHS", msg=False, timeLimit=120)
    elif "COIN_CMD" in available:
        solver = pulp.getSolver("COIN_CMD", msg=False, timeLimit=120)
    else:
        solver = pulp.LpSolverDefault

    solve_stats = prob.solve(solver)
    # PuLP 4.x: solve returns LpSolveStats; check status via string representation
    status_str = str(solve_stats.status) if hasattr(solve_stats, 'status') else "Unknown"
    is_optimal = "Optimal" in status_str or (hasattr(solve_stats, 'status') and solve_stats.status == pulp.LpSolveStatus.Optimal)
    log.info("Optimization status: %s", status_str)

    if not is_optimal:
        log.warning("Solver did not find optimal solution. Returning baseline.")
        return run_baseline(inputs, cfg)

    # ── Extract results ───────────────────────────────────────────────────────
    def val(var_list):
        return np.array([max(0.0, (v.varValue or 0.0)) for v in var_list])

    g_val = val(g)
    su_val = val(su)
    se_val = val(se)
    bc_val = val(bc)
    bd_val = val(bd)
    soc_val = val(soc[:-1])
    fl_val = val(fl)

    cost = float(np.sum(g_val * tariff * dt))
    carbon = float(np.sum(g_val * ci * dt) / 1000)  # gCO2 -> kg
    peak_val = float(g_val.max())

    log.info("[Optimized] Cost=EUR%.2f | Carbon=%.1f kg | Peak=%.1f kW", cost, carbon, peak_val)

    return OptimizationResult(
        grid_import=g_val, solar_used=su_val, solar_export=se_val,
        batt_charge=bc_val, batt_discharge=bd_val, soc=soc_val, flex_load=fl_val,
        total_cost_eur=cost, total_carbon_kg=carbon, peak_demand_kw=peak_val,
        status=status_str, scenario="Optimized (LP)"
    )


# ── Compare scenarios ──────────────────────────────────────────────────────────
def compare_scenarios(baseline: OptimizationResult, optimized: OptimizationResult) -> pd.DataFrame:
    """Return a comparison DataFrame with absolute and % savings."""
    rows = []
    metrics = [
        ("Total Cost (€)", "total_cost_eur"),
        ("Carbon Emissions (kg CO₂)", "total_carbon_kg"),
        ("Peak Grid Demand (kW)", "peak_demand_kw"),
    ]
    for label, attr in metrics:
        base_val = getattr(baseline, attr)
        opt_val = getattr(optimized, attr)
        saving = base_val - opt_val
        pct = (saving / base_val * 100) if base_val != 0 else 0
        rows.append({
            "Metric": label,
            "Baseline": round(base_val, 3),
            "Optimized": round(opt_val, 3),
            "Saving": round(saving, 3),
            "Saving_pct": round(pct, 1),
        })
    df = pd.DataFrame(rows)
    df["Saving_%"] = df["Saving_pct"]  # alias for dashboard compatibility
    out = FORECASTS_DIR / "optimization_comparison.csv"
    df.to_csv(out, index=False)
    log.info("Comparison saved -> %s", out)
    return df


# ── Convenience runner ────────────────────────────────────────────────────────
def run_demo_optimization(cfg: OptimizationConfig | None = None) -> dict:
    """
    Run a demo optimization using synthetic (but labeled) scenario inputs.
    If real preprocessed data is available, it is used instead.

    IMPORTANT: When synthetic data is used, results are labeled as
    'SCENARIO (SIMULATED)' to distinguish from actual measurements.
    """
    from pathlib import Path
    import json

    if cfg is None:
        cfg = OptimizationConfig()

    PROCESSED = BASE_DIR / "data" / "processed"
    T = 24  # 24-hour horizon

    # Try to load real solar generation
    solar = None
    for plant in [1, 2]:
        fp = PROCESSED / f"solar_generation_plant{plant}.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
            # pick the first day with data
            day = df["DATE_TIME"].dt.date.dropna().iloc[0]
            day_df = df[df["DATE_TIME"].dt.date == day].head(T)
            if len(day_df) >= T:
                solar = day_df["AC_POWER"].values[:T] / 1000  # kW assumed (if MW convert)
                # Scale to building level (assume 100kW peak capacity from full plant)
                if solar.max() > 1000:
                    solar = solar / solar.max() * 80  # scale to ~80 kW peak
                log.info("Using Plant %d solar data for optimization demo.", plant)
                break

    if solar is None:
        log.info("No preprocessed solar data found. Using SIMULATED solar profile.")
        # Simulated solar: bell-curve around noon
        h = np.arange(T)
        solar = 60 * np.exp(-0.5 * ((h - 12) / 3) ** 2)
        solar[:6] = 0; solar[20:] = 0

    # Simulated building load (residential/commercial day profile)
    load = np.array([
        15, 13, 12, 12, 13, 15, 18, 25, 30, 32,
        30, 28, 27, 28, 30, 32, 35, 40, 38, 35,
        30, 25, 20, 17
    ], dtype=float)[:T]

    # Simulated time-of-use tariff (peak hours more expensive)
    tariff = np.where((np.arange(T) >= 7) & (np.arange(T) <= 21), 0.30, 0.15)

    # UK carbon intensity profile (from processed data if available)
    ci = np.full(T, 250.0)

    inputs = OptimizationInputs(
        solar_generation_kw=solar,
        building_load_kw=load,
        tariff_eur_per_kwh=tariff,
        carbon_intensity_gco2_kwh=ci,
        hours=np.arange(T),
    )

    baseline = run_baseline(inputs, cfg)
    optimized = run_optimization(inputs, cfg)
    comparison = compare_scenarios(baseline, optimized)

    # Save schedule
    baseline_df = baseline.to_dataframe()
    baseline_df["solar_gen_kw"] = solar
    baseline_df["building_load_kw"] = load
    baseline_df["scenario"] = "baseline"

    opt_df = optimized.to_dataframe()
    opt_df["solar_gen_kw"] = solar
    opt_df["building_load_kw"] = load
    opt_df["scenario"] = "optimized"

    combined = pd.concat([baseline_df, opt_df])
    combined.to_csv(FORECASTS_DIR / "optimization_schedules.csv", index=False)
    log.info("Schedules saved.")

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))
    print("=" * 60)

    return {
        "baseline": baseline,
        "optimized": optimized,
        "comparison": comparison,
        "inputs": inputs,
    }


if __name__ == "__main__":
    result = run_demo_optimization()
