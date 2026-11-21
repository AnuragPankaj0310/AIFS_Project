"""
AC Set-Point Optimizer
=======================
Replicates the demand-response control strategy from:
  Houchati, Beitelmal & Khraisheh (2022)
  "Predictive Modeling for Rooftop Solar Energy Throughput"
  ASME J. Energy Resour. Technol., 144(1).

Method
------
Given hourly SVR forecasts of:
  - PV output  (kW)
  - Building load / AC demand  (kW)

The optimizer raises the indoor AC set point whenever:
  1. Forecast load > forecast PV  (shortfall from grid)
  2. Indoor temperature is below outdoor (cooling needed)
  3. Indoor temperature is below the comfort ceiling (e.g. 24°C)

Raising the set point reduces AC power consumption (elastic load).
The relationship between AC power and ΔT is learned by a third SVR
(or approximated here with a linear thermal model: PAC = k·ΔT/R + b).

Outputs
-------
  - Per-hour AC set point (°C)
  - Per-hour AC power after optimization  (kW)
  - Hourly load reduction  (kW)
  - Summary statistics

Paper constants (Table 1 defaults):
  comfort_ceiling = 24°C
  initial set point = 23°C
  Gaussian SVR kernel for all three SVR models
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("setpoint_optimizer")


# ── Thermal model ─────────────────────────────────────────────────────────────
def ac_power_from_setpoint(
    t_outdoor: np.ndarray,
    t_setpoint: np.ndarray,
    k: float = 0.5,       # kW / (°C · 1/R)  — building thermal conductance
    R: float = 2.0,       # °C/kW thermal resistance
    b: float = 1.0,       # kW base load (fans, pumps)
) -> np.ndarray:
    """
    Steady-state AC power model (linearised from paper eq.):
        P_AC = k · (T_out − T_in) / R  +  b   [kW]
    Clipped to ≥ 0 (no AC power when T_out ≤ T_in).

    Parameters
    ----------
    t_outdoor   : hourly outdoor temperature array (°C)
    t_setpoint  : hourly indoor set-point temperature array (°C)
    k           : thermal conductivity coefficient
    R           : building thermal resistance (°C/kW)
    b           : baseline AC power (fans, pumps) (kW)
    """
    delta_T = t_outdoor - t_setpoint
    p_ac = k * delta_T / R + b
    return np.clip(p_ac, 0.0, None)


# ── Set-point control rule ────────────────────────────────────────────────────
@dataclass
class SetpointOptimizerConfig:
    """Configuration for the AC set-point optimizer."""
    comfort_ceiling: float = 24.0    # °C  — maximum indoor temperature (paper: 24°C)
    initial_setpoint: float = 23.0   # °C  — starting set point
    setpoint_step: float = 0.5       # °C  — increment per control action
    max_setpoint_raise: float = 2.0  # °C  — maximum total raise from initial
    dt_hours: float = 1.0            # hours per interval


@dataclass
class SetpointResult:
    """Result of one set-point optimization run."""
    hour: np.ndarray
    pv_forecast: np.ndarray           # kW
    load_forecast: np.ndarray         # kW (before optimization)
    t_outdoor: np.ndarray             # °C
    setpoint: np.ndarray              # °C (optimized)
    ac_power_baseline: np.ndarray     # kW (original set point throughout)
    ac_power_optimized: np.ndarray    # kW (after set-point shift)
    load_reduction: np.ndarray        # kW saved per hour
    shortfall: np.ndarray             # kW = max(0, load − pv)

    @property
    def total_load_reduction_kwh(self) -> float:
        return float(np.sum(self.load_reduction))

    @property
    def pct_load_reduction(self) -> float:
        base = float(np.sum(self.ac_power_baseline))
        return (self.total_load_reduction_kwh / base * 100) if base > 0 else 0.0

    @property
    def total_grid_draw_baseline(self) -> float:
        return float(np.sum(np.maximum(0, self.load_forecast - self.pv_forecast)))

    @property
    def total_grid_draw_optimized(self) -> float:
        opt_load = self.load_forecast - self.load_reduction
        return float(np.sum(np.maximum(0, opt_load - self.pv_forecast)))

    @property
    def pct_grid_reduction(self) -> float:
        base = self.total_grid_draw_baseline
        opt = self.total_grid_draw_optimized
        return ((base - opt) / base * 100) if base > 0 else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "hour": self.hour,
            "pv_forecast_kw": self.pv_forecast,
            "load_forecast_kw": self.load_forecast,
            "t_outdoor_c": self.t_outdoor,
            "ac_setpoint_c": self.setpoint,
            "ac_power_baseline_kw": self.ac_power_baseline,
            "ac_power_optimized_kw": self.ac_power_optimized,
            "load_reduction_kw": self.load_reduction,
            "shortfall_kw": self.shortfall,
        })

    def summary(self) -> dict:
        return {
            "total_load_reduction_kwh": round(self.total_load_reduction_kwh, 2),
            "pct_load_reduction": round(self.pct_load_reduction, 1),
            "total_grid_draw_baseline_kwh": round(self.total_grid_draw_baseline, 2),
            "total_grid_draw_optimized_kwh": round(self.total_grid_draw_optimized, 2),
            "pct_grid_reduction": round(self.pct_grid_reduction, 1),
        }


def run_setpoint_optimization(
    pv_forecast: np.ndarray,
    load_forecast: np.ndarray,
    t_outdoor: np.ndarray,
    cfg: SetpointOptimizerConfig | None = None,
    k: float = 0.5,
    R: float = 2.0,
    b_power: float = 1.0,
) -> SetpointResult:
    """
    Paper control algorithm (Slide 8 / Fig. in paper):

    For each hour t:
      If load[t] > pv[t]:                     # shortfall → act
        If t_outdoor[t] > t_in[t]:             # cooling needed
          If t_in[t] < comfort_ceiling:        # comfort not violated
            Raise set point by one step
            Re-evaluate AC power
            Repeat until shortfall gone or ceiling hit

    Parameters
    ----------
    pv_forecast   : hourly PV generation forecast (kW)
    load_forecast : hourly building load forecast (kW) — assumed to be AC-dominated
    t_outdoor     : hourly outdoor temperature (°C)
    cfg           : SetpointOptimizerConfig
    k, R, b_power : thermal model coefficients

    Returns
    -------
    SetpointResult with per-hour schedule.
    """
    if cfg is None:
        cfg = SetpointOptimizerConfig()

    T = len(pv_forecast)
    setpoint = np.full(T, cfg.initial_setpoint)
    ac_baseline = ac_power_from_setpoint(t_outdoor, np.full(T, cfg.initial_setpoint), k, R, b_power)
    ac_optimized = ac_baseline.copy()
    load_reduction = np.zeros(T)
    shortfall = np.maximum(0.0, load_forecast - pv_forecast)

    for t in range(T):
        if shortfall[t] <= 0:
            continue  # No shortfall — no action needed

        t_in = setpoint[t]
        t_out = t_outdoor[t]
        cur_power = ac_optimized[t]

        # Paper control loop: raise set point while constraints allow
        while (
            load_forecast[t] - load_reduction[t] > pv_forecast[t]  # still a shortfall
            and t_out > t_in                                          # cooling still needed
            and t_in < cfg.comfort_ceiling                            # comfort not violated
            and setpoint[t] - cfg.initial_setpoint < cfg.max_setpoint_raise
        ):
            t_in = min(t_in + cfg.setpoint_step, cfg.comfort_ceiling)
            setpoint[t] = t_in
            new_power = ac_power_from_setpoint(
                np.array([t_out]), np.array([t_in]), k, R, b_power
            )[0]
            load_reduction[t] = cur_power - new_power
            ac_optimized[t] = new_power

    return SetpointResult(
        hour=np.arange(T),
        pv_forecast=pv_forecast,
        load_forecast=load_forecast,
        t_outdoor=t_outdoor,
        setpoint=setpoint,
        ac_power_baseline=ac_baseline,
        ac_power_optimized=ac_optimized,
        load_reduction=load_reduction,
        shortfall=shortfall,
    )


# ── Convenience demo ──────────────────────────────────────────────────────────
def run_demo_setpoint(cfg: SetpointOptimizerConfig | None = None) -> dict:
    """
    Run a 24-hour demo with synthetic but representative inputs.
    Uses the same sunny-day solar profile as the paper testbed (Qatar, July).
    """
    from pathlib import Path
    import json

    if cfg is None:
        cfg = SetpointOptimizerConfig()

    T = 24
    h = np.arange(T)

    # Simulated solar PV (bell-curve, sunny day)
    pv = 60 * np.exp(-0.5 * ((h - 12) / 3) ** 2)
    pv[:6] = 0.0
    pv[20:] = 0.0

    # Simulated AC-dominated building load (10 kWh/h peak, as in paper)
    load = np.array([
        5, 4, 4, 4, 5, 6, 8, 10, 10, 9,
        9, 8, 8, 9, 10, 10, 9, 8, 7, 6,
        6, 5, 5, 5,
    ], dtype=float)

    # Outdoor temperature: Qatar-like July profile (°C)
    t_outdoor = np.array([
        30, 29, 28, 28, 29, 31, 34, 37, 40, 42,
        43, 44, 44, 43, 42, 41, 39, 37, 35, 33,
        32, 31, 31, 30,
    ], dtype=float)

    result = run_setpoint_optimization(pv, load, t_outdoor, cfg)
    summary = result.summary()

    print("\n" + "=" * 60)
    print("SET-POINT OPTIMIZATION  (Houchati et al. Method)")
    print("=" * 60)
    print(f"  Total load reduction : {summary['total_load_reduction_kwh']:.2f} kWh")
    print(f"  Load reduction       : {summary['pct_load_reduction']:.1f}%")
    print(f"  Grid draw (baseline) : {summary['total_grid_draw_baseline_kwh']:.2f} kWh")
    print(f"  Grid draw (optimized): {summary['total_grid_draw_optimized_kwh']:.2f} kWh")
    print(f"  Grid draw reduction  : {summary['pct_grid_reduction']:.1f}%")
    print("=" * 60)

    # Save schedule
    out_path = Path(__file__).resolve().parents[2] / "outputs" / "forecasts" / "setpoint_schedule.csv"
    result.to_dataframe().to_csv(out_path, index=False)
    log.info("Set-point schedule saved → %s", out_path)

    return {"result": result, "summary": summary}


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_demo_setpoint()
