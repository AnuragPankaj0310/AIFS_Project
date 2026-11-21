"""
Page 5: Energy Optimization
Two tabs:
  1. LP Battery & Flexible-Load Optimizer (original)
  2. AC Set-Point Optimizer — paper method (Houchati, Beitelmal & Khraisheh, ASME 2022)
"""
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
FORECASTS = BASE_DIR / "outputs" / "forecasts"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", family="Inter"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    legend=dict(bgcolor="#0f172a", bordercolor="#334155"),
    margin=dict(l=20, r=20, t=40, b=20),
)


def render():
    st.markdown('<h1 class="section-header">🔋 Energy Optimization</h1>', unsafe_allow_html=True)

    tab_lp, tab_sp = st.tabs([
        "⚡ LP Battery & Load Optimization",
        "🌡️ AC Set-Point Optimizer (Paper Method)",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — LP optimizer (unchanged from original)
    # ════════════════════════════════════════════════════════════════════════
    with tab_lp:
        st.markdown("""
        Configure the building energy system parameters and run the demand response optimization.
        The LP optimizer schedules battery storage and flexible loads to minimize cost,
        carbon emissions, and peak grid demand.
        """)

        # ── Config panel ──────────────────────────────────────────────────
        with st.expander("⚙️ System Configuration", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🔋 Battery Storage**")
                bat_cap = st.slider("Capacity (kWh)", 10.0, 500.0, 100.0, key="bat_cap")
                bat_init = st.slider("Initial SoC (kWh)", 0.0, bat_cap, bat_cap * 0.5, key="bat_init")
                bat_max_c = st.slider("Max Charge Rate (kW)", 5.0, 100.0, 30.0, key="bat_mc")
                bat_max_d = st.slider("Max Discharge Rate (kW)", 5.0, 100.0, 30.0, key="bat_md")
                charge_eff = st.slider("Charge Efficiency", 0.8, 1.0, 0.95, step=0.01, key="c_eff")
                discharge_eff = st.slider("Discharge Efficiency", 0.8, 1.0, 0.95, step=0.01, key="d_eff")

            with col2:
                st.markdown("**⚡ Flexible Load**")
                flex_energy = st.slider("Total Flexible Energy (kWh)", 1.0, 100.0, 20.0, key="flex_e")
                flex_max_kw = st.slider("Max Flex Power (kW)", 1.0, 50.0, 10.0, key="flex_kw")
                flex_start = st.slider("Window Start (hour)", 0, 22, 6, key="flex_s")
                flex_end = st.slider("Window End (hour)", flex_start + 1, 23, 22, key="flex_en")
                grid_limit = st.slider("Grid Import Limit (kW)", 50.0, 500.0, 200.0, key="grid_lim")

            with col3:
                st.markdown("**💰 Tariff & Carbon**")
                tariff_type = st.radio("Tariff Type", ["Flat Rate", "Time-of-Use"], key="tariff_type")
                flat_rate = st.slider("Rate (€/kWh)", 0.10, 0.50, 0.25, step=0.01, key="flat_rate")
                if tariff_type == "Time-of-Use":
                    peak_rate = st.slider("Peak Rate (€/kWh, 7–21h)", flat_rate, 0.60, 0.35, step=0.01, key="peak_rate")
                else:
                    peak_rate = flat_rate

                ci_assumption = st.number_input("Carbon Intensity (gCO₂/kWh)", 50.0, 900.0, 280.0, step=10.0, key="ci_val")
                st.caption("ℹ️ UK avg ~280, DE avg ~350, FR avg ~80 (nuclear-heavy)")

                st.markdown("**🎯 Objective Weights**")
                cost_w = st.slider("Cost Weight", 0.0, 2.0, 1.0, step=0.1, key="cost_w")
                carbon_w = st.slider("Carbon Weight", 0.0, 2.0, 0.5, step=0.1, key="carbon_w")
                peak_w = st.slider("Peak Demand Weight", 0.0, 2.0, 0.3, step=0.1, key="peak_w")

        # ── Run optimization ──────────────────────────────────────────────
        run_btn = st.button("🚀 Run Optimization", type="primary", key="run_opt")

        if run_btn:
            from src.optimization.optimizer import (
                OptimizationConfig, OptimizationInputs,
                run_baseline, run_optimization, compare_scenarios,
            )

            T = 24
            h = np.arange(T)

            if tariff_type == "Time-of-Use":
                tariff_vec = np.where((h >= 7) & (h <= 21), peak_rate, flat_rate)
            else:
                tariff_vec = np.full(T, flat_rate)

            PROCESSED = BASE_DIR / "data" / "processed"
            solar = None
            for plant in [1, 2]:
                fp = PROCESSED / f"solar_generation_plant{plant}.csv"
                if fp.exists():
                    df = pd.read_csv(fp)
                    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
                    day = df["DATE_TIME"].dt.date.dropna().iloc[len(df) // 2]
                    day_df = df[df["DATE_TIME"].dt.date == day].head(T)
                    if len(day_df) >= T:
                        s = day_df["AC_POWER"].values[:T].astype(float)
                        if s.max() > 1000:
                            s = s / s.max() * 80
                        elif s.max() > 0:
                            s = s / s.max() * 60
                        solar = s
                        st.info(f"ℹ️ Using real Plant {plant} solar data (scaled to building level).")
                        break

            if solar is None:
                solar = 60 * np.exp(-0.5 * ((h - 12) / 3) ** 2)
                solar[:6] = 0; solar[20:] = 0
                st.warning("⚠️ Using SIMULATED solar profile (preprocessed data not found).")

            load = np.array([15, 13, 12, 12, 13, 15, 18, 25, 30, 32,
                             30, 28, 27, 28, 30, 32, 35, 40, 38, 35,
                             30, 25, 20, 17], dtype=float)[:T]

            cfg = OptimizationConfig(
                battery_capacity_kwh=bat_cap,
                initial_soc_kwh=bat_init,
                max_charge_kw=bat_max_c,
                max_discharge_kw=bat_max_d,
                charge_efficiency=charge_eff,
                discharge_efficiency=discharge_eff,
                total_flex_energy_kwh=flex_energy,
                max_flex_kw=flex_max_kw,
                flex_window_start=flex_start,
                flex_window_end=flex_end,
                grid_import_limit_kw=grid_limit,
                flat_tariff_eur_per_kwh=flat_rate,
                cost_weight=cost_w,
                carbon_weight_obj=carbon_w,
                peak_weight=peak_w,
                dt_hours=1.0,
            )

            inputs = OptimizationInputs(
                solar_generation_kw=solar,
                building_load_kw=load,
                tariff_eur_per_kwh=tariff_vec,
                carbon_intensity_gco2_kwh=np.full(T, ci_assumption),
                hours=h,
            )

            with st.spinner("⚙️ Running LP optimization (CBC solver)..."):
                baseline = run_baseline(inputs, cfg)
                optimized = run_optimization(inputs, cfg)
                comparison = compare_scenarios(baseline, optimized)

            st.session_state["baseline"] = baseline
            st.session_state["optimized"] = optimized
            st.session_state["comparison"] = comparison
            st.session_state["solar"] = solar
            st.session_state["load"] = load

        # ── Display results ───────────────────────────────────────────────
        if "baseline" in st.session_state and "optimized" in st.session_state:
            baseline = st.session_state["baseline"]
            optimized = st.session_state["optimized"]
            comparison = st.session_state["comparison"]
            solar = st.session_state.get("solar", np.zeros(24))
            load = st.session_state.get("load", np.zeros(24))

            if optimized.status in ["Optimal", "optimal", "1"]:
                st.success(f"✅ Optimization Status: {optimized.status}")
            else:
                st.warning(f"⚠️ Status: {optimized.status} — showing baseline as fallback")

            st.subheader("💡 Scenario Comparison")
            kpi_cols = st.columns(3)
            for i, row in comparison.iterrows():
                with kpi_cols[i]:
                    pct = row["Saving_%"]
                    color = "metric-positive" if pct > 0 else "metric-negative"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.8rem;color:#94a3b8;text-transform:uppercase">{row['Metric']}</div>
                        <div style="margin-top:8px">
                            <span style="color:#94a3b8;font-size:0.9rem">Baseline: </span>
                            <span style="color:#e2e8f0;font-weight:600">{row['Baseline']:.2f}</span>
                        </div>
                        <div>
                            <span style="color:#94a3b8;font-size:0.9rem">Optimized: </span>
                            <span style="color:#38bdf8;font-weight:600">{row['Optimized']:.2f}</span>
                        </div>
                        <div class="{color}" style="margin-top:6px;font-size:0.9rem">
                            {'↓' if pct > 0 else '↑'} {abs(pct):.1f}% {'saving' if pct > 0 else 'increase'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("")

            st.subheader("⚡ Grid Import: Baseline vs Optimized")
            h = np.arange(24)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Baseline Grid Import", x=h, y=baseline.grid_import,
                                 marker_color="rgba(239,68,68,0.7)"))
            fig.add_trace(go.Bar(name="Optimized Grid Import", x=h, y=optimized.grid_import,
                                 marker_color="rgba(59,130,246,0.7)"))
            fig.add_trace(go.Scatter(name="Solar Available", x=h, y=solar,
                                     line=dict(color="#f59e0b", width=2, dash="dot")))
            fig.add_trace(go.Scatter(name="Building Load", x=h, y=load,
                                     line=dict(color="#94a3b8", width=2)))
            fig.update_layout(**PLOTLY_LAYOUT, barmode="group", title="Hourly Grid Import Comparison")
            fig.update_xaxes(title="Hour of Day")
            fig.update_yaxes(title="kW")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔋 Battery State of Charge")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=h, y=optimized.soc, name="SoC (kWh)",
                                      line=dict(color="#8b5cf6", width=2.5),
                                      fill="tozeroy", fillcolor="rgba(139,92,246,0.2)"))
            fig2.add_trace(go.Bar(x=h, y=optimized.batt_charge,
                                  name="Charging (kW)", marker_color="rgba(74,222,128,0.6)"))
            fig2.add_trace(go.Bar(x=h, y=-optimized.batt_discharge,
                                  name="Discharging (kW)", marker_color="rgba(248,113,113,0.6)"))
            fig2.update_layout(**PLOTLY_LAYOUT, barmode="relative", title="Battery Schedule (Optimized)")
            fig2.update_xaxes(title="Hour of Day")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("🔄 Flexible Load Schedule")
            fig3 = go.Figure(go.Bar(x=h, y=optimized.flex_load,
                                    marker_color="#34d399", name="Flex Load (kW)"))
            fig3.update_layout(**PLOTLY_LAYOUT, title="Optimized Flexible Load Schedule")
            fig3.update_xaxes(title="Hour of Day")
            fig3.update_yaxes(title="kW")
            st.plotly_chart(fig3, use_container_width=True)

            combined = pd.concat([baseline.to_dataframe().assign(scenario="baseline"),
                                   optimized.to_dataframe().assign(scenario="optimized")])
            csv = combined.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Optimization Results", csv, "optimization_results.csv")

            st.markdown("""
            <div class="assumption-badge">
                ⚠️ NOTE: Building load profile is representative/simulated. Solar data is scaled
                from actual plant measurements where available. Results are scenario-based, not
                measurements of a specific building.
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — AC Set-Point Optimizer (Houchati et al., ASME 2022)
    # ════════════════════════════════════════════════════════════════════════
    with tab_sp:
        st.markdown("""
        **Based on:** Houchati, Beitelmal & Khraisheh (2022),
        *Predictive Modeling for Rooftop Solar Energy Throughput*,
        ASME J. Energy Resour. Technol., 144(1).

        **Method:** When forecast AC load exceeds forecast PV output, the indoor AC set point is raised
        within a user-defined comfort ceiling. This reduces elastic AC demand and cuts grid draw — no
        battery storage required. The load–temperature relationship is modelled with a linearised
        steady-state heat balance (P_AC = k·ΔT/R + b), as in the paper.
        """)

        with st.expander("🌡️ Set-Point Parameters", expanded=True):
            sp_col1, sp_col2 = st.columns(2)
            with sp_col1:
                comfort_ceiling = st.slider("Comfort Ceiling (°C)", 23.0, 28.0, 24.0, step=0.5, key="sp_ceil")
                initial_sp = st.slider("Initial Set Point (°C)", 18.0, comfort_ceiling, 23.0, step=0.5, key="sp_init")
                sp_step = st.slider("Set-Point Step per Action (°C)", 0.1, 1.0, 0.5, step=0.1, key="sp_step")
                max_raise = st.slider("Max Set-Point Raise (°C)", 0.5, 5.0, 2.0, step=0.5, key="sp_raise")
            with sp_col2:
                k_coeff = st.slider("Thermal Conductance k", 0.1, 2.0, 0.5, step=0.1, key="sp_k")
                R_resist = st.slider("Thermal Resistance R (°C/kW)", 0.5, 5.0, 2.0, step=0.5, key="sp_R")
                b_base = st.slider("Baseline AC Power b (kW)", 0.0, 5.0, 1.0, step=0.5, key="sp_b")
                st.caption("P_AC = k·(T_out − T_in) / R + b   [kW]")

        run_sp = st.button("🚀 Run Set-Point Optimization", type="primary", key="run_sp")

        if run_sp:
            from src.optimization.setpoint_optimizer import (
                SetpointOptimizerConfig, run_setpoint_optimization,
            )

            T_sp = 24
            h_sp = np.arange(T_sp)

            # Simulated sunny-day PV (bell curve, as in paper testbed)
            pv_sp = 60 * np.exp(-0.5 * ((h_sp - 12) / 3) ** 2)
            pv_sp[:6] = 0.0; pv_sp[20:] = 0.0

            # AC-dominated building load (paper scale ~10 kWh/h peak)
            load_sp = np.array([5, 4, 4, 4, 5, 6, 8, 10, 10, 9,
                                 9, 8, 8, 9, 10, 10, 9,  8,  7, 6,
                                 6, 5, 5, 5], dtype=float)

            # Qatar-like outdoor temperature profile (July)
            t_out_sp = np.array([30, 29, 28, 28, 29, 31, 34, 37, 40, 42,
                                  43, 44, 44, 43, 42, 41, 39, 37, 35, 33,
                                  32, 31, 31, 30], dtype=float)

            cfg_sp = SetpointOptimizerConfig(
                comfort_ceiling=comfort_ceiling,
                initial_setpoint=initial_sp,
                setpoint_step=sp_step,
                max_setpoint_raise=max_raise,
            )

            with st.spinner("Running set-point optimization..."):
                sp_result = run_setpoint_optimization(
                    pv_sp, load_sp, t_out_sp, cfg_sp,
                    k=k_coeff, R=R_resist, b_power=b_base,
                )
            summary = sp_result.summary()

            # ── KPIs ─────────────────────────────────────────────────────
            st.subheader("📊 Results")
            kc1, kc2, kc3, kc4 = st.columns(4)
            with kc1:
                st.metric("Load Reduction",
                          f"{summary['total_load_reduction_kwh']:.2f} kWh",
                          f"-{summary['pct_load_reduction']:.1f}%")
            with kc2:
                st.metric("Grid Draw (Baseline)",
                          f"{summary['total_grid_draw_baseline_kwh']:.2f} kWh")
            with kc3:
                st.metric("Grid Draw (Optimized)",
                          f"{summary['total_grid_draw_optimized_kwh']:.2f} kWh")
            with kc4:
                st.metric("Grid Draw Reduction",
                          f"{summary['pct_grid_reduction']:.1f}%")

            # ── Load vs PV chart ──────────────────────────────────────────
            st.subheader("🌡️ AC Load vs PV — Before & After")
            sp_df = sp_result.to_dataframe()

            fig_sp1 = go.Figure()
            fig_sp1.add_trace(go.Scatter(
                x=sp_df["hour"], y=sp_df["pv_forecast_kw"],
                name="PV Forecast (kW)", line=dict(color="#f59e0b", width=2)))
            fig_sp1.add_trace(go.Scatter(
                x=sp_df["hour"], y=sp_df["load_forecast_kw"],
                name="AC Load — Baseline (kW)", line=dict(color="#f87171", width=2)))
            fig_sp1.add_trace(go.Scatter(
                x=sp_df["hour"],
                y=sp_df["load_forecast_kw"] - sp_df["load_reduction_kw"],
                name="AC Load — After Opt. (kW)", line=dict(color="#34d399", width=2, dash="dot")))
            fig_sp1.add_trace(go.Bar(
                x=sp_df["hour"], y=sp_df["load_reduction_kw"],
                name="Load Reduction (kW)", marker_color="rgba(52,211,153,0.35)"))
            fig_sp1.update_layout(**PLOTLY_LAYOUT, barmode="overlay",
                                  title="PV vs AC Load: Before & After Set-Point Optimization")
            fig_sp1.update_xaxes(title="Hour of Day")
            fig_sp1.update_yaxes(title="kW")
            st.plotly_chart(fig_sp1, use_container_width=True)

            # ── Set-point vs outdoor temp ─────────────────────────────────
            st.subheader("🌡️ Set-Point & Temperature Schedule")
            fig_sp2 = go.Figure()
            fig_sp2.add_trace(go.Scatter(
                x=sp_df["hour"], y=sp_df["ac_setpoint_c"],
                name="Indoor Set Point (°C)", line=dict(color="#fb923c", width=2.5)))
            fig_sp2.add_trace(go.Scatter(
                x=sp_df["hour"], y=sp_df["t_outdoor_c"],
                name="Outdoor Temp (°C)", line=dict(color="#94a3b8", width=1.5, dash="dot")))
            fig_sp2.add_hline(
                y=comfort_ceiling, line_dash="dash", line_color="#f87171",
                annotation_text=f"Comfort ceiling {comfort_ceiling}°C",
                annotation_font_color="#f87171")
            fig_sp2.update_layout(**PLOTLY_LAYOUT,
                                  title="AC Set-Point vs Outdoor Temperature")
            fig_sp2.update_xaxes(title="Hour of Day")
            fig_sp2.update_yaxes(title="°C")
            st.plotly_chart(fig_sp2, use_container_width=True)

            csv_sp = sp_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Set-Point Schedule CSV", csv_sp, "setpoint_schedule.csv")

            st.markdown("""
            <div class="assumption-badge">
            ℹ️ <b>Paper method (Houchati et al., ASME 2022):</b>
            Set point raised when load &gt; PV <i>and</i> outdoor &gt; indoor <i>and</i>
            indoor &lt; comfort ceiling.
            Thermal model: P<sub>AC</sub> = k·(T<sub>out</sub> − T<sub>in</sub>) / R + b.
            Inputs are representative/simulated (Qatar sunny-day profile from the paper testbed).
            </div>
            """, unsafe_allow_html=True)
