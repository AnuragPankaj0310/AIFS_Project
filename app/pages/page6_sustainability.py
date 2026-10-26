"""
Page 6: Sustainability Report
"""
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
REPORTS_DIR = BASE_DIR / "outputs" / "reports"
FORECASTS_DIR = BASE_DIR / "outputs" / "forecasts"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", family="Inter"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    legend=dict(bgcolor="#0f172a", bordercolor="#334155"),
    margin=dict(l=20, r=20, t=50, b=20),
)


def render():
    st.markdown('<h1 class="section-header">🌱 Sustainability Report</h1>', unsafe_allow_html=True)

    # Check for optimization results
    sched_path = FORECASTS_DIR / "optimization_schedules.csv"
    comp_path = FORECASTS_DIR / "optimization_comparison.csv"

    if not sched_path.exists() or not comp_path.exists():
        st.warning("⚠️ No optimization results found.")
        st.info("""
        **To generate sustainability results:**
        1. Go to **Energy Optimization** page
        2. Click **Run Optimization**
        3. Return here to view the full report

        Or run from command line:
        ```bash
        python -m src.optimization.optimizer
        ```
        """)
        if st.button("🚀 Run Demo Optimization", key="run_demo"):
            with st.spinner("Running optimization..."):
                try:
                    from src.optimization.optimizer import run_demo_optimization
                    run_demo_optimization()
                    st.success("Done!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        return

    # Load data
    schedules = pd.read_csv(sched_path)
    comparison = pd.read_csv(comp_path)

    baseline = schedules[schedules["scenario"] == "baseline"]
    optimized = schedules[schedules["scenario"] == "optimized"]

    # ── KPI Summary ────────────────────────────────────────────────────────────
    st.subheader("📊 Key Sustainability Indicators")
    cols = st.columns(4)

    for i, row in comparison.iterrows():
        with cols[i % 4]:
            metric = row["Metric"]
            base_val = row["Baseline"]
            opt_val = row["Optimized"]
            saving = row["Saving"]
            pct = row["Saving_%"]
            icon = "↓" if pct > 0 else "↑"
            color = "#4ade80" if pct > 0 else "#f87171"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem;color:#94a3b8;text-transform:uppercase;margin-bottom:6px">{metric}</div>
                <div style="color:#94a3b8;font-size:0.85rem">Base: <b style="color:#e2e8f0">{base_val:.2f}</b></div>
                <div style="color:#94a3b8;font-size:0.85rem">Opt: <b style="color:#38bdf8">{opt_val:.2f}</b></div>
                <div style="color:{color};font-size:1rem;font-weight:700;margin-top:6px">
                    {icon} {abs(pct):.1f}% {'saved' if pct > 0 else 'increase'}
                </div>
                <div style="color:#64748b;font-size:0.75rem">Saving: {saving:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Grid import comparison chart ──────────────────────────────────────────
    st.subheader("⚡ Grid Import Profile")
    if "grid_import_kw" in baseline.columns and "grid_import_kw" in optimized.columns:
        hours = range(len(baseline))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(hours), y=baseline["grid_import_kw"].values,
            name="Baseline Grid Import",
            line=dict(color="#ef4444", width=2),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        ))
        fig.add_trace(go.Scatter(
            x=list(hours), y=optimized["grid_import_kw"].values,
            name="Optimized Grid Import",
            line=dict(color="#3b82f6", width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
        ))
        if "solar_gen_kw" in baseline.columns:
            fig.add_trace(go.Scatter(
                x=list(hours), y=baseline["solar_gen_kw"].values,
                name="Solar Generation",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Baseline vs Optimized Grid Import")
        fig.update_xaxes(title="Hour of Day")
        fig.update_yaxes(title="kW")
        st.plotly_chart(fig, use_container_width=True)

    # ── Carbon Emissions Over Time ────────────────────────────────────────────
    st.subheader("🌿 Estimated Carbon Emissions")
    ci_assumed = 280.0  # gCO2/kWh (documented assumption)
    if "grid_import_kw" in baseline.columns:
        base_co2 = (baseline["grid_import_kw"].values * ci_assumed / 1000)  # kg CO2 per hour
        opt_co2 = (optimized["grid_import_kw"].values * ci_assumed / 1000)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Baseline CO₂", x=list(range(len(base_co2))), y=base_co2,
            marker_color="rgba(239,68,68,0.7)",
        ))
        fig2.add_trace(go.Bar(
            name="Optimized CO₂", x=list(range(len(opt_co2))), y=opt_co2,
            marker_color="rgba(74,222,128,0.7)",
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, barmode="group",
                           title=f"Hourly Carbon Emissions (CI={ci_assumed} gCO₂/kWh assumed)")
        fig2.update_xaxes(title="Hour")
        fig2.update_yaxes(title="kg CO₂")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Solar Self-Consumption ────────────────────────────────────────────────
    st.subheader("☀️ Solar Energy Utilization")
    if "solar_used_kw" in baseline.columns and "solar_gen_kw" in baseline.columns:
        base_solar_used = baseline["solar_used_kw"].sum()
        opt_solar_used = optimized["solar_used_kw"].sum()
        solar_avail = baseline["solar_gen_kw"].sum()

        base_ratio = (base_solar_used / solar_avail * 100) if solar_avail > 0 else 0
        opt_ratio = (opt_solar_used / solar_avail * 100) if solar_avail > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Solar Available (kWh)", f"{solar_avail:.1f}")
        with c2:
            st.metric("Baseline Self-Consumption", f"{base_ratio:.1f}%",
                      delta=None)
        with c3:
            st.metric("Optimized Self-Consumption", f"{opt_ratio:.1f}%",
                      delta=f"+{opt_ratio - base_ratio:.1f}%")

        # Donut chart
        fig3 = go.Figure()
        for label, used, color in [
            ("Baseline", base_solar_used, "#ef4444"),
            ("Optimized", opt_solar_used, "#4ade80"),
        ]:
            fig3.add_trace(go.Pie(
                labels=["Solar Used On-site", "Exported/Curtailed"],
                values=[used, max(0, solar_avail - used)],
                name=label,
                hole=0.5,
                domain={"x": [0, 0.45] if label == "Baseline" else [0.55, 1]},
                marker=dict(colors=[color, "#1e293b"]),
                title=dict(text=label, font=dict(color="#e2e8f0")),
            ))
        fig3.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                           font=dict(color="#e2e8f0"),
                           title="Solar Self-Consumption Rate",
                           margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Assumptions ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚠️ Assumptions & Limitations")
    st.markdown("""
    <div class="assumption-badge" style="display:block;margin-bottom:1rem">
        <b>This analysis uses documented scenario assumptions. Results are NOT measurements of a real building.</b>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View Full Assumptions"):
        st.markdown("""
        | Assumption | Value | Source |
        |---|---|---|
        | Carbon Intensity | 280 gCO₂/kWh (default) | UK National Grid average (proxy) |
        | Building Load | Representative profile (15–40 kW peak) | Synthetic residential/commercial |
        | Solar Generation | Scaled from Plant 1/2 measurements OR synthetic bell-curve | India solar plant (2020) |
        | Electricity Tariff | Configurable flat/TOU rate | User-defined scenario |
        | Battery Specs | Configurable capacity and efficiency | User-defined scenario |

        **Data geographic alignment:**
        - Solar plant data (India) is NOT directly compatible with European demand/weather data.
        - Carbon intensity (UK) is used as a scenario proxy, not the grid where solar plants operate.
        - European demand and weather data are aligned by UTC timestamp for Germany (DE).

        **What this IS:**
        - A working demand response optimization demonstrating the methodology.
        - Quantitative comparison of baseline vs optimized scenarios under specified assumptions.

        **What this is NOT:**
        - A measurement of an actual building's energy use.
        - A prediction of real-world savings for any specific facility.
        """)

    # ── Download Reports ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Download Reports")

    col1, col2, col3 = st.columns(3)
    with col1:
        csv_comp = comparison.to_csv(index=False).encode("utf-8")
        st.download_button("Comparison Summary (CSV)", csv_comp, "sustainability_comparison.csv", "text/csv")
    with col2:
        csv_sched = schedules.to_csv(index=False).encode("utf-8")
        st.download_button("Full Schedule (CSV)", csv_sched, "optimization_schedules.csv", "text/csv")
    with col3:
        # Build text report
        lines = ["# Sustainability Report\n\n## Comparison\n"]
        lines.append(comparison.to_markdown(index=False))
        lines.append("\n\n## Assumptions\n- Carbon intensity: 280 gCO2/kWh (UK proxy)\n")
        report_text = "\n".join(lines).encode("utf-8")
        st.download_button("Full Report (Markdown)", report_text, "sustainability_report.md", "text/markdown")
