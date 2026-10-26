import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed"
FORECASTS = BASE_DIR / "outputs" / "forecasts"
sys.path.insert(0, str(BASE_DIR))

COLORS = {
    "solar": "#f59e0b",
    "grid": "#ef4444",
    "battery": "#8b5cf6",
    "demand": "#3b82f6",
    "carbon": "#10b981",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", family="Inter"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    legend=dict(bgcolor="#0f172a", bordercolor="#334155"),
    margin=dict(l=20, r=20, t=40, b=20),
)


@st.cache_data(ttl=300)
def load_solar_data(plant: int = 1) -> pd.DataFrame | None:
    path = PROCESSED / f"solar_generation_plant{plant}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    return df


@st.cache_data(ttl=300)
def load_carbon_data() -> pd.DataFrame | None:
    path = PROCESSED / "carbon_emissions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    return df


def metric_card(label: str, value: str, delta: str = "", positive: bool = True):
    delta_class = "metric-positive" if positive else "metric-negative"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render():
    st.markdown('<h1 class="section-header">⚡ Energy Overview Dashboard</h1>', unsafe_allow_html=True)

    solar_df = load_solar_data(1)
    carbon_df = load_carbon_data()

    # ── KPI Row ───────────────────────────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        if solar_df is not None:
            total_gen = solar_df["AC_POWER"].sum() * 0.25 / 1000  # kWh (15-min * kW)
            metric_card("Total Solar Gen (Plant 1)", f"{total_gen:,.0f} kWh", "✓ Measured")
        else:
            metric_card("Total Solar Gen", "Not Available", "Run preprocessing first")

    with cols[1]:
        if solar_df is not None:
            peak = solar_df["AC_POWER"].max()
            metric_card("Peak Solar Output", f"{peak:,.1f} kW", "Plant 1")
        else:
            metric_card("Peak Solar Output", "—")

    with cols[2]:
        if carbon_df is not None:
            avg_ci = carbon_df["CARBON_INTENSITY"].mean()
            metric_card("Avg Carbon Intensity", f"{avg_ci:.0f} gCO₂/kWh", "UK Grid (proxy)")
        else:
            metric_card("Carbon Intensity", "—")

    with cols[3]:
        if solar_df is not None:
            avg_pwr = solar_df["AC_POWER"].mean()
            metric_card("Avg Solar Power", f"{avg_pwr:.1f} kW", "Daytime average")
        else:
            metric_card("Avg Power", "—")

    st.markdown("---")

    # ── Solar Generation Time Series ─────────────────────────────────────────
    if solar_df is not None:
        st.subheader("☀️ Solar Generation – Plant 1")
        plant_select = st.radio("Plant", [1, 2], horizontal=True, key="ov_plant")
        df_sel = load_solar_data(plant_select)
        if df_sel is not None:
            # Downsample to hourly for viz performance
            df_plot = df_sel.set_index("DATE_TIME").resample("1h").mean().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot["DATE_TIME"], y=df_plot["AC_POWER"],
                name="AC Power (kW)",
                line=dict(color=COLORS["solar"], width=2),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Hourly AC Power Output")
            fig.update_yaxes(title="kW")
            st.plotly_chart(fig, use_container_width=True)

            # Daily totals
            daily = df_sel.copy()
            daily["date"] = daily["DATE_TIME"].dt.date
            daily_total = daily.groupby("date")["AC_POWER"].sum() * 0.25 / 1000  # kWh
            fig2 = go.Figure(go.Bar(
                x=daily_total.index, y=daily_total.values,
                marker_color=COLORS["solar"], name="Daily Yield (kWh)",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, title="Daily Solar Yield")
            fig2.update_yaxes(title="kWh")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("⚠️ Preprocessed solar data not found. Run: `python -m src.data.preprocessing`")

    st.markdown("---")

    # ── Carbon Intensity ──────────────────────────────────────────────────────
    if carbon_df is not None:
        st.subheader("🌿 UK Grid Carbon Intensity")
        st.markdown("""
        <div class="assumption-badge">
            ⚠️ ASSUMPTION: UK National Grid carbon intensity used as scenario proxy.
            Not representative of India or Europe without documented adjustment.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        # Sample last 7 days worth of data
        last_days = carbon_df.tail(7 * 48)  # 48 half-hourly per day
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=last_days["DATETIME"], y=last_days["CARBON_INTENSITY"],
            name="Carbon Intensity (gCO₂/kWh)",
            line=dict(color=COLORS["carbon"], width=1.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, title="Carbon Intensity (UK Grid) – Last 7 Days in Dataset")
        fig3.update_yaxes(title="gCO₂/kWh")
        st.plotly_chart(fig3, use_container_width=True)

        # Fuel mix
        fuel_cols = ["GAS", "COAL", "NUCLEAR", "WIND", "HYDRO", "BIOMASS", "SOLAR"]
        avail_fuels = [c for c in fuel_cols if c in carbon_df.columns]
        if avail_fuels:
            last_row = carbon_df[avail_fuels].dropna().iloc[-1]
            total = last_row.sum()
            pcts = (last_row / total * 100).round(1)
            fig4 = go.Figure(go.Pie(
                labels=pcts.index, values=pcts.values,
                hole=0.5,
                marker=dict(colors=["#6366f1", "#374151", "#a78bfa", "#34d399", "#60a5fa", "#a3e635", "#f59e0b"]),
            ))
            fig4.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                               font=dict(color="#e2e8f0", family="Inter"),
                               title="UK Grid Fuel Mix (Latest Snapshot)",
                               margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("ℹ️ Carbon data not preprocessed yet. Run preprocessing first.")

    st.markdown("---")
    st.caption("""
    **Data Sources:**
    - Solar Generation: Kaggle Solar Power Generation Data (India, 2020)
    - Carbon Intensity: UK National Grid ESO (df_fuel_ckan.csv)
    - Electricity Demand: Open Power System Data (ENTSOE, Europe)
    - Weather: Open Power System Data (NASA MERRA-2, Europe 1980–2019)
    """)
