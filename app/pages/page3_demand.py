"""
Page 3: Demand Forecasting
"""
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed"
FORECASTS = BASE_DIR / "outputs" / "forecasts"
METRICS = BASE_DIR / "outputs" / "metrics"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", family="Inter"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    legend=dict(bgcolor="#0f172a", bordercolor="#334155"),
    margin=dict(l=20, r=20, t=40, b=20),
)

MODEL_COLORS = {
    "actual": "#3b82f6",
    "persistence": "#94a3b8",
    "linear_ridge": "#60a5fa",
    "random_forest": "#4ade80",
    "xgboost": "#c084fc",
    "forecast_entsoe": "#f59e0b",
}


@st.cache_data(ttl=300)
def load_preds(country: str) -> pd.DataFrame | None:
    path = FORECASTS / f"demand_{country}_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_metrics(country: str) -> pd.DataFrame | None:
    path = METRICS / f"demand_{country}_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_demand_processed(country: str) -> pd.DataFrame | None:
    path = PROCESSED / f"electricity_demand_{country}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def render():
    st.markdown('<h1 class="section-header">⚡ Electricity Demand Forecasting</h1>', unsafe_allow_html=True)

    country = st.selectbox("Country", ["DE", "GB", "FR", "ES", "IT"], key="demand_country")

    preds = load_preds(country)
    metrics = load_metrics(country)
    raw = load_demand_processed(country)

    if raw is not None:
        st.subheader("📊 Historical Demand Overview")

        # Find load column
        load_col = None
        for candidate in [f"load_actual_entsoe_transparency", "load_actual", "load"]:
            if candidate in raw.columns:
                load_col = candidate
                break
        if load_col is None:
            candidates = [c for c in raw.columns if "load" in c.lower() and "actual" in c.lower()]
            if candidates:
                load_col = candidates[0]

        if load_col:
            # Downsample for display
            df_plot = raw[["timestamp", load_col]].dropna().tail(8760)  # ~1 year
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot["timestamp"], y=df_plot[load_col],
                name=f"{country} Load (MW)",
                line=dict(color="#3b82f6", width=1),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
            ))

            # Add official ENTSOE forecast if available
            fc_col = "load_forecast_entsoe_transparency"
            if fc_col in raw.columns:
                df_fc = raw[["timestamp", fc_col]].dropna().tail(8760)
                fig.add_trace(go.Scatter(
                    x=df_fc["timestamp"], y=df_fc[fc_col],
                    name="ENTSOE Forecast (MW)",
                    line=dict(color="#f59e0b", width=1, dash="dot"),
                ))

            fig.update_layout(**PLOTLY_LAYOUT, title=f"{country} Electricity Demand (Last Year in Dataset)")
            fig.update_yaxes(title="MW")
            st.plotly_chart(fig, use_container_width=True)

            # Hourly pattern
            st.subheader("🕐 Average Hourly Demand Pattern")
            raw_copy = raw[["timestamp", load_col]].copy().dropna()
            raw_copy["hour"] = raw_copy["timestamp"].dt.hour
            raw_copy["dow"] = raw_copy["timestamp"].dt.day_name()
            hourly_avg = raw_copy.groupby("hour")[load_col].mean()

            fig2 = go.Figure(go.Scatter(
                x=hourly_avg.index, y=hourly_avg.values,
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2.5),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
                name="Avg Hourly Load",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, title="Average Hourly Demand Profile")
            fig2.update_xaxes(title="Hour of Day", tickvals=list(range(0, 24, 2)))
            fig2.update_yaxes(title="Average MW")
            st.plotly_chart(fig2, use_container_width=True)

        # Solar / Wind generation available?
        solar_col = [c for c in raw.columns if "solar_generation" in c.lower()]
        if solar_col:
            st.subheader("☀️ Solar Generation in Demand Dataset")
            df_s = raw[["timestamp", solar_col[0]]].dropna().tail(8760)
            fig3 = go.Figure(go.Scatter(
                x=df_s["timestamp"], y=df_s[solar_col[0]],
                line=dict(color="#f59e0b", width=1),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
                name="Solar Generation (MW)",
            ))
            fig3.update_layout(**PLOTLY_LAYOUT, title=f"{country} Solar Generation (ENTSOE)")
            fig3.update_yaxes(title="MW")
            st.plotly_chart(fig3, use_container_width=True)

    # ── Model Forecasts ───────────────────────────────────────────────────────
    if preds is None or metrics is None:
        st.warning("⚠️ Demand models not trained yet.")
        st.info("Run: `python -m src.models.forecasting`")
        if st.button("🚀 Train Demand Models", key="train_demand"):
            with st.spinner("Training... (may take several minutes for large dataset)"):
                try:
                    from src.models.forecasting import train_demand_forecasting
                    train_demand_forecasting(country=country)
                    st.success("✅ Done!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        return

    st.subheader("🤖 Model Performance")
    display_cols = [c for c in ["model", "mae", "rmse", "r2", "nmae"] if c in metrics.columns]
    st.dataframe(metrics[display_cols], use_container_width=True)

    # Chart
    models_avail = [c for c in MODEL_COLORS if c in preds.columns and c != "actual"]
    n_show = st.slider("Test rows to display", 50, min(500, len(preds)), 168, key="dem_horizon")
    plot_df = preds.tail(n_show).reset_index(drop=True)

    fig4 = go.Figure()
    if "actual" in plot_df.columns:
        fig4.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["actual"],
            name="Actual", line=dict(color=MODEL_COLORS["actual"], width=2.5),
        ))
    for m in models_avail:
        if m in plot_df.columns:
            fig4.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[m],
                name=m.replace("_", " ").title(),
                line=dict(color=MODEL_COLORS.get(m, "#ffffff"), dash="dot"),
            ))
    fig4.update_layout(**PLOTLY_LAYOUT, title=f"Demand Forecast – {country} Test Set")
    fig4.update_yaxes(title="MW")
    st.plotly_chart(fig4, use_container_width=True)

    csv = preds.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions CSV", csv, f"demand_{country}_predictions.csv")
