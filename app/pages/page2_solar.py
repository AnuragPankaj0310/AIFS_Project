"""
Page 2: Solar Forecasting
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
MODELS_DIR = BASE_DIR / "models"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", family="Inter"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    legend=dict(bgcolor="#0f172a", bordercolor="#334155"),
    margin=dict(l=20, r=20, t=40, b=20),
)

MODEL_COLORS = {
    "actual": "#f59e0b",
    "persistence": "#94a3b8",
    "linear_ridge": "#60a5fa",
    "random_forest": "#4ade80",
    "xgboost": "#c084fc",
    "lstm": "#f472b6",
}


@st.cache_data(ttl=300)
def load_predictions(plant: int) -> pd.DataFrame | None:
    path = FORECASTS / f"solar_plant{plant}_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_metrics(plant: int) -> pd.DataFrame | None:
    path = METRICS / f"solar_plant{plant}_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def render():
    st.markdown('<h1 class="section-header">☀️ Solar Power Forecasting</h1>', unsafe_allow_html=True)

    plant = st.selectbox("Select Plant", [1, 2], key="solar_plant_sel")

    preds = load_predictions(plant)
    metrics = load_metrics(plant)

    # ── Check if trained ──────────────────────────────────────────────────────
    if preds is None or metrics is None:
        st.warning("⚠️ Models not trained yet.")
        st.info("""
        **To train solar forecasting models, run:**
        ```bash
        python -m src.models.forecasting
        ```
        This will train Persistence, Linear, Random Forest, and XGBoost models
        on Plant 1 and Plant 2 data and save predictions to `outputs/forecasts/`.
        """)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Train Solar Models Now", key="train_solar"):
                with st.spinner("Training models (this may take a few minutes)..."):
                    try:
                        from src.models.forecasting import train_solar_forecasting
                        result = train_solar_forecasting(plant=plant)
                        st.success("✅ Training complete!")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training error: {e}")
        return

    # ── Metrics Table ─────────────────────────────────────────────────────────
    st.subheader("📊 Model Performance Comparison")
    display_cols = ["model", "mae", "rmse", "r2", "nmae"]
    display_cols = [c for c in display_cols if c in metrics.columns]
    st.dataframe(
        metrics[display_cols].style
        .highlight_min(subset=["mae", "rmse", "nmae"], color="#14532d")
        .highlight_max(subset=["r2"], color="#14532d")
        .format({c: "{:.4f}" for c in ["mae", "rmse", "r2", "nmae"] if c in display_cols}),
        use_container_width=True,
    )
    st.caption("🟢 Green = Best value. MAE/RMSE/NMAE: lower is better. R²: higher is better.")

    # ── Model Comparison Bar Chart ─────────────────────────────────────────────
    fig_m = go.Figure()
    for metric_col, label in [("mae", "MAE (kW)"), ("rmse", "RMSE (kW)")]:
        if metric_col in metrics.columns:
            fig_m.add_trace(go.Bar(
                name=label, x=metrics["model"], y=metrics[metric_col],
                marker_color=["#60a5fa" if metric_col == "mae" else "#c084fc"] * len(metrics),
                text=metrics[metric_col].round(2), textposition="outside",
            ))
    fig_m.update_layout(**PLOTLY_LAYOUT, title="Error Metrics by Model", barmode="group")
    st.plotly_chart(fig_m, use_container_width=True)

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    st.subheader("📈 Actual vs Predicted Generation")
    models_available = [c for c in MODEL_COLORS.keys() if c in preds.columns and c != "actual"]
    selected_models = st.multiselect(
        "Show Models",
        models_available,
        default=models_available[:3] if len(models_available) >= 3 else models_available,
        key="solar_model_sel",
    )

    ts_col = "timestamp" if "timestamp" in preds.columns else preds.columns[0]

    # Horizon control
    n_show = st.slider("Show last N test rows", 50, min(500, len(preds)), 200, key="solar_horizon")
    plot_df = preds.tail(n_show).reset_index(drop=True)

    fig = go.Figure()
    if "actual" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["actual"],
            name="Actual", line=dict(color=MODEL_COLORS["actual"], width=2.5),
        ))
    for model in selected_models:
        if model in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[model],
                name=model.replace("_", " ").title(),
                line=dict(color=MODEL_COLORS.get(model, "#ffffff"), width=1.5, dash="dot"),
            ))
    fig.update_layout(**PLOTLY_LAYOUT, title=f"Solar Forecasting – Plant {plant} (Test Set, Last {n_show} Points)")
    fig.update_yaxes(title="AC Power (kW)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Scatter Plot ───────────────────────────────────────────────────────────
    if "actual" in preds.columns and len(selected_models) > 0:
        best_model = selected_models[0]
        if best_model in preds.columns:
            st.subheader(f"🔵 Actual vs Predicted Scatter – {best_model.title()}")
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(
                x=preds["actual"], y=preds[best_model],
                mode="markers",
                marker=dict(color=MODEL_COLORS.get(best_model, "#60a5fa"), size=3, opacity=0.5),
                name="Predictions",
            ))
            max_val = max(preds["actual"].max(), preds[best_model].max())
            fig_s.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", line=dict(color="#f59e0b", dash="dash"), name="Perfect Fit",
            ))
            fig_s.update_layout(**PLOTLY_LAYOUT, title="Actual vs Predicted Scatter")
            fig_s.update_xaxes(title="Actual (kW)")
            fig_s.update_yaxes(title="Predicted (kW)")
            st.plotly_chart(fig_s, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    fi_path = METRICS / f"solar_plant{plant}_rf_feature_importance.csv"
    if fi_path.exists():
        st.subheader("🎯 Feature Importance (Random Forest)")
        fi = pd.read_csv(fi_path).head(15)
        fig_fi = go.Figure(go.Bar(
            x=fi["importance"], y=fi["feature"], orientation="h",
            marker_color="#4ade80",
        ))
        fig_fi.update_layout(**PLOTLY_LAYOUT, title="Top 15 Feature Importances",
                             yaxis=dict(gridcolor="#1e293b", linecolor="#334155", autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.subheader("⬇️ Download Forecast Results")
    csv = preds.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions CSV", csv, f"solar_plant{plant}_predictions.csv", "text/csv")
