"""
SolarAI – Predictive Energy Forecasting & Dynamic Load Optimization
====================================================================
Three-page Streamlit dashboard.

Run:
    python -m streamlit run app/main.py
"""

import sys, json, io, warnings
from pathlib import Path
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
PROC    = BASE / "data"     / "processed"
METRICS = BASE / "outputs"  / "metrics"
FCASTS  = BASE / "outputs"  / "forecasts"
MDLS    = BASE / "models"

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
NAVY    = "#0B1628"
CARD    = "#0F2040"
BORDER  = "#1E3A5F"
ACCENT  = "#00C49A"        # green accent
BLUE    = "#2D9CDB"
ORANGE  = "#F4A261"
AMBER   = "#F2994A"
RED     = "#EB5757"
WHITE   = "#F0F4F8"
MUTED   = "#7D9BB5"

THEME = dict(
    paper_bgcolor=NAVY,
    plot_bgcolor=NAVY,
    font=dict(color=WHITE, family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="#1A2E4A", linecolor=BORDER, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor="#1A2E4A", linecolor=BORDER, tickfont=dict(color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, font=dict(color=WHITE)),
    margin=dict(l=24, r=24, t=40, b=24),
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {NAVY};
    color: {WHITE};
}}
[data-testid="stAppViewContainer"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] {{
    background: #081120;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

/* Radio buttons → pill nav */
div[data-testid="stRadio"] > label {{ display: none; }}
div[data-testid="stRadio"] div[role="radiogroup"] {{
    display: flex; flex-direction: column; gap: 6px;
}}
div[data-testid="stRadio"] label {{
    display: flex !important;
    align-items: center;
    padding: 10px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    color: {MUTED} !important;
    transition: all 0.2s;
}}
div[data-testid="stRadio"] label:has(input:checked) {{
    background: {CARD};
    border-left: 3px solid {ACCENT};
    color: {WHITE} !important;
}}
div[data-testid="stRadio"] label:hover {{
    background: {CARD};
    color: {WHITE} !important;
}}

/* KPI Cards */
.kpi-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    height: 110px;
    display: flex; flex-direction: column; justify-content: space-between;
}}
.kpi-label {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
    color: {MUTED}; text-transform: uppercase;
}}
.kpi-value {{
    font-size: 26px; font-weight: 700; color: {WHITE}; line-height: 1.1;
}}
.kpi-sub {{
    font-size: 11px; color: {MUTED};
}}
.kpi-accent {{ color: {ACCENT}; }}
.kpi-amber  {{ color: {AMBER}; }}
.kpi-red    {{ color: {RED}; }}
.kpi-blue   {{ color: {BLUE}; }}

/* Section headers */
.section-title {{
    font-size: 14px; font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; color: {MUTED};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px; margin: 24px 0 16px 0;
}}

/* Assumption badge */
.badge-warn {{
    display: inline-block;
    background: rgba(242,153,74,0.15);
    border: 1px solid {AMBER};
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px; color: {AMBER};
    margin: 4px 0;
}}

/* Metric delta */
.delta-pos {{ color: {ACCENT}; font-size:12px; }}
.delta-neg {{ color: {RED};   font-size:12px; }}

/* Plotly chart border */
[data-testid="stPlotlyChart"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}

/* Streamlit table */
.stDataFrame {{ background: {CARD}; border-radius: 8px; }}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT}, {BLUE});
    color: #000; font-weight: 600; border: none; border-radius: 8px;
    padding: 8px 24px;
}}
.stDownloadButton > button {{
    background: {CARD}; color: {WHITE}; border: 1px solid {BORDER};
    border-radius: 8px; font-size: 12px;
}}
hr {{ border-color: {BORDER}; }}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SolarAI | Predictive Energy Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def kpi(label, value, sub="", color="kpi-accent"):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {color}">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def warn_badge(text):
    st.markdown(f'<span class="badge-warn">⚠ {text}</span>', unsafe_allow_html=True)

def apply_theme(fig, title="", height=340):
    fig.update_layout(**THEME, title=dict(text=title, font=dict(size=14, color=WHITE)),
                      height=height)
    return fig

@st.cache_data(ttl=600)
def load_solar_gen(plant: int) -> pd.DataFrame:
    p = PROC / f"solar_generation_plant{plant}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    df = df.dropna(subset=["DATE_TIME"]).sort_values("DATE_TIME")
    return df

@st.cache_data(ttl=600)
def load_solar_weather(plant: int) -> pd.DataFrame:
    p = PROC / f"solar_weather_plant{plant}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], errors="coerce")
    return df.dropna(subset=["DATE_TIME"]).sort_values("DATE_TIME")

@st.cache_data(ttl=600)
def load_demand() -> pd.DataFrame:
    p = PROC / "electricity_demand_DE.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")

@st.cache_data(ttl=600)
def load_carbon() -> pd.DataFrame:
    p = PROC / "carbon_emissions.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False, nrows=50_000)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    return df.dropna(subset=["DATETIME"]).sort_values("DATETIME")

@st.cache_data(ttl=600)
def load_metrics(task: str) -> pd.DataFrame:
    p = METRICS / f"{task}_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=600)
def load_predictions(task: str) -> pd.DataFrame:
    p = FCASTS / f"{task}_predictions.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")

@st.cache_data(ttl=600)
def load_feature_importance(task: str, model: str) -> pd.DataFrame:
    p = METRICS / f"{task}_{model}_feature_importance.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='padding:20px 0 16px 0; text-align:center;'>
      <div style='font-size:32px'>⚡</div>
      <div style='font-size:18px; font-weight:700; color:{ACCENT}; letter-spacing:0.04em'>SolarAI</div>
      <div style='font-size:11px; color:{MUTED}; margin-top:2px'>Predictive Energy Platform</div>
    </div>
    <hr style='border-color:{BORDER}; margin:0 0 16px 0'/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["📊  Overview", "🤖  AI Models & Forecasting", "🔋  Energy Optimization"],
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <hr style='border-color:{BORDER}; margin:20px 0 12px 0'/>
    <div style='font-size:11px; color:{MUTED}; padding:0 4px'>
        <b style='color:{WHITE}'>Data sources</b><br>
        Solar (India, 2020) · EU Demand · UK Carbon · OPSD Weather<br><br>
        <b style='color:{WHITE}'>Models</b><br>
        Persistence · Ridge · RF · XGBoost<br><br>
        <b style='color:{WHITE}'>Optimizer</b><br>
        PuLP + HiGHS LP solver
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        plant_sel = st.selectbox("Solar Plant", [1, 2], key="ov_plant")
    with c2:
        view_res = st.selectbox("Resolution", ["15-min", "Hourly", "Daily"], key="ov_res")

    gen_df = load_solar_gen(plant_sel)
    wx_df  = load_solar_weather(plant_sel)

    if gen_df.empty:
        st.error("Preprocessed solar data not found. Run: `python -m src.data.preprocessing`")
        st.stop()

    dmin = gen_df["DATE_TIME"].dt.date.min()
    dmax = gen_df["DATE_TIME"].dt.date.max()
    with c3:
        date_range = st.date_input(
            "Date Range",
            value=(dmin, dmax),
            min_value=dmin, max_value=dmax, key="ov_dates"
        )
    if len(date_range) == 2:
        d0, d1 = date_range
        gen_df = gen_df[(gen_df["DATE_TIME"].dt.date >= d0) & (gen_df["DATE_TIME"].dt.date <= d1)]
        if not wx_df.empty:
            wx_df = wx_df[(wx_df["DATE_TIME"].dt.date >= d0) & (wx_df["DATE_TIME"].dt.date <= d1)]

    # Resample
    dt_map = {"15-min": "15min", "Hourly": "1h", "Daily": "D"}
    freq = dt_map[view_res]
    gen_r = gen_df.set_index("DATE_TIME").resample(freq).mean(numeric_only=True).reset_index()

    st.markdown(f"<h2 style='color:{WHITE}; font-size:22px; font-weight:700; margin:12px 0 4px 0'>Energy Overview — Plant {plant_sel}</h2>", unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    section("Key Performance Indicators")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    dt_h = 0.25  # 15-min intervals = 0.25 h per reading
    total_kwh  = gen_df["AC_POWER"].sum() * dt_h / 1000
    avg_kw     = gen_df["AC_POWER"].mean()
    peak_kw    = gen_df["AC_POWER"].max()
    n_days     = max(1, (gen_df["DATE_TIME"].max() - gen_df["DATE_TIME"].min()).days)
    avg_daily  = total_kwh / n_days

    with k1: kpi("Total Generation", f"{total_kwh:,.0f} kWh", f"{n_days} days", "kpi-accent")
    with k2: kpi("Avg Solar Power", f"{avg_kw:,.1f} kW", "all intervals", "kpi-blue")
    with k3: kpi("Peak Power", f"{peak_kw:,.1f} kW", "single interval", "kpi-amber")
    with k4: kpi("Avg Daily Yield", f"{avg_daily:,.1f} kWh/day", f"plant {plant_sel}", "kpi-accent")

    # Carbon and demand from separate datasets
    cb_df = load_carbon()
    if not cb_df.empty and "CARBON_INTENSITY" in cb_df.columns:
        avg_ci = cb_df["CARBON_INTENSITY"].mean()
        co2_avoided = total_kwh * avg_ci / 1000  # kg
        with k5: kpi("CO₂ Avoided (est.)", f"{co2_avoided:,.0f} kg", f"CI={avg_ci:.0f} gCO₂/kWh (UK proxy)", "kpi-accent")
    else:
        with k5: kpi("CO₂ Avoided", "Not available", "Carbon data missing", "kpi-blue")

    dm_df = load_demand()
    if not dm_df.empty:
        load_col = next((c for c in dm_df.columns if "load_actual" in c.lower()), None)
        if load_col:
            avg_dem = dm_df[load_col].mean()
            with k6: kpi("DE Avg Demand", f"{avg_dem:,.0f} MW", "Germany ENTSOE", "kpi-amber")
        else:
            with k6: kpi("Grid Demand", "Not available", "No matching column", "kpi-blue")
    else:
        with k6: kpi("Grid Demand", "Not available", "Demand data missing", "kpi-blue")

    # ── Solar Generation Chart ─────────────────────────────────────────────
    section("Solar Generation Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gen_r["DATE_TIME"], y=gen_r["AC_POWER"],
        name="AC Power (kW)", mode="lines",
        line=dict(color=ACCENT, width=1.5),
        fill="tozeroy", fillcolor=f"rgba(0,196,154,0.12)"
    ))
    if "DC_POWER" in gen_r.columns:
        fig.add_trace(go.Scatter(
            x=gen_r["DATE_TIME"], y=gen_r["DC_POWER"],
            name="DC Power (kW)", mode="lines",
            line=dict(color=BLUE, width=1, dash="dot")
        ))
    apply_theme(fig, f"Solar Generation – Plant {plant_sel} ({view_res})", 320)
    fig.update_yaxes(title="kW")
    st.plotly_chart(fig, use_container_width=True)

    # ── Daily Profile + Weather side-by-side ──────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        section("Average Daily Generation Profile")
        gen_df_h = gen_df.copy()
        gen_df_h["hour"] = gen_df_h["DATE_TIME"].dt.hour
        hourly_avg = gen_df_h.groupby("hour")["AC_POWER"].mean()
        fig2 = go.Figure(go.Bar(
            x=hourly_avg.index, y=hourly_avg.values,
            marker_color=ACCENT, name="Avg AC Power",
            marker_line_width=0,
        ))
        apply_theme(fig2, "Hourly Average Profile", 300)
        fig2.update_xaxes(title="Hour of Day", tickvals=list(range(0, 24, 2)))
        fig2.update_yaxes(title="kW")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        section("Weather – Temperature & Irradiation")
        if not wx_df.empty:
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Scatter(
                x=wx_df["DATE_TIME"], y=wx_df["AMBIENT_TEMPERATURE"],
                name="Ambient Temp (°C)", line=dict(color=ORANGE, width=1.2)
            ), secondary_y=False)
            fig3.add_trace(go.Scatter(
                x=wx_df["DATE_TIME"], y=wx_df["IRRADIATION"],
                name="Irradiation (W/m²)", line=dict(color=AMBER, width=1, dash="dot")
            ), secondary_y=True)
            fig3.update_layout(**THEME, height=300)
            fig3.update_yaxes(title_text="°C", secondary_y=False,
                              gridcolor="#1A2E4A", tickfont=dict(color=MUTED))
            fig3.update_yaxes(title_text="W/m²", secondary_y=True,
                              tickfont=dict(color=MUTED))
            fig3.update_xaxes(gridcolor="#1A2E4A", tickfont=dict(color=MUTED))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Weather sensor data not available for this plant.")

    # ── Irradiation vs Generation scatter ────────────────────────────────
    if not wx_df.empty and "IRRADIATION" in wx_df.columns:
        section("Irradiation vs. Solar Output")
        merged = pd.merge_asof(
            gen_df.sort_values("DATE_TIME")[["DATE_TIME", "AC_POWER"]],
            wx_df.sort_values("DATE_TIME")[["DATE_TIME", "IRRADIATION", "AMBIENT_TEMPERATURE"]],
            on="DATE_TIME", direction="nearest", tolerance=pd.Timedelta("5min")
        ).dropna()
        merged = merged[merged["IRRADIATION"] > 0]
        if len(merged) > 10:
            fig4 = go.Figure(go.Scatter(
                x=merged["IRRADIATION"], y=merged["AC_POWER"],
                mode="markers",
                marker=dict(
                    color=merged["AMBIENT_TEMPERATURE"],
                    colorscale="RdYlGn", size=3, opacity=0.55,
                    colorbar=dict(title="Temp (°C)", tickfont=dict(color=WHITE))
                ),
                name="Power vs Irradiation",
            ))
            apply_theme(fig4, "Irradiation vs AC Power (colour = temperature)", 280)
            fig4.update_xaxes(title="Irradiation (W/m²)")
            fig4.update_yaxes(title="AC Power (kW)")
            st.plotly_chart(fig4, use_container_width=True)

    # ── Sustainability Summary ────────────────────────────────────────────
    section("Sustainability Summary")
    warn_badge("Carbon avoidance is estimated using UK National Grid average intensity as a proxy — not a direct measurement from plant location.")
    st.markdown("")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Renewable Generation</div>
          <div class="kpi-value kpi-accent">{total_kwh:,.0f} kWh</div>
          <div class="kpi-sub">100 % measured solar · Plant {plant_sel}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        if not cb_df.empty and "CARBON_INTENSITY" in cb_df.columns:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">CO₂ Avoidance (proxy est.)</div>
              <div class="kpi-value kpi-accent">{co2_avoided:,.0f} kg</div>
              <div class="kpi-sub">UK grid CI {avg_ci:.0f} gCO₂/kWh (simulated)</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-card"><div class="kpi-label">CO₂ Avoidance</div><div class="kpi-value kpi-blue">N/A</div><div class="kpi-sub">Carbon data not loaded</div></div>', unsafe_allow_html=True)
    with s3:
        equiv_homes = int(total_kwh / 900)  # rough UK avg
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Equivalent Homes (1 month)</div>
          <div class="kpi-value kpi-amber">~{equiv_homes:,}</div>
          <div class="kpi-sub">900 kWh/home/month assumed</div>
        </div>""", unsafe_allow_html=True)

    # ── Dataset Info ──────────────────────────────────────────────────────
    with st.expander("📂 Dataset Information", expanded=False):
        rows = []
        rows.append({"Dataset": f"Solar Generation Plant {plant_sel}", "Source": "Kaggle (India, 2020)",
                     "Period": f"{dmin} – {dmax}", "Rows": f"{len(gen_df):,}", "Freq": "15 min", "Completeness": f"{100*(1-gen_df.isnull().mean().mean()):.1f}%"})
        if not wx_df.empty:
            rows.append({"Dataset": f"Weather Sensor Plant {plant_sel}", "Source": "Co-located sensor",
                         "Period": f"{wx_df['DATE_TIME'].dt.date.min()} – {wx_df['DATE_TIME'].dt.date.max()}",
                         "Rows": f"{len(wx_df):,}", "Freq": "15 min", "Completeness": f"{100*(1-wx_df.isnull().mean().mean()):.1f}%"})
        if not dm_df.empty:
            rows.append({"Dataset": "Electricity Demand DE", "Source": "ENTSOE via OPSD",
                         "Period": f"{dm_df['timestamp'].dt.date.min()} – {dm_df['timestamp'].dt.date.max()}",
                         "Rows": f"{len(dm_df):,}", "Freq": "60 min", "Completeness": "See report"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — AI MODELS & FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif "Forecasting" in page:

    st.markdown(f"<h2 style='color:{WHITE}; font-size:22px; font-weight:700; margin:12px 0 4px 0'>AI Models & Forecasting</h2>", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
    with ctrl1:
        target = st.selectbox("Forecast Target",
            ["Solar Generation – Plant 1", "Solar Generation – Plant 2", "Electricity Demand (DE)"],
            key="fc_target")
    with ctrl2:
        model_sel = st.selectbox("Model",
            ["random_forest", "xgboost", "linear_ridge", "persistence"],
            format_func=lambda x: {"random_forest":"Random Forest","xgboost":"XGBoost",
                                   "linear_ridge":"Linear Regression (Ridge)","persistence":"Persistence Baseline"}[x],
            key="fc_model")
    with ctrl3:
        n_show = st.slider("Test points to display", 50, 500, 200, key="fc_n")
    with ctrl4:
        show_ci = st.checkbox("Show residual band", value=True, key="fc_ci")

    # Map target to task key
    if "Plant 1" in target:
        task = "solar_plant1"
        unit = "kW (AC Power)"
    elif "Plant 2" in target:
        task = "solar_plant2"
        unit = "kW (AC Power)"
    else:
        task = "demand_DE"
        unit = "MW (Load)"

    metrics_df = load_metrics(task)
    preds_df   = load_predictions(task)

    if preds_df.empty:
        st.warning(f"No predictions found for **{task}**. Train models first:")
        st.code("python -m src.models.forecasting")
        st.stop()

    # ── Model Comparison Table ─────────────────────────────────────────────
    section("Model Comparison — Test Set Metrics")
    if not metrics_df.empty:
        rename = {"model":"Model","mae":"MAE","rmse":"RMSE","r2":"R²","nmae":"NMAE","mape_pct":"MAPE (%)"}
        display_m = metrics_df.rename(columns=rename)
        display_m["Model"] = display_m["Model"].map({
            "random_forest":"Random Forest", "xgboost":"XGBoost",
            "linear_ridge":"Linear Ridge", "persistence":"Persistence"
        }).fillna(display_m["Model"])
        # highlight best
        num_cols = ["MAE","RMSE","R²","NMAE","MAPE (%)"]
        num_cols = [c for c in num_cols if c in display_m.columns]
        st.dataframe(
            display_m[["Model"] + num_cols].style
              .highlight_min(subset=["MAE","RMSE"], color="#0a2a1a")
              .highlight_max(subset=["R²"], color="#0a2a1a")
              .format({c: "{:.4f}" for c in num_cols}),
            use_container_width=True, hide_index=True
        )

    # ── KPI Cards for selected model ──────────────────────────────────────
    if not metrics_df.empty:
        row = metrics_df[metrics_df["model"] == model_sel]
        if not row.empty:
            row = row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            with m1: kpi("MAE", f"{row['mae']:,.1f}", unit, "kpi-accent")
            with m2: kpi("RMSE", f"{row['rmse']:,.1f}", unit, "kpi-blue")
            with m3: kpi("R²", f"{row['r2']:.4f}", "1.0 = perfect", "kpi-amber")
            with m4:
                if pd.notna(row.get("mape_pct", float("nan"))):
                    kpi("MAPE", f"{row['mape_pct']:.2f}%", "% error", "kpi-accent")
                else:
                    kpi("MAPE", "N/A", "zeros in actuals", "kpi-blue")

    # ── Actual vs Predicted chart ─────────────────────────────────────────
    section("Actual vs Predicted — Test Set")
    plot_df = preds_df.tail(n_show).reset_index(drop=True)

    MODEL_COLORS = {
        "persistence":  MUTED,
        "linear_ridge": BLUE,
        "random_forest": ACCENT,
        "xgboost":      ORANGE,
    }

    fig = go.Figure()
    if "actual" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df["timestamp"], y=plot_df["actual"],
            name="Actual", mode="lines",
            line=dict(color=WHITE, width=2)
        ))
    if model_sel in plot_df.columns:
        color = MODEL_COLORS.get(model_sel, ACCENT)
        fig.add_trace(go.Scatter(
            x=plot_df["timestamp"], y=plot_df[model_sel],
            name=model_sel.replace("_", " ").title(), mode="lines",
            line=dict(color=color, width=1.5, dash="dot")
        ))
        if show_ci and "actual" in plot_df.columns:
            residuals = plot_df["actual"] - plot_df[model_sel]
            sigma = residuals.std()
            fig.add_trace(go.Scatter(
                x=pd.concat([plot_df["timestamp"], plot_df["timestamp"].iloc[::-1]]),
                y=pd.concat([plot_df[model_sel] + sigma, (plot_df[model_sel] - sigma).iloc[::-1]]),
                fill="toself", fillcolor=f"rgba(0,196,154,0.08)",
                line=dict(color="rgba(0,0,0,0)"), name="±1σ band", showlegend=True,
            ))
    apply_theme(fig, f"{task} — Actual vs {model_sel.replace('_',' ').title()} (last {n_show} test points)", 360)
    fig.update_yaxes(title=unit)
    st.plotly_chart(fig, use_container_width=True)

    # ── All models overlay ────────────────────────────────────────────────
    with st.expander("Compare All Models", expanded=False):
        fig_all = go.Figure()
        if "actual" in plot_df.columns:
            fig_all.add_trace(go.Scatter(
                x=plot_df["timestamp"], y=plot_df["actual"],
                name="Actual", line=dict(color=WHITE, width=2)
            ))
        for m, c in MODEL_COLORS.items():
            if m in plot_df.columns:
                fig_all.add_trace(go.Scatter(
                    x=plot_df["timestamp"], y=plot_df[m],
                    name=m.replace("_"," ").title(),
                    line=dict(color=c, width=1.2, dash="dot")
                ))
        apply_theme(fig_all, "All Models vs Actual", 360)
        fig_all.update_yaxes(title=unit)
        st.plotly_chart(fig_all, use_container_width=True)

    # ── Residuals ─────────────────────────────────────────────────────────
    if "actual" in plot_df.columns and model_sel in plot_df.columns:
        section("Prediction Error (Residuals)")
        col_r1, col_r2 = st.columns(2)
        residuals = plot_df["actual"] - plot_df[model_sel]

        with col_r1:
            fig_res = go.Figure(go.Scatter(
                x=plot_df["timestamp"], y=residuals,
                mode="lines", line=dict(color=ORANGE, width=1),
                fill="tozeroy", fillcolor="rgba(244,162,97,0.12)"
            ))
            apply_theme(fig_res, "Residuals over Time", 280)
            fig_res.add_hline(y=0, line_dash="dash", line_color=MUTED)
            fig_res.update_yaxes(title=f"Error ({unit})")
            st.plotly_chart(fig_res, use_container_width=True)

        with col_r2:
            fig_hist = go.Figure(go.Histogram(
                x=residuals, nbinsx=40,
                marker_color=ACCENT, opacity=0.8,
                name="Error distribution"
            ))
            apply_theme(fig_hist, "Error Distribution", 280)
            fig_hist.update_xaxes(title=f"Residual ({unit})")
            fig_hist.update_yaxes(title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)

    # ── Feature Importance ─────────────────────────────────────────────────
    if model_sel in ("random_forest", "xgboost"):
        fi_df = load_feature_importance(task, {"random_forest":"rf","xgboost":"xgb"}[model_sel])
        if not fi_df.empty:
            section("Feature Importance")
            fi_top = fi_df.head(15)
            fig_fi = go.Figure(go.Bar(
                x=fi_top["importance"], y=fi_top["feature"],
                orientation="h", marker_color=ACCENT, marker_line_width=0
            ))
            apply_theme(fig_fi, f"Top-15 Features — {model_sel.replace('_',' ').title()}", 340)
            fig_fi.update_layout(yaxis=dict(autorange="reversed", gridcolor="#1A2E4A", tickfont=dict(color=WHITE, size=11)))
            st.plotly_chart(fig_fi, use_container_width=True)

    # ── Forecast limitation note ──────────────────────────────────────────
    section("Forecast Limitations")
    warn_badge("Future forecasts require future weather inputs (irradiation, temperature). Without live sensor data, only in-sample test-set predictions are available above.")
    st.markdown(f"""
    <div style='font-size:12px; color:{MUTED}; margin-top:8px'>
    The charts above show model performance on held-out test data (chronological 70/15/15 split).
    Real-time forecasting would require a live weather feed or NWP model output.
    </div>""", unsafe_allow_html=True)

    # ── Downloads ─────────────────────────────────────────────────────────
    section("Downloads")
    d1, d2 = st.columns(2)
    with d1:
        csv = preds_df.to_csv(index=False).encode("utf-8")
        st.download_button(f"⬇ Predictions CSV — {task}", csv,
                           f"{task}_predictions.csv", "text/csv")
    with d2:
        if not metrics_df.empty:
            csv_m = metrics_df.to_csv(index=False).encode("utf-8")
            st.download_button(f"⬇ Metrics CSV — {task}", csv_m,
                               f"{task}_metrics.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ENERGY OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Optimization" in page:

    st.markdown(f"<h2 style='color:{WHITE}; font-size:22px; font-weight:700; margin:12px 0 4px 0'>Energy Optimization — Building-to-Grid Demand Response</h2>", unsafe_allow_html=True)
    warn_badge("Battery specs, tariff, and carbon intensity are user-configurable scenario assumptions — not measured dataset values.")
    st.markdown("")

    # ── Configuration Panel ───────────────────────────────────────────────
    with st.expander("⚙️ System Configuration", expanded=True):
        cfg1, cfg2, cfg3 = st.columns(3)

        with cfg1:
            st.markdown(f"<div style='font-size:12px;font-weight:600;color:{MUTED};text-transform:uppercase;margin-bottom:8px'>Battery Storage</div>", unsafe_allow_html=True)
            bat_cap  = st.slider("Capacity (kWh)", 10.0, 500.0, 100.0, 5.0, key="o_bcap")
            bat_init = st.slider("Initial SoC (kWh)", 0.0, bat_cap, bat_cap * 0.5, 5.0, key="o_binit")
            bat_mc   = st.slider("Max Charge Rate (kW)", 5.0, 100.0, 30.0, 5.0, key="o_bmc")
            bat_md   = st.slider("Max Discharge Rate (kW)", 5.0, 100.0, 30.0, 5.0, key="o_bmd")
            c_eff    = st.slider("Charge Efficiency", 0.80, 1.0, 0.95, 0.01, key="o_ceff")
            d_eff    = st.slider("Discharge Efficiency", 0.80, 1.0, 0.95, 0.01, key="o_deff")

        with cfg2:
            st.markdown(f"<div style='font-size:12px;font-weight:600;color:{MUTED};text-transform:uppercase;margin-bottom:8px'>Tariff & Carbon</div>", unsafe_allow_html=True)
            tariff_type = st.radio("Tariff Structure", ["Flat Rate", "Time-of-Use"], key="o_ttype")
            flat_rate   = st.slider("Base Rate (€/kWh)", 0.10, 0.50, 0.25, 0.01, key="o_flat")
            if tariff_type == "Time-of-Use":
                peak_rate = st.slider("Peak Rate €/kWh (7:00–21:00)", flat_rate, 0.60, 0.35, 0.01, key="o_peak")
            else:
                peak_rate = flat_rate
            ci_val = st.number_input("Carbon Intensity (gCO₂/kWh)", 50.0, 900.0, 280.0, 10.0, key="o_ci")
            st.caption("UK avg ~280 · DE avg ~350 · FR avg ~80 (nuclear-heavy)")

        with cfg3:
            st.markdown(f"<div style='font-size:12px;font-weight:600;color:{MUTED};text-transform:uppercase;margin-bottom:8px'>Optimization Objective & Flex Load</div>", unsafe_allow_html=True)
            objective = st.selectbox("Objective", [
                "Minimize Cost", "Minimize Carbon", "Minimize Peak Demand", "Balanced"
            ], key="o_obj")
            weight_map = {
                "Minimize Cost":         (2.0, 0.1, 0.1),
                "Minimize Carbon":       (0.1, 2.0, 0.1),
                "Minimize Peak Demand":  (0.1, 0.1, 2.0),
                "Balanced":              (1.0, 1.0, 1.0),
            }
            cost_w, carbon_w, peak_w = weight_map[objective]
            flex_e    = st.slider("Total Flexible Energy (kWh)", 0.0, 80.0, 20.0, 1.0, key="o_fe")
            flex_kw   = st.slider("Max Flex Power (kW)", 1.0, 40.0, 10.0, 1.0, key="o_fkw")
            flex_s    = st.slider("Flex Window Start (h)", 0, 22, 6, key="o_fs")
            flex_e_end = st.slider("Flex Window End (h)", flex_s + 1, 23, 22, key="o_fen")
            grid_lim  = st.slider("Grid Import Limit (kW)", 50.0, 500.0, 200.0, 10.0, key="o_glim")

    run_btn = st.button("▶  Run Optimization", type="primary", key="o_run")

    # ── Run ───────────────────────────────────────────────────────────────
    if run_btn or ("opt_result" in st.session_state):
        if run_btn:
            from src.optimization.optimizer import (
                OptimizationConfig, OptimizationInputs,
                run_baseline, run_optimization, compare_scenarios
            )

            T = 24
            h = np.arange(T)
            tariff_vec = np.where((h >= 7) & (h <= 21), peak_rate, flat_rate) if tariff_type == "Time-of-Use" else np.full(T, flat_rate)

            # Try real solar data
            solar = None
            for pl in [1, 2]:
                fp = PROC / f"solar_generation_plant{pl}.csv"
                if fp.exists():
                    sdf = pd.read_csv(fp)
                    sdf["DATE_TIME"] = pd.to_datetime(sdf["DATE_TIME"])
                    mid = sdf["DATE_TIME"].dt.date.dropna().iloc[len(sdf)//2]
                    day_df = sdf[sdf["DATE_TIME"].dt.date == mid].head(T)
                    if len(day_df) >= T:
                        s = day_df["AC_POWER"].values[:T].astype(float)
                        s = s / max(s.max(), 1) * 60
                        solar = s
                        break

            if solar is None:
                solar = 60 * np.exp(-0.5 * ((h - 12) / 3) ** 2)
                solar[:6] = 0; solar[20:] = 0

            load = np.array([15,13,12,12,13,16,19,26,31,33,31,29,27,29,31,33,36,41,39,36,30,25,20,17], dtype=float)

            cfg = OptimizationConfig(
                battery_capacity_kwh=bat_cap,
                initial_soc_kwh=bat_init,
                max_charge_kw=bat_mc,
                max_discharge_kw=bat_md,
                charge_efficiency=c_eff,
                discharge_efficiency=d_eff,
                total_flex_energy_kwh=flex_e,
                max_flex_kw=flex_kw,
                flex_window_start=flex_s,
                flex_window_end=flex_e_end,
                grid_import_limit_kw=grid_lim,
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
                carbon_intensity_gco2_kwh=np.full(T, ci_val),
                hours=h,
            )

            with st.spinner("Running LP optimization with HiGHS…"):
                baseline  = run_baseline(inputs, cfg)
                optimized = run_optimization(inputs, cfg)
                comparison = compare_scenarios(baseline, optimized)

            st.session_state["opt_result"]  = {"baseline": baseline, "optimized": optimized,
                                                "comparison": comparison, "solar": solar, "load": load,
                                                "tariff": tariff_vec, "ci": ci_val}
            st.success(f"✅ Optimization complete — Status: {optimized.status}")

        # ── Read from session state ───────────────────────────────────────
        res = st.session_state.get("opt_result", {})
        if not res:
            st.info("Configure the parameters above and click **Run Optimization**.")
            st.stop()

        baseline   = res["baseline"]
        optimized  = res["optimized"]
        comparison = res["comparison"]
        solar      = res["solar"]
        load       = res["load"]
        ci_val_r   = res.get("ci", 280.0)

        # ── Comparison KPI Cards ──────────────────────────────────────────
        section("Baseline vs Optimized Results")
        n_metrics = min(len(comparison), 4)
        kpi_cols = st.columns(n_metrics if n_metrics > 0 else 4)
        metric_icons = {"Total Cost": "€", "Carbon": "kg", "Peak": "kW", "Solar": "%"}

        for i, (_, row) in enumerate(comparison.iterrows()):
            if i >= len(kpi_cols): break
            pct = row.get("Saving_%", row.get("Saving_pct", 0))
            with kpi_cols[i]:
                arrow = "↓" if pct > 0 else "↑"
                color_cls = "kpi-accent" if pct > 0 else "kpi-red"
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-label">{row['Metric']}</div>
                  <div style='font-size:13px; color:{MUTED}; margin:2px 0'>
                    Base: <b style='color:{WHITE}'>{row['Baseline']:.2f}</b>
                  </div>
                  <div style='font-size:13px; color:{MUTED}'>
                    Opt: <b style='color:{ACCENT}'>{row['Optimized']:.2f}</b>
                  </div>
                  <div class="{color_cls}" style='font-size:14px;font-weight:700;margin-top:4px'>
                    {arrow} {abs(pct):.1f}% {'saved' if pct > 0 else 'increase'}
                  </div>
                </div>""", unsafe_allow_html=True)

        # ── Energy Balance Chart ──────────────────────────────────────────
        section("Energy Balance")
        h = np.arange(24)
        tab1, tab2 = st.tabs(["Optimized Schedule", "Baseline vs Optimized"])

        with tab1:
            fig_bal = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.65, 0.35], vertical_spacing=0.06)
            # Top: power flows
            fig_bal.add_trace(go.Bar(x=h, y=optimized.grid_import, name="Grid Import",
                                     marker_color="#2D9CDB", opacity=0.85), row=1, col=1)
            fig_bal.add_trace(go.Bar(x=h, y=optimized.solar_used, name="Solar Used",
                                     marker_color=ACCENT, opacity=0.85), row=1, col=1)
            fig_bal.add_trace(go.Bar(x=h, y=optimized.batt_discharge, name="Battery Discharge",
                                     marker_color=ORANGE, opacity=0.85), row=1, col=1)
            fig_bal.add_trace(go.Scatter(x=h, y=load + optimized.flex_load,
                                          name="Total Load", mode="lines+markers",
                                          line=dict(color=WHITE, width=2)), row=1, col=1)
            fig_bal.add_trace(go.Scatter(x=h, y=solar, name="Solar Available",
                                          line=dict(color=AMBER, width=1.5, dash="dot")), row=1, col=1)
            # Bottom: SoC
            fig_bal.add_trace(go.Scatter(x=h, y=optimized.soc, name="Battery SoC",
                                          fill="tozeroy", fillcolor="rgba(139,92,246,0.2)",
                                          line=dict(color="#8B5CF6", width=2)), row=2, col=1)
            fig_bal.update_layout(**THEME, barmode="stack", height=480,
                                   title="Optimized Energy Schedule (24 h)")
            fig_bal.update_yaxes(title="kW", row=1, col=1, gridcolor="#1A2E4A")
            fig_bal.update_yaxes(title="kWh (SoC)", row=2, col=1, gridcolor="#1A2E4A")
            fig_bal.update_xaxes(title="Hour of Day", row=2, col=1,
                                  tickvals=list(range(0, 24, 2)), gridcolor="#1A2E4A")
            st.plotly_chart(fig_bal, use_container_width=True)

        with tab2:
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=h, y=baseline.grid_import, name="Baseline Import",
                                          line=dict(color=RED, width=2),
                                          fill="tozeroy", fillcolor="rgba(235,87,87,0.12)"))
            fig_cmp.add_trace(go.Scatter(x=h, y=optimized.grid_import, name="Optimized Import",
                                          line=dict(color=ACCENT, width=2),
                                          fill="tozeroy", fillcolor="rgba(0,196,154,0.12)"))
            fig_cmp.add_trace(go.Scatter(x=h, y=solar, name="Solar Available",
                                          line=dict(color=AMBER, width=1.5, dash="dot")))
            apply_theme(fig_cmp, "Baseline vs Optimized Grid Import", 340)
            fig_cmp.update_xaxes(title="Hour of Day", tickvals=list(range(0, 24, 2)))
            fig_cmp.update_yaxes(title="kW")
            st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Flex Load & Charging Schedule ────────────────────────────────
        fl1, fl2 = st.columns(2)
        with fl1:
            section("Flexible Load Schedule")
            fig_fl = go.Figure(go.Bar(x=h, y=optimized.flex_load,
                                       marker_color=ACCENT, name="Flex Load (kW)",
                                       marker_line_width=0))
            apply_theme(fig_fl, "Scheduled Flexible Load", 240)
            fig_fl.update_xaxes(title="Hour")
            fig_fl.update_yaxes(title="kW")
            st.plotly_chart(fig_fl, use_container_width=True)

        with fl2:
            section("Battery Charge / Discharge")
            fig_bat = go.Figure()
            fig_bat.add_trace(go.Bar(x=h, y=optimized.batt_charge, name="Charging",
                                      marker_color=ACCENT, opacity=0.9))
            fig_bat.add_trace(go.Bar(x=h, y=-optimized.batt_discharge, name="Discharging",
                                      marker_color=RED, opacity=0.9))
            apply_theme(fig_bat, "Battery Charge / Discharge", 240)
            fig_bat.update_layout(barmode="relative")
            fig_bat.update_xaxes(title="Hour")
            fig_bat.update_yaxes(title="kW (+ charge / − discharge)")
            st.plotly_chart(fig_bat, use_container_width=True)

        # ── Recommended Schedule Table ────────────────────────────────────
        section("Recommended Hourly Schedule")
        sched = pd.DataFrame({
            "Hour": h,
            "Solar Available (kW)": solar.round(2),
            "Building Load (kW)": load.round(2),
            "Grid Import (kW)": optimized.grid_import.round(2),
            "Battery SoC (kWh)": optimized.soc.round(2),
            "Batt Charge (kW)": optimized.batt_charge.round(2),
            "Batt Discharge (kW)": optimized.batt_discharge.round(2),
            "Flex Load (kW)": optimized.flex_load.round(2),
        })
        st.dataframe(sched, use_container_width=True, hide_index=True)

        # ── Downloads ─────────────────────────────────────────────────────
        section("Export Results")
        d1, d2, d3, d4 = st.columns(4)

        with d1:
            csv_s = sched.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Schedule CSV", csv_s, "optimized_schedule.csv", "text/csv")

        with d2:
            cmp_csv = comparison.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Comparison CSV", cmp_csv, "baseline_vs_optimized.csv", "text/csv")

        with d3:
            lines = [
                "# SolarAI Sustainability Report\n",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n",
                "## Optimization Results\n",
            ]
            for _, row in comparison.iterrows():
                pct = row.get("Saving_%", row.get("Saving_pct", 0))
                lines.append(f"- **{row['Metric']}**: Baseline={row['Baseline']:.2f} → Optimized={row['Optimized']:.2f} (Saving: {pct:.1f}%)\n")
            lines += [
                "\n## Documented Assumptions\n",
                f"- Carbon intensity: {ci_val_r:.0f} gCO2/kWh (user-configured scenario)\n",
                f"- Battery capacity: {bat_cap} kWh (scenario)\n",
                f"- Flat tariff: EUR{flat_rate}/kWh\n",
                "- Building load profile: representative synthetic/scaled profile\n",
                "- Solar generation: scaled from real plant measurements where available\n",
                "\n## Data Sources\n",
                "- Solar: Kaggle Solar Power Generation Data (India, 2020)\n",
                "- Carbon: UK National Grid ESO\n",
                "- Demand: Open Power System Data / ENTSOE (Germany)\n",
            ]
            report_bytes = "".join(lines).encode("utf-8")
            st.download_button("⬇ Sustainability Report (MD)", report_bytes, "sustainability_report.md", "text/markdown")

        with d4:
            assumptions = {
                "battery_capacity_kwh": bat_cap, "initial_soc_kwh": bat_init,
                "max_charge_kw": bat_mc, "max_discharge_kw": bat_md,
                "charge_efficiency": c_eff, "discharge_efficiency": d_eff,
                "tariff_type": tariff_type, "flat_rate_eur_kwh": flat_rate,
                "carbon_intensity_gco2_kwh": ci_val_r,
                "optimization_objective": objective,
                "flex_energy_kwh": flex_e, "grid_limit_kw": grid_lim,
                "generated_at": datetime.now().isoformat(),
                "note": "All battery/tariff/CI values are user-configured scenario assumptions."
            }
            st.download_button("⬇ Assumptions JSON", json.dumps(assumptions, indent=2).encode("utf-8"),
                               "optimization_assumptions.json", "application/json")

    else:
        st.info("Configure the parameters in the panel above, then click **▶ Run Optimization** to generate the schedule.")
        st.markdown(f"""
        <div style='background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:20px; margin-top:12px'>
          <div style='font-size:14px; font-weight:600; color:{WHITE}; margin-bottom:12px'>What this optimizer does</div>
          <ul style='font-size:13px; color:{MUTED}; line-height:1.8'>
            <li>Schedules battery charge/discharge over a 24-hour horizon</li>
            <li>Shifts flexible loads into low-cost or low-carbon windows</li>
            <li>Minimizes electricity cost, carbon emissions, or peak grid demand</li>
            <li>Uses PuLP linear programming with the HiGHS solver</li>
            <li>Respects all physical constraints: SoC bounds, power limits, energy balance</li>
          </ul>
          <div class="badge-warn" style='margin-top:8px'>Battery specs, tariffs, and carbon intensity are user scenario inputs — not measured data</div>
        </div>
        """, unsafe_allow_html=True)
