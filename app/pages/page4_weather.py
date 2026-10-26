"""
Page 4: Weather Analysis
"""
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", family="Inter"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155"),
    legend=dict(bgcolor="#0f172a", bordercolor="#334155"),
    margin=dict(l=20, r=20, t=40, b=20),
)


@st.cache_data(ttl=300)
def load_european_weather(country: str) -> pd.DataFrame | None:
    path = PROCESSED / f"weather_{country}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_plant_weather(plant: int) -> pd.DataFrame | None:
    path = PROCESSED / f"solar_weather_plant{plant}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    return df


@st.cache_data(ttl=300)
def load_plant_generation(plant: int) -> pd.DataFrame | None:
    path = PROCESSED / f"solar_generation_plant{plant}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    return df


def render():
    st.markdown('<h1 class="section-header">🌤️ Weather Analysis</h1>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["☀️ Plant Weather Sensors", "🌍 European Weather Data"])

    # ── Tab 1: Plant-level weather sensors ───────────────────────────────────
    with tab1:
        st.markdown("""
        Plant weather sensors are co-located with solar panels in India.
        These are the weather variables used for solar generation forecasting.
        """)
        plant = st.selectbox("Select Plant", [1, 2], key="wx_plant")
        ws = load_plant_weather(plant)
        gen = load_plant_generation(plant)

        if ws is None:
            st.warning("⚠️ Plant weather data not preprocessed. Run: `python -m src.data.preprocessing`")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                avg_t = ws["AMBIENT_TEMPERATURE"].mean()
                st.metric("Avg Ambient Temp", f"{avg_t:.1f}°C")
            with c2:
                avg_irr = ws["IRRADIATION"].mean()
                st.metric("Avg Irradiation", f"{avg_irr:.3f} W/m²")
            with c3:
                if "MODULE_TEMPERATURE" in ws.columns:
                    avg_mod = ws["MODULE_TEMPERATURE"].mean()
                    st.metric("Avg Module Temp", f"{avg_mod:.1f}°C")

            # Temperature over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ws["DATE_TIME"], y=ws["AMBIENT_TEMPERATURE"],
                name="Ambient Temp (°C)", line=dict(color="#60a5fa", width=1.5),
            ))
            if "MODULE_TEMPERATURE" in ws.columns:
                fig.add_trace(go.Scatter(
                    x=ws["DATE_TIME"], y=ws["MODULE_TEMPERATURE"],
                    name="Module Temp (°C)", line=dict(color="#f87171", width=1.5),
                ))
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Plant {plant} Temperature Profile")
            fig.update_yaxes(title="°C")
            st.plotly_chart(fig, use_container_width=True)

            # Irradiation
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=ws["DATE_TIME"], y=ws["IRRADIATION"],
                name="Irradiation (W/m²)",
                line=dict(color="#f59e0b", width=1.5),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.15)",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, title="Solar Irradiation at Plant Sensor")
            fig2.update_yaxes(title="W/m²")
            st.plotly_chart(fig2, use_container_width=True)

            # Irradiation vs Generation scatter
            if gen is not None:
                merged = ws.merge(gen[["DATE_TIME", "AC_POWER"]], on="DATE_TIME", how="inner")
                merged = merged.dropna(subset=["IRRADIATION", "AC_POWER"])
                merged_day = merged[merged["IRRADIATION"] > 0]

                if len(merged_day) > 0:
                    st.subheader("☀️ Irradiation vs Solar Generation")
                    fig3 = go.Figure(go.Scatter(
                        x=merged_day["IRRADIATION"],
                        y=merged_day["AC_POWER"],
                        mode="markers",
                        marker=dict(
                            color=merged_day["AMBIENT_TEMPERATURE"],
                            colorscale="RdYlBu_r",
                            size=3, opacity=0.6,
                            colorbar=dict(title="Temp (°C)"),
                        ),
                        name="Irrad vs Power",
                    ))
                    fig3.update_layout(**PLOTLY_LAYOUT, title="Irradiation vs AC Power (colored by temperature)")
                    fig3.update_xaxes(title="Irradiation (W/m²)")
                    fig3.update_yaxes(title="AC Power (kW)")
                    st.plotly_chart(fig3, use_container_width=True)

                    corr = merged_day[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "AC_POWER"]].corr()
                    st.subheader("📊 Correlation Matrix")
                    fig4 = px.imshow(
                        corr, text_auto=True, color_continuous_scale="RdBu",
                        aspect="auto",
                    )
                    fig4.update_layout(paper_bgcolor="#0f172a", font=dict(color="#e2e8f0"),
                                       margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 2: European weather ────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        Open Power System Data (NASA MERRA-2 reanalysis), hourly, 1980–2019.
        Country-level aggregates (population-weighted). Used for European demand forecasting.
        """)
        country = st.selectbox("Country", ["DE", "GB", "FR", "ES", "IT", "NL"], key="eu_wx_country")
        df = load_european_weather(country)

        if df is None:
            st.warning("⚠️ European weather not preprocessed. Run: `python -m src.data.preprocessing`")
        else:
            avail = [c for c in df.columns if c not in ["timestamp", "country"]]
            st.info(f"Available weather variables for {country}: {', '.join(avail)}")

            # Temperature
            if "temperature" in df.columns:
                df_plot = df[["timestamp", "temperature"]].dropna().tail(8760)
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(
                    x=df_plot["timestamp"], y=df_plot["temperature"],
                    name="Temperature (°C)", line=dict(color="#60a5fa", width=1),
                    fill="tozeroy", fillcolor="rgba(96,165,250,0.1)",
                ))
                fig5.update_layout(**PLOTLY_LAYOUT, title=f"{country} Temperature – Last Year")
                fig5.update_yaxes(title="°C")
                st.plotly_chart(fig5, use_container_width=True)

            # Radiation
            rad_cols = [c for c in avail if "radiation" in c.lower()]
            if rad_cols:
                fig6 = go.Figure()
                colors_rad = ["#f59e0b", "#fde68a"]
                for i, rc in enumerate(rad_cols):
                    df_r = df[["timestamp", rc]].dropna().tail(8760)
                    fig6.add_trace(go.Scatter(
                        x=df_r["timestamp"], y=df_r[rc],
                        name=rc.replace("_", " "),
                        line=dict(color=colors_rad[i % 2], width=1),
                    ))
                fig6.update_layout(**PLOTLY_LAYOUT, title=f"{country} Solar Radiation")
                fig6.update_yaxes(title="W/m²")
                st.plotly_chart(fig6, use_container_width=True)

            # Seasonality: monthly averages
            if "temperature" in df.columns:
                df_m = df[["timestamp", "temperature"]].copy().dropna()
                df_m["month"] = df_m["timestamp"].dt.month
                monthly = df_m.groupby("month")["temperature"].mean()
                fig7 = go.Figure(go.Bar(
                    x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                    y=monthly.values,
                    marker_color="#60a5fa",
                ))
                fig7.update_layout(**PLOTLY_LAYOUT, title=f"{country} – Average Monthly Temperature (all years)")
                fig7.update_yaxes(title="°C")
                st.plotly_chart(fig7, use_container_width=True)

            st.caption("""
            **Data note**: Weather data spans 1980–2019. For demand forecasting alignment,
            only 2015–2019 overlap with ENTSOE data is used.
            """)
