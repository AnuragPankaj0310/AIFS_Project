"""
Solar AI Sustainability Dashboard
===================================
Multi-page Streamlit application for energy forecasting and optimization.

Run:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Page configuration
st.set_page_config(
    page_title="Solar AI Sustainability Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    .css-1d391kg { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }

    /* Main background */
    .main { background: #0f172a; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #38bdf8; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-delta { font-size: 0.9rem; margin-top: 0.3rem; }
    .metric-positive { color: #4ade80; }
    .metric-negative { color: #f87171; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    /* Assumption badge */
    .assumption-badge {
        background: #422006;
        border: 1px solid #f59e0b;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        color: #fbbf24;
    }

    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
    }

    /* Selectbox */
    .stSelectbox select { background: #1e293b; color: white; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e293b;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0'>
        <div style='font-size: 2.5rem'>⚡</div>
        <div style='font-size: 1.2rem; font-weight: 700; color: #38bdf8'>Solar AI</div>
        <div style='font-size: 0.8rem; color: #64748b'>Sustainability Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=[
            "🏠 Energy Overview",
            "☀️ Solar Forecasting",
            "⚡ Demand Forecasting",
            "🌤️ Weather Analysis",
            "🔋 Energy Optimization",
            "🌱 Sustainability Report",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.75rem; color: #475569; text-align: center'>
        v1.0 · Building-to-Grid DR<br>
        Powered by PuLP + XGBoost
    </div>
    """, unsafe_allow_html=True)

# ── Route to pages ──────────────────────────────────────────────────────────
if page == "🏠 Energy Overview":
    from app.pages.page1_overview import render
elif page == "☀️ Solar Forecasting":
    from app.pages.page2_solar import render
elif page == "⚡ Demand Forecasting":
    from app.pages.page3_demand import render
elif page == "🌤️ Weather Analysis":
    from app.pages.page4_weather import render
elif page == "🔋 Energy Optimization":
    from app.pages.page5_optimization import render
elif page == "🌱 Sustainability Report":
    from app.pages.page6_sustainability import render
else:
    from app.pages.page1_overview import render

render()
