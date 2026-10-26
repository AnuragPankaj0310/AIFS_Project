"""
Shared styles and utilities for all dashboard pages.
Each page can call inject_css() to ensure the design system is applied
whether the page is run standalone or from dashboard.py.
"""

import streamlit as st


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Main background */
    .main { background: #0f172a; }
    [data-testid="stAppViewContainer"] { background: #0f172a; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s;
        margin-bottom: 0.5rem;
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

    /* General text color in dark theme */
    h1, h2, h3, h4, h5, p, label { color: #e2e8f0; }
    .stMarkdown { color: #e2e8f0; }
</style>
"""


def inject_css():
    """Inject shared CSS into the current Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_config(title: str, icon: str = "⚡"):
    """Set page config — safe to call multiple times (Streamlit ignores repeats after first)."""
    try:
        st.set_page_config(
            page_title=f"{title} | Solar AI",
            page_icon=icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass  # Already set by dashboard.py
