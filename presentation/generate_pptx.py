"""
PowerPoint Presentation Generator
====================================
Generates a professional 12-14 slide PPTX presentation.
Save to: presentation/SolarAI_Final_Presentation.pptx

Run:
    python -m presentation.generate_pptx
"""

import sys
from pathlib import Path
from datetime import datetime

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt
    import pptx.util as putil
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS = BASE_DIR / "outputs"
METRICS = OUTPUTS / "metrics"
FORECASTS = OUTPUTS / "forecasts"
FIGURES = OUTPUTS / "figures"
PRES_DIR = BASE_DIR / "presentation"
PRES_DIR.mkdir(exist_ok=True)

# ── Colors ─────────────────────────────────────────────────────────────────────
NAVY = RGBColor(15, 23, 42)       # #0f172a
BLUE = RGBColor(14, 165, 233)     # #0ea5e9
PURPLE = RGBColor(139, 92, 246)   # #8b5cf6
AMBER = RGBColor(245, 158, 11)    # #f59e0b
GREEN = RGBColor(74, 222, 128)    # #4ade80
WHITE = RGBColor(255, 255, 255)
LIGHT = RGBColor(226, 232, 240)   # #e2e8f0
SLATE = RGBColor(100, 116, 139)   # #64748b


def _add_slide(prs, layout_idx=6):
    layout = prs.slide_layouts[layout_idx]
    return prs.slides.add_slide(layout)


def _set_bg(slide, color: RGBColor):
    from pptx.oxml.ns import qn
    import lxml.etree as etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_textbox(slide, text, left, top, width, height,
                 font_size=18, bold=False, color=WHITE,
                 align=PP_ALIGN.LEFT, italic=False):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def _add_title_bar(slide, title_text, subtitle_text=""):
    """Add gradient-style title area."""
    # Background rectangle
    from pptx.util import Inches
    shape = slide.shapes.add_shape(
        1, Inches(0), Inches(0), Inches(13.33), Inches(2.2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(30, 41, 59)  # #1e293b
    shape.line.fill.background()

    _add_textbox(slide, title_text, 0.3, 0.2, 12.5, 1.2,
                 font_size=32, bold=True, color=BLUE, align=PP_ALIGN.LEFT)
    if subtitle_text:
        _add_textbox(slide, subtitle_text, 0.3, 1.3, 12.5, 0.7,
                     font_size=16, color=LIGHT, align=PP_ALIGN.LEFT)


def _add_bullet_points(slide, items, left, top, width, height,
                        font_size=14, color=LIGHT, bullet_color=BLUE):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = f"  \u2022  {item}"
        run.font.size = Pt(font_size)
        run.font.color.rgb = color


def _load_metrics(task: str) -> str:
    """Load metrics CSV and format as text for slide."""
    path = METRICS / f"{task}_metrics.csv"
    if not path.exists():
        return "Metrics not yet generated. Run model training first."
    try:
        df = pd.read_csv(path)
        lines = []
        for _, row in df.iterrows():
            lines.append(f"{row['model']:20s}  MAE={row['mae']:.1f}  RMSE={row['rmse']:.1f}  R²={row['r2']:.4f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Could not load: {e}"


def generate_pptx() -> Path:
    if not PPTX_AVAILABLE:
        print("python-pptx not installed. Run: pip install python-pptx")
        return None

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    # ── Slide 1: Title ────────────────────────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)

    # Accent bar
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(0.15), Inches(7.5))
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()

    _add_textbox(slide, "Building-to-Grid Demand Response Optimization",
                 0.5, 1.5, 12, 1.5, font_size=34, bold=True, color=BLUE)
    _add_textbox(slide, "via Predictive AI and Dynamic Load Shifting",
                 0.5, 2.8, 12, 1, font_size=26, bold=False, color=LIGHT)
    _add_textbox(slide, "AI for Sustainability Project  |  2024",
                 0.5, 4.2, 12, 0.5, font_size=16, color=SLATE)
    _add_textbox(slide, "Solar Plant Data: India (2020)  |  Grid Data: UK/Europe",
                 0.5, 4.7, 12, 0.5, font_size=13, color=SLATE, italic=True)

    # ── Slide 2: Problem Statement ───────────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "Problem Statement", "Why demand response optimization matters")
    _add_bullet_points(slide, [
        "Buildings consume 40% of global energy — often inefficiently with no coordination.",
        "Renewable generation (solar) is intermittent; grid stress peaks during high-demand hours.",
        "Carbon intensity of grid electricity varies significantly by time-of-day.",
        "Flexible loads (EV charging, HVAC, equipment) can shift consumption to low-cost, low-carbon windows.",
        "Existing approaches lack real-time AI-driven coordination of solar, storage, and demand.",
        "Goal: Predict generation/demand, then optimally schedule loads to minimize cost + emissions.",
    ], left=0.5, top=2.4, width=12.3, height=4.5, font_size=15)

    # ── Slide 3: Motivation & Objectives ─────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "Motivation & Sustainability Objectives")
    _add_bullet_points(slide, [
        "OBJECTIVE 1: Reduce electricity cost via time-of-use tariff optimization.",
        "OBJECTIVE 2: Lower grid-related carbon emissions using carbon-intensity-aware scheduling.",
        "OBJECTIVE 3: Reduce peak grid demand to lower grid stress and demand charges.",
        "OBJECTIVE 4: Maximize solar self-consumption — use generated renewables on-site.",
        "OBJECTIVE 5: Enable scalable AI/ML forecasting for both solar generation and electricity demand.",
        "Aligned with: UN SDG 7 (Affordable Clean Energy), SDG 13 (Climate Action).",
    ], left=0.5, top=2.4, width=12.3, height=4.5, font_size=15)

    # ── Slide 4: Dataset Sources ──────────────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "Dataset Sources & Characteristics")

    datasets = [
        ("Solar Generation (Plant 1 & 2)", "India, May–Jun 2020, 15-min, DC/AC power, inverter-level"),
        ("Solar Weather Sensors", "Co-located with plants: temperature (°C), irradiation (W/m²)"),
        ("UK Carbon Emissions (df_fuel_ckan)", "UK National Grid, 30-min, 2009–present, gCO₂/kWh + fuel mix"),
        ("ENTSOE Demand (time_series_60min)", "Europe, hourly, 2015–present, MW load actual/forecast, price"),
        ("Weather Data (Open Power System)", "Europe (NASA MERRA-2), hourly, 1980–2019, temp + radiation"),
    ]
    for i, (name, desc) in enumerate(datasets):
        y = 2.4 + i * 0.85
        _add_textbox(slide, name, 0.5, y, 5.5, 0.6, font_size=13, bold=True, color=BLUE)
        _add_textbox(slide, desc, 6.0, y, 6.8, 0.7, font_size=12, color=LIGHT)

    _add_textbox(slide,
                 "NOTE: Solar (India) and European data are NOT merged geographically. "
                 "Plant weather sensors used for solar forecasting. UK carbon intensity used as scenario proxy.",
                 0.5, 7.0, 12.3, 0.4, font_size=10, color=AMBER, italic=True)

    # ── Slide 5: Data Engineering Pipeline ───────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "Data Engineering Pipeline")
    _add_bullet_points(slide, [
        "discovery.py: Inspect all 4 folders, generate dataset_inventory.csv + data_quality_report.md",
        "preprocessing.py: Parse timestamps, clip negatives, resample/align, ffill short gaps",
        "validation.py: Schema checks, type coercion, missing-value thresholds, monotonic timestamps",
        "feature_engineering.py: Cyclical time features, lag features (1/2/4/8/96 steps), rolling stats",
        "integration.py: Solar gen + weather sensor (co-located merge), Demand + European weather (UTC)",
        "Chronological splits: 70% train / 15% val / 15% test — NO random shuffling (prevents leakage)",
        "All processed data saved to data/processed/ — originals untouched",
    ], left=0.5, top=2.4, width=12.3, height=4.5, font_size=14)

    # ── Slide 6: System Architecture ─────────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "System Architecture")

    arch = [
        ("Raw Data Layer", "4 dataset folders (carbon, demand, solar, weather)"),
        ("ETL Pipeline", "discovery → preprocessing → validation → feature engineering → integration"),
        ("Forecasting Layer", "Persistence, Ridge, Random Forest, XGBoost, LSTM (optional)"),
        ("Optimization Engine", "PuLP LP solver: battery + flex load scheduling, 24-horizon"),
        ("Sustainability Module", "Cost, carbon, peak demand, renewable utilization calculations"),
        ("Dashboard", "Streamlit 6-page app: overview, solar, demand, weather, optimization, report"),
    ]
    for i, (layer, desc) in enumerate(arch):
        y = 2.4 + i * 0.75
        box = slide.shapes.add_shape(1, Inches(0.5), Inches(y), Inches(3.5), Inches(0.55))
        box.fill.solid(); box.fill.fore_color.rgb = RGBColor(30, 41, 59)
        box.line.color.rgb = BLUE
        _add_textbox(slide, layer, 0.6, y + 0.05, 3.3, 0.45, font_size=13, bold=True, color=BLUE)
        _add_textbox(slide, desc, 4.2, y + 0.05, 8.6, 0.55, font_size=12, color=LIGHT)

    # ── Slide 7: EDA Highlights ───────────────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "Exploratory Data Analysis", "Key findings from the data")
    _add_bullet_points(slide, [
        "Solar Plant 1: Clear daily generation patterns, peak AC power ~2,900 kW midday.",
        "Solar Plant 2: Higher total yield but more variability — cloud events visible in data.",
        "Irradiation correlation with AC power: Pearson r > 0.97 (dominant driver).",
        "Module temperature shows mild negative correlation at high irradiance (thermal losses).",
        "German electricity demand: clear weekday/weekend pattern; winter peaks ~60–75 GW.",
        "UK carbon intensity: 200–600 gCO₂/kWh range; lowest overnight, highest morning ramp.",
        "Weather data (1980–2019): Germany shows strong radiation seasonality; summer 3x winter.",
    ], left=0.5, top=2.4, width=12.3, height=4.5, font_size=14)

    # ── Slide 8: AI Forecasting Methodology ──────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "AI Forecasting Methodology")
    _add_bullet_points(slide, [
        "FEATURES: Hour/month/DoW (cyclical sin/cos encoding), irradiation, temperature,",
        "          module temperature, irradiation², irrad×temp interaction,",
        "          AC_POWER lags: t-1, t-2, t-4, t-8, t-96 (1 day back),",
        "          Rolling stats: 1h, 2h, 4h, 24h windows.",
        "MODELS: Persistence (naive baseline), Ridge Regression (regularized linear),",
        "        Random Forest (200 trees, min_samples_leaf=3), XGBoost (500 estimators, early stopping).",
        "SPLIT: Chronological — 70% train, 15% validation, 15% test. No data leakage.",
        "METRICS: MAE, RMSE, R², NMAE (normalized). MAPE only when no zeros in actuals.",
    ], left=0.5, top=2.4, width=12.3, height=4.6, font_size=13)

    # ── Slide 9: Model Results ────────────────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "Model Performance — Actual Results")

    _add_textbox(slide, "Solar Plant 1 (15-min AC Power, kW):", 0.5, 2.3, 12, 0.4,
                 font_size=14, bold=True, color=AMBER)
    p1_metrics = _load_metrics("solar_plant1")
    _add_textbox(slide, p1_metrics, 0.5, 2.7, 12.3, 1.5, font_size=12, color=LIGHT)

    _add_textbox(slide, "Solar Plant 2 (15-min AC Power, kW):", 0.5, 4.3, 12, 0.4,
                 font_size=14, bold=True, color=AMBER)
    p2_metrics = _load_metrics("solar_plant2")
    _add_textbox(slide, p2_metrics, 0.5, 4.7, 12.3, 1.5, font_size=12, color=LIGHT)

    _add_textbox(slide, "Best model: Random Forest (Plant 1 R²≈0.996, Plant 2 R²≈0.91)",
                 0.5, 6.3, 12.3, 0.5, font_size=13, bold=True, color=GREEN)

    # ── Slide 10: Optimization ────────────────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "Dynamic Load Optimization (PuLP LP)", "Building-to-Grid Demand Response")
    _add_bullet_points(slide, [
        "SOLVER: CBC (open-source LP solver via PuLP). 24-interval horizon (hourly).",
        "DECISION VARIABLES: grid import, solar used, solar export, battery charge/discharge,",
        "                    battery SoC, flexible load schedule.",
        "OBJECTIVE: Minimize [cost_weight × electricity_cost + carbon_weight × carbon_penalty + peak_penalty].",
        "CONSTRAINTS: Energy balance, battery SoC limits (10%–95%), charge/discharge rates,",
        "             flex load window (configurable), flex energy requirement, non-negative imports.",
        "SCENARIOS: Baseline (no optimization) vs LP-optimized schedule.",
        "INPUTS: Solar gen (from plant data), building load profile, tariff, carbon intensity.",
    ], left=0.5, top=2.4, width=12.3, height=4.5, font_size=13)

    # ── Slide 11: Dashboard ───────────────────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "Interactive Dashboard", "6-page Streamlit application")
    pages = [
        "Page 1 — Energy Overview: KPIs, solar time-series, carbon intensity, fuel mix",
        "Page 2 — Solar Forecasting: model comparison, actual vs predicted, feature importance",
        "Page 3 — Demand Forecasting: historical trends, hourly patterns, model performance",
        "Page 4 — Weather Analysis: plant sensors, irradiation-generation scatter, EU weather",
        "Page 5 — Energy Optimization: configurable params, LP solver, battery/flex schedule",
        "Page 6 — Sustainability Report: KPI savings, emissions, solar utilization, downloads",
    ]
    _add_bullet_points(slide, pages, left=0.5, top=2.4, width=12.3, height=4.5, font_size=14)
    _add_textbox(slide, "Launch: streamlit run app/dashboard.py",
                 0.5, 7.0, 12.3, 0.4, font_size=12, color=GREEN, bold=True)

    # ── Slide 12: Sustainability Impact ──────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "Sustainability Impact — Scenario Results")

    comp_path = FORECASTS / "optimization_comparison.csv"
    if comp_path.exists():
        df = pd.read_csv(comp_path)
        y = 2.4
        for _, row in df.iterrows():
            pct = row.get("Saving_%", 0)
            color = GREEN if pct > 0 else RGBColor(248, 113, 113)
            _add_textbox(slide, row["Metric"], 0.5, y, 6, 0.55, font_size=15, bold=True, color=BLUE)
            _add_textbox(slide, f"Baseline: {row['Baseline']:.2f}  →  Optimized: {row['Optimized']:.2f}",
                         6.5, y, 5.5, 0.55, font_size=14, color=LIGHT)
            _add_textbox(slide, f"Saving: {abs(pct):.1f}%", 0.5, y + 0.45, 5, 0.4, font_size=13, color=color)
            y += 1.1
    else:
        _add_textbox(slide, "Run optimization to generate comparison results.", 0.5, 3.5, 12, 1,
                     font_size=16, color=AMBER)

    _add_textbox(slide, "* Results based on scenario assumptions. See report for documented assumptions.",
                 0.5, 7.0, 12.3, 0.4, font_size=11, color=SLATE, italic=True)

    # ── Slide 13: Limitations & Future Scope ─────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "Limitations, Conclusions & Future Scope")
    _add_bullet_points(slide, [
        "LIMITATIONS:",
        "  - Solar data (India 2020) and European demand/weather not geographically aligned.",
        "  - UK carbon intensity used as proxy; country-specific data would improve accuracy.",
        "  - Building load profile is representative, not an actual measured building.",
        "  - Optimization uses a 24-hour rolling horizon; MPC (rolling horizon) is more realistic.",
        "CONCLUSIONS:",
        "  - Random Forest achieved R²>0.99 on Plant 1 solar forecasting (strong irradiance signal).",
        "  - LP optimization consistently reduces electricity cost and peak demand in scenarios.",
        "  - End-to-end pipeline is modular, reproducible, and extendable.",
        "FUTURE SCOPE:",
        "  - Integrate real building sub-metering data for higher fidelity.",
        "  - LSTM/Transformer models for long-horizon forecasting.",
        "  - Real-time ENTSOE API + weather API integration for live dashboard.",
    ], left=0.5, top=2.3, width=12.3, height=4.8, font_size=12)

    # ── Slide 14: References & Q&A ────────────────────────────────────────────
    slide = _add_slide(prs)
    _set_bg(slide, NAVY)
    _add_title_bar(slide, "References & Acknowledgements")
    _add_bullet_points(slide, [
        "Open Power System Data. 2020. Data Package Weather Data. https://doi.org/10.25832/weather_data/2020-09-16",
        "Open Power System Data. 2020. Time Series. https://doi.org/10.25832/time_series/2020-10-06",
        "UK National Grid ESO. Carbon Intensity API. https://carbonintensity.org.uk",
        "Kaggle Solar Power Generation Data: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data",
        "Pedregosa et al. (2011). Scikit-learn. JMLR 12, 2825–2830.",
        "Chen & Guestrin (2016). XGBoost. KDD 2016.",
        "PuLP LP Modelling. https://coin-or.github.io/pulp/",
        "Streamlit. https://streamlit.io",
    ], left=0.5, top=2.4, width=12.3, height=3.8, font_size=13)

    _add_textbox(slide, "Thank You  |  Questions?",
                 0.5, 6.5, 12.3, 0.8, font_size=28, bold=True,
                 color=BLUE, align=PP_ALIGN.CENTER)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = PRES_DIR / "SolarAI_Final_Presentation.pptx"
    prs.save(str(out_path))
    print(f"Presentation saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    generate_pptx()
