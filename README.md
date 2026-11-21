# Building-to-Grid Demand Response Optimization
## via Predictive AI and Dynamic Load Shifting

> **AI for Sustainability (AIFS) — Group Project**
> An end-to-end machine-learning pipeline integrating solar PV forecasting, electricity demand forecasting, SVR with similar-day averaging (paper replication), LP-based battery/load optimization, AC set-point demand response, and an interactive Streamlit dashboard.

---

## 👥 Group Members

| Name | Role |
|---|---|
| Ramavath Pavan | u23AI130 |
| Vishwakarma Ashish Diwakar| U3AI129 |
| Anurag Kumar Pankaj | U3AI126 |

---

## 🔗 Repository

```
https://github.com/AnuragPankaj0310/AIFS_Project
```

**Clone the repo:**
```bash
git clone https://github.com/AnuragPankaj0310/AIFS_Project.git
cd AIFS_Project
```

---

## 🗏 Project Structure

```
AIFS PROJECT/
├── app/
│   ├── dashboard.py              # Streamlit entry point
│   └── pages/
│       ├── page1_overview.py     # Energy overview & KPIs
│       ├── page2_solar.py        # Solar forecasting (all models incl. SVR)
│       ├── page3_demand.py       # Demand forecasting
│       ├── page4_weather.py      # Weather analysis
│       ├── page5_optimization.py # LP optimizer + AC Set-Point optimizer (paper)
│       └── page6_sustainability.py # Sustainability report
├── src/
│   ├── data/
│   │   ├── discovery.py          # Dataset inventory
│   │   ├── preprocessing.py      # ETL pipeline
│   │   ├── validation.py         # Schema validation
│   │   ├── feature_engineering.py # Feature creation + chronological split
│   │   └── integration.py        # Multi-source dataset merging
│   ├── models/
│   │   └── forecasting.py        # Persistence, Ridge, RF, XGBoost, SVR, LSTM
│   ├── optimization/
│   │   ├── optimizer.py          # PuLP LP demand response optimization
│   │   └── setpoint_optimizer.py # AC set-point optimizer (paper method)
│   └── sustainability/
│       └── analysis.py           # Cost, carbon, renewables calculations
├── tests/
│   └── test_suite.py             # Unit tests
├── presentation/
│   └── generate_pptx.py          # Auto PPTX generator
├── data/
│   ├── raw/                      # Original datasets (untouched)
│   └── processed/                # Preprocessed CSVs
├── outputs/
│   ├── forecasts/                # Model predictions + optimization results
│   ├── metrics/                  # Per-model performance metrics CSVs
│   └── reports/                  # Sustainability markdown reports
└── models/                       # Serialized trained models (.pkl / .joblib)
```

---

## 📊 Datasets Used

| Dataset | Source | Format | Resolution | Key Variables |
|---|---|---|---|---|
| Solar Generation — Plant 1 & 2 | [Kaggle — Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) | CSV | 15-min | AC Power, DC Power, Daily Yield, Total Yield (kW/kWh) |
| Solar Weather Sensors | Co-located with plants (Kaggle) | CSV | 15-min | Ambient Temp (°C), Module Temp (°C), Irradiation (W/m²) |
| UK Carbon Emissions | [UK National Grid ESO — Carbon Intensity API](https://carbonintensity.org.uk) | CSV | 30-min | gCO₂/kWh, fuel mix (gas/coal/wind/solar/nuclear) |
| European Electricity Demand | [Open Power System Data — ENTSOE](https://doi.org/10.25832/time_series/2020-10-06) | CSV | Hourly | MW load actual/forecast, EUR/MWh price |
| European Weather | [Open Power System Data — NASA MERRA-2](https://doi.org/10.25832/weather_data/2020-09-16) | CSV | Hourly | Temperature (°C), Radiation direct/diffuse (W/m²), 1980–2019 |

> ⚠️ **Geographic note:** Solar plant data is from **India** (May–June 2020) and is used only for solar generation forecasting. European demand/weather data covers **Germany (DE)** for demand forecasting. These two geographic domains are **not merged**.

### Download Instructions

1. **Solar Generation & Weather:** Download from Kaggle → place `Plant_1_Generation_Data.csv`, `Plant_1_Weather_Sensor_Data.csv`, `Plant_2_Generation_Data.csv`, `Plant_2_Weather_Sensor_Data.csv` into `Solar Panel Electricity Generation/`
2. **UK Carbon Emissions:** Download `df_fuel_ckan.csv` from the Carbon Intensity API historical data → place in `UK Carbon Intensity/`
3. **European Electricity Demand:** Download `time_series_60min_singleindex.csv` from OPSD → place in `European Electricity Demand/`
4. **European Weather:** Download `weather_data.csv` from OPSD → place in `Weather and solar energy/`

---

## 🛠️ Implementation Environment

### System Requirements

| Component | Version |
|---|---|
| **Operating System** | Windows 11 (10.0.26300) |
| **Python** | 3.14.6 |
| **Package Manager** | pip 26.1.2 |

### Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | 3.1.0 | Data manipulation & time-series handling |
| `numpy` | 2.5.3 | Numerical computing |
| `scikit-learn` | 1.9.1 | Ridge, Random Forest, SVR, StandardScaler, KMeans, GridSearchCV |
| `xgboost` | 3.4.1 | XGBoost Regressor |
| `tensorflow` | 2.22.0rc0 | LSTM (optional, `use_lstm=True`) |
| `keras` | 3.15.1 | Keras backend for LSTM |
| `PuLP` | 4.2.0 | Linear programming (LP) optimization |
| `highspy` | 1.15.1 | HiGHS solver backend for PuLP |
| `streamlit` | 1.66.1 | Interactive dashboard |
| `plotly` | 7.1.0 | Interactive charts |
| `joblib` | 1.6.0 | Model serialization |
| `scipy` | 1.18.1 | Statistical utilities |
| `matplotlib` | 3.11.2 | Static plots |
| `seaborn` | 0.13.2 | Statistical visualization |
| `python-pptx` | 1.0.2 | PowerPoint generation |
| `openpyxl` | 3.1.5 | Excel file support |
| `pyarrow` | 26.0.0 | Parquet / fast I/O |
| `tabulate` | 0.10.0 | Table formatting |

### Installation

```bash
pip install pandas numpy scikit-learn xgboost pulp highspy streamlit plotly \
            joblib scipy matplotlib seaborn python-pptx openpyxl pyarrow \
            tabulate tensorflow
```

> **Solver:** PuLP 4.x uses HiGHS by default (`pip install highspy`)
> **LSTM:** TensorFlow 2.x required only if running with `use_lstm=True`

---

## 🚀 Quick Start — Full Pipeline

Run these steps **once** in order to process raw data, train all models, and produce all outputs:

```bash
# Step 1: Dataset discovery (inspect all raw data folders)
python -m src.data.discovery

# Step 2: Preprocessing & ETL (clean, parse, normalise all datasets)
python -m src.data.preprocessing

# Step 3: Integration (merge solar+weather, demand+weather)
python -m src.data.integration

# Step 4: Feature engineering (lag features, rolling stats, cyclical encoding)
python -m src.data.feature_engineering

# Step 5: Train all forecasting models (Persistence, Ridge, RF, XGBoost, SVR)
python -m src.models.forecasting

# Step 6: Run LP battery & load optimization demo
python -m src.optimization.optimizer

# Step 7: Run AC set-point optimization demo (paper method)
python -m src.optimization.setpoint_optimizer
```

### Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

### Run Tests

```bash
python -m pytest tests/test_suite.py -v
```

### Regenerate the PowerPoint Presentation

```bash
python -m presentation.generate_pptx
```

---

## 🤖 Models & Methods

### 1. Forecasting Models

All models use **chronological train/validation/test splits** (70% / 15% / 15%) — no random shuffling, preventing data leakage.

| Model | Description |
|---|---|
| **Persistence** | Naïve baseline: ŷ(t) = y(t−1) |
| **Linear Ridge** | Ridge regression with StandardScaler; α=1.0 |
| **Random Forest** | 200 estimators, min_samples_leaf=3, n_jobs=-1 |
| **XGBoost** | 500 estimators, lr=0.05, max_depth=6, early stopping (30 rounds) |
| **SVR + Similar-Day** | RBF kernel, grid-search over C/ε/γ, + similar-day average feature (paper method) |
| **LSTM** *(optional)* | 2-layer stacked LSTM (64→32 units), 20% Dropout, lookback=96 steps, 10 epochs |

### 2. SVR with Similar-Day Averaging (Paper Method)

**Reference:** Houchati, Beitelmal & Khraisheh (2022), *"Predictive Modeling for Rooftop Solar Energy Throughput"*, ASME J. Energy Resour. Technol., 144(1).

The paper method implemented in `src/models/forecasting.py`:

1. **K-means sky classification** — days grouped into 4 clusters (foggy / cloudy / sunny / rainy) based on daily-mean weather features.
2. **N=3 similar-day matching** — for each target day, find the 3 nearest historical training days (Euclidean distance on weather features, within the same sky cluster).
3. **Similar-day average feature** — the hourly mean of those 3 days' target values becomes an extra feature column fed to SVR.
4. **SVR with RBF (Gaussian) kernel** — hyperparameters C, ε, γ grid-searched (C∈{1,10}, ε∈{0.1,0.5}, γ∈{scale,0.1}).

### 3. AC Set-Point Optimizer (Paper Method)

Implemented in `src/optimization/setpoint_optimizer.py`. Replicates the demand-response control rule from the paper:

```
For each hour t:
  If load[t] > pv[t]:           → shortfall exists, act
    If T_outdoor > T_indoor:    → cooling is needed
      If T_indoor < 24°C:       → comfort not violated
        Raise set point by Δ°C → re-evaluate AC power
```

**Thermal model:** `P_AC = k · (T_out − T_in) / R + b` (linearised steady-state heat balance from paper).

Outputs an hourly AC set-point schedule that reduces grid draw while keeping indoor temperature below the comfort ceiling.

### 4. LP Battery & Load Optimizer

Implemented in `src/optimization/optimizer.py` using **PuLP + HiGHS solver**.

**Decision variables:** grid import, solar used/exported, battery charge/discharge, state of charge (SoC), flexible load schedule.

**Objective:** Minimize `Σ_t [ tariff(t)·grid_import(t) + carbon_intensity(t)·grid_import(t)·w_carbon + w_peak·peak_demand ]`

**Constraints:** Energy balance, battery SoC dynamics (charge/discharge efficiency), capacity limits, flexible load window and total energy requirement.

---

## 📈 Model Performance — Actual Results

### Solar Generation Forecasting — Plant 1 (India, 15-min AC Power, kW)

| Model | MAE (kW) | RMSE (kW) | R² | NMAE |
|---|---|---|---|---|
| Persistence | 1218.7 | 2325.3 | 0.9200 | 0.0420 |
| Linear Ridge | 255.4 | 510.8 | 0.9961 | 0.0088 |
| **Random Forest** | **244.1** | **507.1** | **0.9962** ✅ | **0.0084** |
| XGBoost | 273.3 | 571.0 | 0.9952 | 0.0094 |
| SVR + Similar-Day | 3818.0 | 6402.2 | 0.3938 | 0.1314 |

> SVR best params: `C=10, ε=0.1, γ=scale`

### Solar Generation Forecasting — Plant 2 (India, 15-min AC Power, kW)

| Model | MAE (kW) | RMSE (kW) | R² | NMAE |
|---|---|---|---|---|
| Persistence | 1213.2 | 2347.2 | 0.8264 | 0.0541 |
| Linear Ridge | 838.2 | 1660.4 | 0.9132 | 0.0374 |
| Random Forest | 704.5 | 1694.7 | 0.9095 | 0.0314 |
| **XGBoost** | **712.7** | **1692.0** | **0.9098** ✅ | **0.0318** |
| SVR + Similar-Day | 2304.1 | 3745.1 | 0.5582 | 0.1028 |

> SVR best params: `C=10, ε=0.5, γ=scale`

### Electricity Demand Forecasting — Germany DE (Hourly, MW)

| Model | MAE (MW) | RMSE (MW) | R² | NMAE | MAPE (%) |
|---|---|---|---|---|---|
| Persistence | 1855.0 | 2409.2 | 0.9407 | 0.0425 | 3.55 |
| Linear Ridge | 844.3 | 1107.0 | 0.9875 | 0.0194 | 1.61 |
| Random Forest | 587.7 | 787.1 | 0.9937 | 0.0135 | 1.13 |
| **XGBoost** | **530.5** | **695.1** | **0.9951** ✅ | **0.0122** | **1.03** |
| SVR + Similar-Day | 6244.7 | 7604.7 | 0.4089 | 0.1431 | 12.67 |

> SVR best params: `C=10, ε=0.1, γ=scale` (trained on 5 000-row subsample — SVR is O(n²); full 35 280-row training would require approximate solvers)

> **Why SVR trails here:** The paper achieved >90% accuracy on a small, clean, **sunny-day-only** Qatar dataset (~300 days, 1 building). Our datasets span full years across all sky conditions and 50 000+ hourly demand records. SVR's quadratic scaling makes it impractical on large demand data without subsampling, limiting its accuracy advantage.

---

## ⚡ Optimization Results

### LP Battery & Load Optimization (Example Scenario)

| Metric | Baseline | Optimized | Saving |
|---|---|---|---|
| Total Cost (€) | 153.50 | 156.60 | −2.0% *(cost priority vs carbon)* |
| Carbon Emissions (kg CO₂) | 171.92 | 175.39 | −2.0% |
| **Peak Grid Demand (kW)** | **41.0** | **36.2** | **↓ 11.7%** ✅ |

> *Note: With `cost_weight=1.0, carbon_weight=0.5`, the optimizer prioritizes peak shaving and cost. Increase `carbon_weight` in the dashboard to shift towards emissions reduction.*

### AC Set-Point Optimization (Houchati et al. Method — Sunny Day Scenario)

| Metric | Value |
|---|---|
| Total Load Reduction | 2.75 kWh |
| Load Reduction (%) | 2.8% |
| Grid Draw — Baseline | 51.06 kWh |
| Grid Draw — Optimized | 48.31 kWh |
| **Grid Draw Reduction** | **↓ 5.4%** ✅ |

> Comfort ceiling: 24°C. Set point raised from 23°C during shortfall hours (00–05h, 19–23h). No action needed at solar peak (06–18h). Thermal model: `P_AC = 0.5·ΔT/2.0 + 1.0 kW`.

---

## 🌱 Sustainability Reports

Auto-generated after running the optimizer:

| File | Description |
|---|---|
| `outputs/reports/sustainability_report.md` | Baseline vs optimized comparison with methodology |
| `outputs/reports/full_report.md` | Full report: sustainability + all model metrics |
| `outputs/reports/sustainability_comparison.csv` | Machine-readable comparison table |
| `outputs/forecasts/optimization_comparison.csv` | LP optimizer scenario comparison |
| `outputs/forecasts/setpoint_schedule.csv` | Hourly AC set-point schedule (paper method) |

Regenerate reports:
```bash
python -m src.sustainability.analysis
```

---

## ⚠️ Key Assumptions

| Assumption | Value | Rationale |
|---|---|---|
| Carbon Intensity | 280 gCO₂/kWh | UK National Grid average (proxy; clearly labeled) |
| Building Load Profile | 15–40 kW representative profile | Synthetic residential/commercial |
| Tariff | Configurable (€0.25/kWh default) | User scenario — adjustable in dashboard |
| Battery Capacity | Configurable (100 kWh default) | User scenario — adjustable in dashboard |
| LP Optimizer Horizon | 24 hourly intervals | Single-day rolling LP (MPC extension possible) |
| Solar scaling | Plant data scaled to building level (~80 kW peak) | Documented in dashboard |
| SVR training cap | 5 000 rows (demand only) | SVR is O(n²); subsampling preserves speed |
| AC Set-Point scenario | Qatar sunny-day profile (paper testbed) | Representative; not measured in a real building |

---

## 📄 References

1. **Houchati, M., Beitelmal, A. H., & Khraisheh, M. (2022).** Predictive Modeling for Rooftop Solar Energy Throughput. *ASME Journal of Energy Resources Technology*, 144(1). *(Paper replicated in this project — SVR + similar-day averaging + AC set-point optimizer)*
2. Open Power System Data (2020). *Weather Data*. https://doi.org/10.25832/weather_data/2020-09-16
3. Open Power System Data (2020). *Time Series*. https://doi.org/10.25832/time_series/2020-10-06
4. UK National Grid ESO. *Carbon Intensity API*. https://carbonintensity.org.uk
5. Kaggle. *Solar Power Generation Data*. https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
6. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
7. Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
8. PuLP Linear Programming Library. https://coin-or.github.io/pulp/
9. Streamlit. https://streamlit.io

---

## 📦 Output Files Reference

| File | Description |
|---|---|
| `outputs/forecasts/solar_plant1_predictions.csv` | Solar Plant 1 — test-set predictions (all models) |
| `outputs/forecasts/solar_plant2_predictions.csv` | Solar Plant 2 — test-set predictions (all models) |
| `outputs/forecasts/demand_DE_predictions.csv` | Demand DE — test-set predictions (all models) |
| `outputs/forecasts/optimization_comparison.csv` | LP optimizer baseline vs optimized comparison |
| `outputs/forecasts/optimization_schedules.csv` | Hourly battery/load/solar schedule (LP) |
| `outputs/forecasts/setpoint_schedule.csv` | Hourly AC set-point schedule (paper method) |
| `outputs/metrics/solar_plant1_metrics.csv` | Per-model metrics — Solar Plant 1 |
| `outputs/metrics/solar_plant2_metrics.csv` | Per-model metrics — Solar Plant 2 |
| `outputs/metrics/demand_DE_metrics.csv` | Per-model metrics — Demand DE |
| `outputs/metrics/solar_plant1_rf_feature_importance.csv` | RF feature importances — Solar Plant 1 |
| `outputs/metrics/solar_plant1_xgb_feature_importance.csv` | XGB feature importances — Solar Plant 1 |
| `models/solar_plant1_svr.pkl` | Trained SVR model — Solar Plant 1 |
| `models/solar_plant1_svr_features.json` | SVR feature list + best hyperparameters |

---

*Last updated: November 2026 — Python 3.14.6 · Windows 11 · scikit-learn 1.9.1 · XGBoost 3.4.1 · TensorFlow 2.22 · PuLP 4.2.0 + HiGHS 1.15.1 · Streamlit 1.66.1*
