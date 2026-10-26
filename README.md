# Building-to-Grid Demand Response Optimization
## via Predictive AI and Dynamic Load Shifting

An end-to-end AI for Sustainability project integrating solar forecasting, electricity demand forecasting, linear programming optimization, and a Streamlit dashboard.

---

## 🏗️ Project Structure

```
AIFS PROJECT/
├── app/
│   ├── dashboard.py              # Streamlit entry point
│   └── pages/
│       ├── page1_overview.py     # Energy overview & KPIs
│       ├── page2_solar.py        # Solar forecasting
│       ├── page3_demand.py       # Demand forecasting
│       ├── page4_weather.py      # Weather analysis
│       ├── page5_optimization.py # LP optimization (interactive)
│       └── page6_sustainability.py # Sustainability report
├── src/
│   ├── data/
│   │   ├── discovery.py          # Dataset inventory
│   │   ├── preprocessing.py      # ETL pipeline
│   │   ├── validation.py         # Schema validation
│   │   ├── feature_engineering.py # Feature creation + chronological split
│   │   └── integration.py        # Multi-source dataset merging
│   ├── models/
│   │   └── forecasting.py        # Persistence, Ridge, RF, XGBoost, LSTM
│   ├── optimization/
│   │   └── optimizer.py          # PuLP LP demand response optimization
│   └── sustainability/
│       └── analysis.py           # Cost, carbon, renewables calculations
├── tests/
│   └── test_suite.py             # 31 unit tests (100% pass)
├── presentation/
│   ├── generate_pptx.py          # Auto PPTX generator
│   └── SolarAI_Final_Presentation.pptx  # Generated presentation
├── data/
│   ├── raw/                      # Original datasets (untouched)
│   └── processed/                # Preprocessed CSVs
├── outputs/
│   ├── forecasts/                # Model predictions + optimization results
│   ├── metrics/                  # Per-model performance metrics
│   └── reports/                  # Sustainability markdown reports
└── models/                       # Serialized trained models (.joblib)
```

---

## 📊 Datasets Used

| Dataset | Source | Format | Key Variables |
|---|---|---|---|
| Solar Generation (Plant 1 & 2) | Kaggle (India, 2020) | CSV 15-min | AC/DC Power, daily yield |
| Solar Weather Sensors | Co-located with plants | CSV 15-min | Temperature (°C), Irradiation (W/m²) |
| UK Carbon Emissions | UK National Grid ESO | CSV 30-min | gCO₂/kWh, fuel mix (gas/coal/wind/solar...) |
| European Electricity Demand | Open Power System Data (ENTSOE) | CSV hourly | MW load, forecast, price (EUR/MWh) |
| European Weather | Open Power System Data (NASA MERRA-2) | CSV hourly | Temperature, radiation (1980–2019) |

> **⚠️ Geographic note:** Solar plant data is from India and is used only for solar generation forecasting. European demand/weather data is used for demand forecasting. These datasets are NOT merged geographically.

---

## 🚀 Quick Start

### 1. Run the full pipeline (one time)

```bash
# Step 1: Dataset discovery (inspect all 4 folders)
python -m src.data.discovery

# Step 2: Preprocessing & ETL
python -m src.data.preprocessing

# Step 3: Integration (merge solar + weather, demand + weather)
python -m src.data.integration

# Step 4: Feature engineering
python -m src.data.feature_engineering

# Step 5: Train all forecasting models
python -m src.models.forecasting

# Step 6: Run LP optimization demo
python -m src.optimization.optimizer
```

### 2. Run the dashboard

```bash
streamlit run app/dashboard.py
```

### 3. Run tests

```bash
python -m pytest tests/test_suite.py -v
```

### 4. Regenerate the PowerPoint presentation

```bash
python -m presentation.generate_pptx
```

---

## 🤖 Model Performance (Actual Results)

### Solar Generation Forecasting (Plant 1, 15-min AC Power, kW)

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Persistence | 1218.7 | 2325.3 | 0.920 |
| Linear Ridge | 255.4 | 510.8 | **0.996** |
| Random Forest | **244.1** | **507.1** | **0.996** |
| XGBoost | 273.3 | 571.0 | 0.995 |
| LSTM (2-layer, 64 units) | 1373.1 | 2412.4 | 0.903 |

### Solar Generation Forecasting (Plant 2, 15-min AC Power, kW)

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Persistence | 1213.2 | 2347.2 | 0.826 |
| Linear Ridge | 838.2 | 1660.4 | 0.913 |
| Random Forest | 704.5 | 1694.7 | 0.910 |
| XGBoost | 712.7 | 1692.0 | 0.910 |
| LSTM (2-layer, 64 units) | 1328.0 | 2460.6 | 0.811 |

### Electricity Demand Forecasting (Germany, DE, hourly MW)

| Model | MAE (MW) | RMSE (MW) | R² |
|---|---|---|---|
| Persistence | 1855.0 | 2409.2 | 0.941 |
| Linear Ridge | 844.3 | 1107.0 | 0.988 |
| Random Forest | 587.7 | 787.1 | 0.994 |
| **XGBoost** | **530.5** | **695.1** | **0.995** |
| LSTM (2-layer, 64 units) | 1648.1 | 2198.2 | 0.951 |

> **LSTM architecture:** 2-layer stacked LSTM (64 → 32 units) + 20% Dropout, Adam optimizer, MSE loss, EarlyStopping (patience=5), lookback window = 96 steps, trained with TensorFlow 2.22.
> **Note:** LSTM underperforms tree models here due to the short training window (34 days solar / 6 years demand). With more data or longer fine-tuning it would be competitive.

---

## ⚡ Optimization Results (Example Scenario)

| Metric | Baseline | Optimized | Saving |
|---|---|---|---|
| Total Cost (€) | 159.75 | 150.37 | **↓ 5.9%** |
| Carbon Emissions (kg CO₂) | 150.0 | 154.5 | ↑ 3.0%* |
| Peak Grid Demand (kW) | 40.0 | 39.3 | **↓ 1.8%** |

> *Note: With higher carbon weight in the objective function, carbon savings improve. With cost_weight=1.0, carbon_weight=0.5, the optimizer prioritizes cost.*

---

## 🌱 Sustainability Reports

Automatically generated reports are saved to `outputs/reports/` after running the optimizer:

| File | Description |
|---|---|
| `outputs/reports/sustainability_report.md` | Baseline vs. optimized comparison with methodology |
| `outputs/reports/full_report.md` | Full report: sustainability + all model metrics |
| `outputs/reports/sustainability_comparison.csv` | Machine-readable comparison table |

To regenerate:
```bash
python -m src.sustainability.analysis
# or from the dashboard → 🌱 Sustainability Report → Run Demo Optimization
```

---

## ⚠️ Key Assumptions

| Assumption | Value | Rationale |
|---|---|---|
| Carbon Intensity | 280 gCO₂/kWh | UK National Grid average (proxy; labeled clearly) |
| Building Load | 15–40 kW representative profile | Synthetic residential/commercial |
| Tariff | Configurable (€0.25/kWh default) | User scenario |
| Battery Capacity | Configurable (100 kWh default) | User scenario |
| Optimizer Horizon | 24 hourly intervals | Single-day LP (rolling MPC possible extension) |
| Solar scaling | Plant data scaled to building-level | Documented in dashboard |

---

## 📦 Dependencies

```bash
pip install pandas numpy scikit-learn xgboost pulp highspy streamlit plotly \
            openpyxl python-pptx joblib scipy matplotlib seaborn pyarrow \
            tensorflow tabulate
```

> **Solver:** PuLP 4.x + HiGHS (`pip install highspy`)
> **LSTM:** TensorFlow 2.x (`pip install tensorflow`)

---

## 📂 Data Documentation

See [`data/README.md`](data/README.md) for full documentation of all processed datasets:
- File schemas and column descriptions
- Time ranges and resolutions
- Missing data handling strategies
- Data sources and citations
- Pipeline reproduction steps

---

## 📄 References

1. Open Power System Data (2020). *Weather Data*. https://doi.org/10.25832/weather_data/2020-09-16
2. Open Power System Data (2020). *Time Series*. https://doi.org/10.25832/time_series/2020-10-06
3. UK National Grid ESO. Carbon Intensity API. https://carbonintensity.org.uk
4. Kaggle Solar Power Generation Data. https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
5. Pedregosa et al. (2011). Scikit-learn. JMLR 12, 2825–2830.
6. Chen & Guestrin (2016). XGBoost. KDD 2016.
7. PuLP. https://coin-or.github.io/pulp/
