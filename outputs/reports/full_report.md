# Full Sustainability & Performance Report
*Generated automatically from AIFS pipeline*

---

## 1. Sustainability Impact Summary

| Metric                 | Unit   |   Baseline |   Optimized |   Saving |   Saving_% |
|:-----------------------|:-------|-----------:|------------:|---------:|-----------:|
| Electricity Cost       | EUR    |     92.886 |       165.3 |  -72.414 |      -78   |
| Carbon Emissions       | kg CO₂ |    114.957 |       168   |  -53.043 |      -46.1 |
| Peak Grid Demand       | kW     |     30     |        40   |  -10     |      -33.3 |
| Solar Self-Consumption | kWh    |    309.44  |         0   |  309.44  |      100   |

---

## 2. Assumptions & Limitations

- **Carbon Intensity**: 280 gCO2/kWh – UK National Grid half-hourly average (proxy; NOT representative of any specific European grid)
- **Building Load Profile**: Representative 30 kW flat commercial load (synthetic)
- **Solar Generation**: Bell-curve profile scaled to 60 kW peak (based on Plant 1 data)
- **Electricity Tariff**: Time-of-use: EUR0.15/kWh (off-peak 00:00-06:59), EUR0.30/kWh (peak 07:00-23:59)
- **Battery Capacity**: 100 kWh (default; user-configurable in dashboard)
- **Optimizer Horizon**: 24 hourly intervals (single-day LP)
- **Optimizer**: PuLP 4.x with HiGHS solver
- **Demand Forecast Model**: XGBoost (best R2=0.995, MAE=530.5 MW on Germany DE)
- **Solar Forecast Model**: Random Forest (best R2=0.996, MAE=244.1 kW on Plant 1)

---

## 3. Forecasting Model Performance

### Solar Plant 1 (15-min AC Power, kW)

| model         |      mae |     rmse |     r2 |   nmae |   mape_pct |
|:--------------|---------:|---------:|-------:|-------:|-----------:|
| persistence   | 1218.7   | 2325.33  | 0.92   | 0.042  |        nan |
| linear_ridge  |  255.434 |  510.84  | 0.9961 | 0.0088 |        nan |
| random_forest |  244.117 |  507.095 | 0.9962 | 0.0084 |        nan |
| xgboost       |  273.308 |  571.017 | 0.9952 | 0.0094 |        nan |
| lstm          | 1373.13  | 2412.42  | 0.9025 | 0.0473 |        nan |

### Solar Plant 2 (15-min AC Power, kW)

| model         |      mae |    rmse |     r2 |   nmae |   mape_pct |
|:--------------|---------:|--------:|-------:|-------:|-----------:|
| persistence   | 1213.2   | 2347.21 | 0.8264 | 0.0541 |        nan |
| linear_ridge  |  838.201 | 1660.42 | 0.9132 | 0.0374 |        nan |
| random_forest |  704.458 | 1694.67 | 0.9095 | 0.0314 |        nan |
| xgboost       |  712.749 | 1692.02 | 0.9098 | 0.0318 |        nan |
| lstm          | 1328.04  | 2460.65 | 0.8106 | 0.0592 |        nan |

### Electricity Demand – Germany DE (hourly MW)

| model         |      mae |     rmse |     r2 |   nmae |   mape_pct |
|:--------------|---------:|---------:|-------:|-------:|-----------:|
| persistence   | 1855.01  | 2409.23  | 0.9407 | 0.0425 |       3.55 |
| linear_ridge  |  844.333 | 1107.04  | 0.9875 | 0.0194 |       1.61 |
| random_forest |  587.747 |  787.086 | 0.9937 | 0.0135 |       1.13 |
| xgboost       |  530.47  |  695.149 | 0.9951 | 0.0122 |       1.03 |
| lstm          | 1648.11  | 2198.22  | 0.9506 | 0.0378 |       3.26 |

---

## 4. Methodology

### Energy Balance
```
grid_import(t) = building_load(t) - solar_used(t) - batt_discharge(t) + batt_charge(t)
```

### LP Objective
```
Minimize: sum_t [ tariff(t)*grid_import(t) + carbon_weight*CI(t)*grid_import(t) + peak_penalty*peak ]
```

### Carbon Formula
```
Emissions(t) = grid_import(t)[kW] x CI(t)[gCO2/kWh] x dt[h]  ->  gCO2
Total = sum_t Emissions(t) / 1000  [kgCO2]
```

---
*End of report*