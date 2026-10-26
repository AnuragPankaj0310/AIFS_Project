# Sustainability Analysis Report

## Results Summary

| Metric                 | Unit   |   Baseline |   Optimized |   Saving |   Saving_% |
|:-----------------------|:-------|-----------:|------------:|---------:|-----------:|
| Electricity Cost       | EUR    |     92.886 |       165.3 |  -72.414 |      -78   |
| Carbon Emissions       | kg CO₂ |    114.957 |       168   |  -53.043 |      -46.1 |
| Peak Grid Demand       | kW     |     30     |        40   |  -10     |      -33.3 |
| Solar Self-Consumption | kWh    |    309.44  |         0   |  309.44  |      100   |

## Assumptions & Limitations

- **Carbon Intensity**: 280 gCO2/kWh – UK National Grid half-hourly average (proxy; NOT representative of any specific European grid)
- **Building Load Profile**: Representative 30 kW flat commercial load (synthetic)
- **Solar Generation**: Bell-curve profile scaled to 60 kW peak (based on Plant 1 data)
- **Electricity Tariff**: Time-of-use: EUR0.15/kWh (off-peak 00:00-06:59), EUR0.30/kWh (peak 07:00-23:59)
- **Battery Capacity**: 100 kWh (default; user-configurable in dashboard)
- **Optimizer Horizon**: 24 hourly intervals (single-day LP)
- **Optimizer**: PuLP 4.x with HiGHS solver
- **Demand Forecast Model**: XGBoost (best R2=0.995, MAE=530.5 MW on Germany DE)
- **Solar Forecast Model**: Random Forest (best R2=0.996, MAE=244.1 kW on Plant 1)

## Methodology

### Electricity Cost
```
Total Cost = Σ_t [ grid_import(t) × tariff(t) × Δt ]
Units: kW × EUR/kWh × h = EUR
```

### Carbon Emissions
```
Emissions = Σ_t [ grid_import(t) × carbon_intensity(t) × Δt ]
Units: kW × gCO₂/kWh × h = gCO₂ (÷ 1000 = kg CO₂)
```

### Carbon Intensity Source
- UK National Grid half-hourly data (df_fuel_ckan.csv).
- Used as proxy carbon intensity; NOT representative of any specific European grid.
- For accurate results, replace with country-specific grid carbon intensity data.

### Renewable Utilization
```
Self-Consumption Ratio = Σ solar_used / Σ solar_available
```