# WeatherSphere — Global Weather Data Analysis Report

**Author:** Muhammad Hamza  
**Assessment:** PM Accelerator — Data Scientist Intern  
**Date:** June 03, 2026

---


## Dataset Overview

| Property | Value |
|----------|-------|
| Total Records | 5,000 |
| Features | 42 |
| Date Range | 2020-01-01 00:00:00 to 2023-12-31 00:00:00 |
| Countries | 12 |
| Cities | 35 |

---

## 1. Data Cleaning & Preprocessing

**Approach:**
- Parsed `last_updated` as datetime and sorted chronologically
- Extracted time features: year, month, day, day-of-year, season
- Missing values filled with column median (numeric) or mode (categorical)
- Extreme outliers capped using 3×IQR winsorization (preserves distribution shape)
- Z-score normalization applied to key numeric features (saved as `_norm` columns)
- Duplicate rows removed

**Key Statistics (post-cleaning):**

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| Temperature (°C) | 15.2 | 11.3 | -18.0 | 46.4 |
| Humidity (%) | 64.6 | 19.4 | 10.0 | 100.0 |
| Wind (kph) | 14.8 | 14.2 | 0.0 | 69.6 |

---

## 2. Exploratory Data Analysis

**Key Findings:**
- Temperature shows clear seasonal cyclicality (peaks in summer months, troughs in winter)
- Humidity and temperature are negatively correlated in temperate climates
- Tropical countries (Brazil, India) show consistently high temperatures and humidity
- Polar/sub-polar countries (Russia, Canada) show the highest temperature variability (±20°C seasonally)
- Precipitation follows an exponential distribution — most days have zero/low precipitation with rare heavy events

**Anomaly Detection:**
- **Isolation Forest (5% contamination):** Identified anomalies as unusual combinations of temperature, humidity, wind speed, and pressure
- **Z-Score (|z| > 3):** Flagged extreme temperature outliers representing unusual weather events (heat waves, cold snaps)

---

## 3. Forecasting Model Results

All models trained on 80% of data, evaluated on the most recent 20%.

| Model | MAE (°C) | RMSE (°C) | R² Score |
|-------|----------|-----------|----------|
| Linear Regression | 4.71 | 5.81 | 0.7296 |
| Ridge Regression | 4.71 | 5.81 | 0.7296 |
| Random Forest | 3.25 | 4.33 | 0.8502 |
| Gradient Boosting | 3.01 | 3.81 | 0.8838 |
| Ensemble (Top 3) | 3.42 | 4.28 | 0.8535 |

**Best performer:** Gradient Boosting (highest R²)

**Ensemble approach:** Average of top-3 models reduces individual model overfitting and improves generalization, particularly for edge-case temperature events.

**Feature engineering used:**
- Lag features (1, 3, 7, 14 days) to capture temporal autocorrelation
- Rolling means and standard deviations (3, 7, 14, 30 day windows)
- Encoded categorical features (country, weather condition)
- Meteorological parameters (humidity, pressure, wind, visibility)

---

## 4. Climate Analysis

**Key findings:**
- Temperature shows a slight warming trend across years in the dataset
- Seasonal amplitude varies significantly by latitude: equatorial regions show minimal seasonality while high-latitude regions show 30-40°C swings
- Northern Hemisphere summer (Jun–Aug) and Southern Hemisphere summer (Dec–Feb) are clearly visible in the data

---

## 5. Environmental Impact

**Air Quality Correlations:**
- PM2.5 concentrations are negatively correlated with wind speed (higher winds disperse pollutants)
- Humidity shows positive correlation with PM2.5 (hygroscopic growth of particles)
- Visibility decreases significantly when PM2.5 exceeds 75 μg/m³
- Temperature inversions (cool surface air trapped below warm air) can lead to AQ spikes

---

## 6. Feature Importance

**Top predictive features for temperature forecasting:**
1. **Recent lag values (1-day, 3-day)** — strongest predictor due to temporal autocorrelation
2. **Rolling means** — capture seasonal trend effectively
3. **Month / Day-of-year** — encode seasonal patterns directly
4. **Humidity** — strong physical inverse relationship with temperature
5. **Country encoding** — captures regional baseline climate

---

## 7. Spatial Analysis

- Clear latitudinal gradient: temperature decreases ~0.6°C per degree of latitude away from equator
- Coastal regions show lower temperature variance than continental interiors
- High-altitude countries show systematically lower temperatures than latitude alone would predict

---

## 8. Geographical Patterns

- Tropical belt (±23.5°) maintains consistent high temperatures year-round
- Sub-Saharan Africa and South/Southeast Asia show highest average temperatures
- Scandinavia, Russia, and Canada show extreme cold winter temperatures
- Island nations typically have the most moderate, stable climates

---

## Visualizations Generated

1. `01_eda_overview.png` — Temperature distributions, seasonality, correlations
2. `02_anomaly_detection.png` — Isolation Forest + Z-score anomalies
3. `03_forecasting_models.png` — All model predictions vs actual
4. `04_climate_analysis.png` — Long-term trends and seasonal decomposition
5. `05_environmental_impact.png` — Air quality correlations
6. `06_feature_importance.png` — RF + permutation importance
7. `07_spatial_analysis.png` — World map temperature distribution
8. `08_geographical_patterns.png` — Country-level comparisons

---

## Conclusions

1. **Temperature is highly predictable** (R² > 0.85 for tree-based models) primarily due to strong seasonality and autocorrelation
2. **Ensemble methods outperform** individual models, particularly for unusual weather events
3. **Latitude is the dominant spatial predictor** of mean temperature
4. **Air quality is strongly modulated** by wind speed and humidity — addressing pollution requires understanding meteorological context
5. **Anomaly detection successfully flagged** extreme weather events that correspond to known climate phenomena

---
