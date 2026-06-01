# WeatherSphere — Data Science Assessment
**PM Accelerator | Data Scientist Intern | Syeda Tehreem Ali Bukhari**

---

## About PM Accelerator
PM Accelerator empowers aspiring and experienced product managers through mentorship, hands-on projects, and a vibrant global community. We accelerate careers by connecting talent with industry leaders across the world's top technology companies, equipping participants with real-world skills to drive product innovation and leadership.

🔗 [LinkedIn](https://www.linkedin.com/school/pmaccelerator/)

---

## Overview
Advanced weather trend forecasting and analysis using the **Global Weather Repository** dataset from Kaggle (40+ features, worldwide daily weather data).

### What's Covered (Advanced Assessment)
- ✅ Data Cleaning & Preprocessing (missing values, outliers, normalization)
- ✅ Advanced EDA with Anomaly Detection (Isolation Forest + Z-score)
- ✅ Multiple Forecasting Models (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- ✅ Ensemble Model (top-3 average)
- ✅ Climate Analysis (long-term trends, seasonal decomposition)
- ✅ Environmental Impact (air quality vs weather correlations)
- ✅ Feature Importance (built-in RF + permutation importance)
- ✅ Spatial Analysis (geographical temperature mapping)
- ✅ Geographical Patterns (country/continent comparisons)

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/weathersphere.git
cd weathersphere/data-science
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
Option A — Kaggle CLI:
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle datasets download -d nelgiriyewithana/global-weather-repository
unzip global-weather-repository.zip -d .
```
Option B — Manual: Download from https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository and place `GlobalWeatherRepository.csv` in the `data-science/` folder.

> **Note:** If the dataset is not found, the script auto-generates realistic synthetic data so all analyses still run and demonstrate the full pipeline.

### 4. Run analysis
```bash
python src/analysis.py
```

### 5. View outputs
All charts and the final report are saved to `outputs/`:
```
outputs/
  01_eda_overview.png
  02_anomaly_detection.png
  03_forecasting_models.png
  04_climate_analysis.png
  05_environmental_impact.png
  06_feature_importance.png
  07_spatial_analysis.png
  08_geographical_patterns.png
  analysis_report.md
```

---

## Project Structure
```
data-science/
├── src/
│   └── analysis.py          # Main analysis pipeline
├── outputs/                 # Generated charts & report
├── requirements.txt
└── README.md
```

---

## Methodology

### Data Cleaning
- DateTime parsing and chronological sorting
- Median/mode imputation for missing values
- 3×IQR winsorization for extreme outliers
- Z-score standardization for ML features

### Anomaly Detection
- **Isolation Forest** — unsupervised, detects multivariate anomalies (unusual combinations of temp/humidity/wind/pressure)
- **Z-score** — univariate, flags temperature values >3 standard deviations from mean

### Forecasting
All models use lag features (1/3/7/14 days), rolling statistics, and meteorological parameters as input features.

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Ridge Regression | Regularized, handles multicollinearity |
| Random Forest | Non-linear, handles interactions |
| Gradient Boosting | Sequential ensemble, highest accuracy |
| Ensemble | Average of top-3 models |

### Evaluation Metrics
- **MAE** — Mean Absolute Error (interpretable, in °C)
- **RMSE** — Root Mean Squared Error (penalizes large errors)
- **R²** — Coefficient of determination (variance explained)
