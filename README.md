# WeatherSphere: Advanced Weather Analysis & Forecasting

## Overview

WeatherSphere is an advanced data science project focused on global weather trend forecasting and environmental analysis using the Global Weather Repository dataset from Kaggle. The dataset contains 40+ features covering worldwide daily weather observations.

This project demonstrates an end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, anomaly detection, forecasting, feature engineering, and climate analysis.

---

## Features

### Data Processing and Exploratory Analysis
- Missing value handling using median and mode imputation
- Outlier detection and treatment using IQR-based winsorization
- Feature scaling and normalization
- Comprehensive exploratory data analysis (EDA)

### Anomaly Detection
- Isolation Forest for multivariate anomaly detection
- Z-score method for detecting extreme temperature values

### Forecasting Models
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Ensemble Model (average of top-performing models)

### Climate and Environmental Analysis
- Long-term climate trend analysis
- Seasonal decomposition of weather patterns
- Air quality and weather correlation analysis
- Country and continent-level comparisons

### Feature Engineering
- Lag features (1, 3, 7, and 14 days)
- Rolling statistical features
- Meteorological feature transformations
- Feature importance analysis using Random Forest and permutation importance

### Spatial Analysis
- Geographic temperature distribution visualization
- Regional climate pattern comparisons

---

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Statsmodels

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/weathersphere.git
cd weathersphere/data-science
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

### Option 1: Kaggle CLI

```bash
pip install kaggle

# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d nelgiriyewithana/global-weather-repository
unzip global-weather-repository.zip -d .
```

### Option 2: Manual Download

Download the dataset from:

https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository

Place the following file in the project root directory:

```text
GlobalWeatherRepository.csv
```

> **Note:** If the dataset is unavailable, the application automatically generates realistic synthetic data to demonstrate the complete analysis pipeline.

---

## Running the Project

Execute the main analysis pipeline:

```bash
python src/analysis.py
```

---

## Output Files

Generated visualizations and reports are stored in the `outputs/` directory.

```text
outputs/
├── 01_eda_overview.png
├── 02_anomaly_detection.png
├── 03_forecasting_models.png
├── 04_climate_analysis.png
├── 05_environmental_impact.png
├── 06_feature_importance.png
├── 07_spatial_analysis.png
├── 08_geographical_patterns.png
└── analysis_report.md
```

---

## Project Structure

```text
data-science/
├── src/
│   └── analysis.py
├── outputs/
├── requirements.txt
└── README.md
```

---

## Methodology

### Data Cleaning
- Datetime parsing and chronological sorting
- Missing value imputation
- Outlier handling using 3×IQR winsorization
- Feature scaling and standardization

### Anomaly Detection
- Isolation Forest for multivariate anomaly identification
- Z-score analysis for extreme temperature events

### Forecasting
Models utilize:
- Historical lag features
- Rolling averages
- Weather-related predictors
- Environmental variables

#### Models Evaluated

| Model | Description |
|--------|-------------|
| Linear Regression | Baseline forecasting model |
| Ridge Regression | Regularized linear model |
| Random Forest | Non-linear ensemble model |
| Gradient Boosting | Sequential boosting model |
| Ensemble | Average of top-performing models |

### Evaluation Metrics

| Metric | Purpose |
|---------|----------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |

---

## Results

The project provides:

- Weather trend forecasting
- Climate pattern analysis
- Temperature anomaly detection
- Air quality impact assessment
- Geographic weather comparisons
- Feature importance interpretation
- Visual analytics and reporting

---

## Future Improvements

- Deep learning models (LSTM, GRU)
- Real-time weather API integration
- Interactive dashboards using Streamlit
- Automated model retraining
- Deployment to cloud platforms

---

## License

This project is intended for educational, research, and portfolio purposes.

---

## Author

**Tehreem Ali**  
AI/ML Engineer | Data Science Enthusiast

GitHub: https://github.com/tehrimaly
LinkedIn: https://linkedin.com/in/tehrimaly
