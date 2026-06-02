WeatherSphere: Advanced Weather Analysis & Forecasting
Overview

WeatherSphere is an advanced data science project focused on global weather trend forecasting and environmental analysis using the Global Weather Repository dataset from Kaggle. The dataset includes 40+ features covering worldwide daily weather observations.

This project demonstrates an end-to-end machine learning pipeline, including data preprocessing, exploratory data analysis, anomaly detection, forecasting, and climate insights.

Key Features
Data Processing & EDA
Missing value handling using median and mode imputation
Outlier detection and treatment using IQR-based winsorization
Feature scaling and normalization
Exploratory data analysis for pattern discovery
Anomaly Detection
Isolation Forest for multivariate anomaly detection
Z-score method for detecting extreme temperature values
Machine Learning Models
Linear Regression (baseline model)
Ridge Regression (regularized linear model)
Random Forest Regressor
Gradient Boosting Regressor
Ensemble model (average of top-performing models)
Climate and Environmental Analysis
Long-term climate trend analysis
Seasonal decomposition of weather patterns
Air quality and weather correlation analysis
Geographic comparisons at country and continent level
Feature Engineering
Lag features (1, 3, 7, 14 days)
Rolling statistical features
Meteorological transformations
Feature importance using Random Forest and permutation methods
Spatial Analysis
Geographic temperature distribution visualization
Regional climate pattern comparisons
Setup and Installation
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/weathersphere.git
cd weathersphere/data-science
2. Install Dependencies
pip install -r requirements.txt
3. Dataset Setup
Option A: Kaggle CLI (Recommended)
pip install kaggle

# Add kaggle.json to ~/.kaggle/
kaggle datasets download -d nelgiriyewithana/global-weather-repository
unzip global-weather-repository.zip -d .
Option B: Manual Download

Download the dataset from:
https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository

Place the file:

GlobalWeatherRepository.csv

inside the data-science/ directory.

Note: If the dataset is not found, the project automatically generates synthetic data so the full pipeline can still run.

Run the Project
python src/analysis.py
Outputs

All generated charts and reports are saved in the outputs directory:

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
Project Structure
data-science/
├── src/
│   └── analysis.py        # Main pipeline script
├── outputs/               # Generated visualizations and reports
├── requirements.txt
└── README.md
Methodology
Data Cleaning
Datetime parsing and sorting
Missing value imputation using median and mode
Outlier handling using 3×IQR winsorization
Feature scaling for machine learning models
Anomaly Detection
Isolation Forest for multivariate anomaly detection
Z-score method for identifying extreme values
Forecasting Models

Models are trained using lag features and rolling statistical indicators.

Model	Description
Linear Regression	Baseline model
Ridge Regression	Regularized linear regression
Random Forest	Non-linear ensemble model
Gradient Boosting	High-performance boosting model
Ensemble	Average of top models
Evaluation Metrics
MAE (Mean Absolute Error): average prediction error
RMSE (Root Mean Squared Error): penalizes large errors
R² Score: variance explained by model
Summary

WeatherSphere provides a complete machine learning pipeline for weather analysis, combining data engineering, predictive modeling, and climate insights in a structured and reproducible workflow.
