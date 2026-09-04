# Horse Race Finish Time Prediction

Exploratory data analysis and regression modeling on horse racing data to identify the factors that influence race outcomes and predict a horse's finish time.

## Overview

This project analyzes two datasets (race-level records and individual horse/run records) to understand what drives horse racing performance and to build a model that predicts finish time. The workflow covers data cleaning, exploratory analysis, feature engineering, dimensionality reduction, and regression modeling, along with feature selection to identify the strongest predictors of race outcomes.

## Dataset

Two CSV files were used:

- **`races.csv`** — race-level data (6,349 instances, 37 features), including venue, track conditions, distance, and prize information.
- **`runs.csv`** — individual horse/run-level data (79,447 instances, 37 features), including horse attributes, jockey/trainer information, and sectional times.

## Methodology

**Data Cleaning & Preprocessing**
- Reviewed missingness patterns across both datasets and applied different strategies depending on the extent and nature of the missing data:
  - Dropped rows with a small number of missing values (e.g. `horse_country`, `horse_type`)
  - Used KNN imputation for numerical columns with a meaningful proportion of missing values (e.g. `place_odds`, sectional times)
  - Dropped columns with near-total missingness (e.g. `sec_time7`, `time7`)
  - Converted sparse categorical columns into binary flags where appropriate (e.g. `win_combination2`)
- Converted date fields into separate year/month/day components for use in modeling
- Label-encoded categorical variables (`horse_country`, `horse_type`, `horse_gear`, `venue`, `config`, `going`, `horse_ratings`)
- Merged race-level and run-level data into a single dataframe on `race_id`

**Exploratory Data Analysis**
- Correlation heatmaps to assess multicollinearity across numerical features
- Analysis of factors affecting horse performance, including track conditions and jockey/trainer win percentages

**Modeling**
- Standardized features using `StandardScaler`
- Split data into 80% training / 20% test sets
- Applied Principal Component Analysis (PCA) to reduce dimensionality and address multicollinearity
- Trained and compared Elastic Net regression models on both the original scaled features and PCA-transformed features
- Applied Recursive Feature Elimination (RFE) with a linear regression estimator to identify the 20 most predictive features, then retrained the model on this reduced feature set

## Key Findings

- Elastic Net regression on the original scaled features outperformed the PCA-transformed model, indicating that dimensionality reduction did not improve predictive performance for this dataset.
- Recursive Feature Elimination identified race distance, sectional times, and prior placement metrics as the strongest predictors of finish time.
- High multicollinearity was present across several numerical features, which informed the feature selection strategy.

## Tech Stack

- **Language:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine learning:** scikit-learn (KNNImputer, LabelEncoder, StandardScaler, PCA, LinearRegression, ElasticNet, RFE)
- **Environment:** Google Colab / Jupyter Notebook

## Repository Contents

- `WAF_Semester2_code.html` — Full notebook (exported as HTML) containing the analysis, code, and visualizations

## Notes

This project is a stand-alone analysis and is intended to demonstrate an end-to-end data analysis and predictive modeling workflow, from raw data to model evaluation.
