# Bank Fraud Detection Analysis

## Project Overview

This project builds a bank fraud detection pipeline using data preprocessing, exploratory data analysis (EDA), and a deep learning model. The main objective is to determine whether transaction-level features can be used to predict fraudulent activity and to analyze patterns in both actual fraud labels and model-predicted fraud risk across different behavioral and temporal dimensions.

A neural network is used because fraud detection involves complex, nonlinear relationships between variables such as transaction amount, time, customer profile, device type, and behavioral patterns. These interactions are difficult for traditional linear models to capture effectively.

The project separates:
- Actual fraud analysis (ground truth)
- Model-based fraud risk analysis (predictions)

---

## Dataset

Dataset source: Kaggle  
Dataset link: https://www.kaggle.com/datasets/orangelmendez/bank-fraud  

The dataset contains transaction-level banking data including:

- Customer demographics
- Account information
- Transaction details
- Device and location information
- Fraud labels

---

## Files Included

### preprocess.py
Handles data cleaning, feature engineering, and exploratory data analysis.

- Loads raw dataset from Kaggle
- Cleans missing and irrelevant columns
- Encodes categorical variables using one-hot encoding
- Converts date and time fields into numerical features
- Creates additional features:
  - Day of week
  - Weekend indicator
  - Time-based grouping (used later in analysis)
- Generates exploratory visualizations (based on actual fraud labels):
  - Fraud rate by weekday vs weekend
  - Fraud rate by day of week (Monday–Sunday)
  - Fraud rate by gender
  - Fraud rate by device type
  - Fraud rate by time of day
- Saves final cleaned dataset as `cleaned_bankFraud.csv`
- Saves feature order as `feature_order.pkl` for model consistency

---

### nn_model.py
Trains the neural network fraud detection model.

- Loads cleaned dataset
- Splits data into training and testing sets using stratified sampling
- Applies feature scaling using StandardScaler
- Builds a deep neural network with:
  - Dense layers
  - Batch normalization
  - Dropout regularization
- Uses binary cross-entropy loss for classification
- Applies early stopping to prevent overfitting
- Saves trained model as `fraud_neural_network.h5`
- Saves scaler as `scaler.pkl`

---

### nn_pattern.py
Performs fraud risk analysis using model predictions.

- Loads trained model, scaler, and feature order
- Generates fraud probability predictions for each transaction (`Pred_Prob`)
- Analyzes model-learned fraud patterns using predictions:
  - Predicted fraud risk by weekday vs weekend
  - Predicted fraud risk by day of week (Monday–Sunday)
  - Predicted fraud risk by gender
  - Predicted fraud risk by device type
  - Predicted fraud risk by time of day

All visualizations in this file are based on model predictions rather than actual labels.

---

## Methods Used

- Data cleaning and preprocessing
- Feature engineering (temporal + behavioral features)
- Exploratory data analysis (EDA)
- One-hot encoding
- Feature scaling
- Deep neural network classification
- Prediction-based behavioral analysis

---

## Main Findings

- Fraud patterns are nonlinear and not strongly visible in raw feature correlations.
- Neural network captures hidden interactions between transaction features.
- Aggregated prediction analysis reveals meaningful behavioral patterns.
- Time-based, demographic, and device-related features show variation in fraud risk.
- Model-based analysis provides different insights compared to raw label analysis.

---

## Key Insight

The neural network learns complex interactions between transaction behavior, timing, device usage, and user attributes. While individual predictions may appear noisy, aggregated model outputs reveal consistent fraud risk patterns across groups.

This distinction between:
- Actual fraud distribution
- Predicted fraud probability distribution

is central to understanding model behavior.

---

## Dependencies

- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- kagglehub

---

## Running the Project

Run files in order:

```bash
python preprocess.py
python nn_model.py
python nn_pattern.py
```

---

## Output Files

- cleaned_bankFraud.csv → processed dataset  
- fraud_neural_network.h5 → trained neural network model  
- scaler.pkl → feature scaler  
- feature_order.pkl → feature alignment reference
```
