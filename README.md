# Bank Fraud Detection Using Neural Networks

## Project Overview

This project implements an end-to-end machine learning pipeline for detecting fraudulent bank transactions using a deep neural network. The system combines data preprocessing, exploratory data analysis (EDA), feature engineering, and model-based behavioral analysis to identify patterns associated with fraudulent activity.

The primary objective is not only to classify fraud but also to understand how fraud risk varies across time, demographic, and behavioral dimensions using model predictions.

A neural network is used due to its ability to capture nonlinear relationships between transaction features such as amount, time, account behavior, and device usage, which are difficult for traditional linear models to learn effectively.

---

## Dataset

Source: Kaggle  
Dataset: https://www.kaggle.com/datasets/orangelmendez/bank-fraud  

The dataset contains transaction-level banking records including:

- Customer demographics
- Account attributes
- Transaction details
- Device and location information
- Fraud labels (binary target)

---

## Project Structure

### preprocess.py
Responsible for data cleaning, feature engineering, and exploratory analysis.

Key responsibilities:
- Loads raw dataset from Kaggle
- Removes irrelevant identifiers and missing values
- Encodes categorical variables using one-hot encoding
- Converts date and time features into numerical formats
- Creates additional engineered features:
  - Transaction day of week
  - Weekend indicator
  - Time-based grouping (used only for analysis)
- Generates exploratory visualizations:
  - Fraud rate by weekday vs weekend
  - Fraud rate by day of week
  - Fraud rate by gender
  - Fraud rate by device type
  - Fraud rate by time of day
- Saves processed dataset as `cleaned_bankFraud.csv`
- Saves feature ordering for model consistency (`feature_order.pkl`)

Note: Some engineered variables (e.g., DayName, Period) are used only for EDA and are excluded from model training to prevent data leakage.

---

### nn_model.py
Implements and trains the deep learning model for fraud classification.

Key components:
- Loads cleaned dataset
- Removes non-numeric / EDA-only features before training
- Splits data using stratified sampling (80/20)
- Applies feature scaling using StandardScaler
- Builds a fully connected neural network with:
  - Dense layers
  - Batch normalization
  - Dropout regularization
- Uses binary cross-entropy loss for classification
- Applies early stopping to reduce overfitting
- Saves trained model as `fraud_neural_network.h5`
- Saves scaler as `scaler.pkl`

This stage focuses purely on predictive learning using numerical, structured features.

---

### nn_pattern.py
Performs behavioral analysis using model predictions.

Key responsibilities:
- Loads trained model, scaler, and feature order
- Generates fraud probability predictions (`Pred_Prob`)
- Analyzes fraud risk patterns using aggregated predictions:
  - Weekday vs weekend risk
  - Day of week trends
  - Gender-based risk distribution
  - Device-based risk patterns
  - Time-of-day risk variation

Unlike preprocessing analysis, this file focuses entirely on model-inferred behavior rather than actual labels.

---

## Methods Used

- Data preprocessing and cleaning
- Feature engineering (temporal and behavioral)
- One-hot encoding for categorical variables
- Feature scaling (StandardScaler)
- Deep neural network classification
- Stratified train-test splitting
- Early stopping for regularization
- Prediction-based behavioral analysis

---

## Key Insights

- Fraud behavior is nonlinear and not strongly visible through simple correlations.
- Neural networks capture complex interactions between transaction attributes.
- Aggregated prediction analysis reveals clearer behavioral patterns than raw labels alone.
- Time, device type, and demographic features influence predicted fraud risk differently across segments.
- Separating EDA features from training data improves model reliability and prevents data leakage.

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- KaggleHub

---

## How to Run

Execute files in the following order:

```bash
python preprocess.py
python nn_model.py
python nn_pattern.py
```

---

## Output Files

- `cleaned_bankFraud.csv` → processed dataset  
- `fraud_neural_network.h5` → trained neural network model  
- `scaler.pkl` → feature scaler  
- `feature_order.pkl` → feature alignment reference  

---

## Project Summary

This project demonstrates a full machine learning workflow from raw financial transaction data to a trained neural network model and behavioral pattern analysis. It highlights the importance of feature engineering, data leakage prevention, and post-model interpretability through prediction-based aggregation analysis.

The final system not only predicts fraud but also provides insights into how fraud risk varies across different user and transaction contexts.
