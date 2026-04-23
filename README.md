# Bank Fraud Detection Analysis

## Project Overview

This project builds a bank fraud detection pipeline using data preprocessing, exploratory data analysis, and a deep learning model. The main objective is to determine whether transaction-level features can be used to predict fraudulent activity and to analyze patterns in predicted fraud risk across different behavioral and temporal dimensions.

A neural network is used because fraud detection involves complex, nonlinear relationships between variables such as transaction amount, time, customer profile, and device usage. These interactions are difficult for traditional linear models to capture effectively.

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
Handles all data preparation and exploratory analysis before model training.

- Loads raw dataset from Kaggle
- Cleans missing and irrelevant columns
- Encodes categorical variables using one-hot encoding
- Converts date and time fields into numerical features
- Generates exploratory visualizations:
  - Fraud class distribution
  - Transaction amount distribution
  - Account balance distribution
  - Correlation heatmap
- Saves final cleaned dataset as `cleaned_bankFraud.csv`

---

### nn_model.py
Trains the neural network fraud detection model.

- Loads cleaned dataset
- Splits data into training and testing sets (stratified split)
- Applies feature scaling using StandardScaler
- Builds deep neural network with dense layers, batch normalization, and dropout
- Uses binary cross-entropy loss for classification
- Applies early stopping to reduce overfitting
- Saves trained model as `fraud_neural_network.h5`
- Saves scaler as `scaler.pkl`

---

### nn_pattern.py
Performs prediction-based fraud behavior analysis.

- Loads trained model and scaler
- Generates fraud probability predictions for all transactions
- Analyzes fraud risk patterns using model outputs:
  - Fraud risk by age group and gender
  - Fraud risk by time of day
  - Fraud risk by month
  - Fraud risk by day of month
- All visualizations are based on predicted probabilities

---

## Methods Used

- Data cleaning and preprocessing
- Feature engineering
- Exploratory data analysis (EDA)
- One-hot encoding
- Feature scaling
- Deep neural network classification
- Behavioral analysis using predictions

---

## Main Findings

- Fraud patterns are nonlinear and not strongly visible in raw correlations.
- Neural network captures hidden interactions between transaction features.
- Aggregated prediction analysis reveals meaningful behavioral trends.
- Time-based and demographic features show variation in fraud risk.

---

## Key Insight

The neural network learns complex interactions between transaction behavior, timing, and user attributes. While individual predictions may appear noisy, aggregated results provide meaningful insight into fraud risk patterns.

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
- fraud_neural_network.h5 → trained model
- scaler.pkl → feature scaler
```

1. Run `preprocess.py`
2. Run `NN_model.py`
3. Run `NN_pattern.py`
