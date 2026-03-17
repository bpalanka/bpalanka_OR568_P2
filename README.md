# Bank Fraud Detection Analysis

## Project Overview

This project analyzes bank transaction data to identify fraud patterns using preprocessing, exploratory data analysis, and machine learning models. The goal was to evaluate whether transaction features could be used to predict fraudulent activity.

## Dataset

Dataset source: Kaggle
Dataset link: https://www.kaggle.com/datasets/orangelmendez/bank-fraud

The dataset contains transaction-level banking information including customer demographics, transaction details, account information, device information, and fraud labels.

## Files Included

* `preprocess.py`
  Cleans the raw dataset, removes unnecessary columns, converts categorical variables, extracts date features, checks skewness, applies transformations, and saves the cleaned dataset.

* `NN_model.py`
  Trains the neural network model for fraud classification.

* `NN_pattern.py`
  Uses the trained neural network model to analyze and visualize fraud patterns.

* `SVM_model.py`
  Trains the support vector machine model for fraud classification.

* `SVM_pattern.py`
  Uses the trained SVM model to analyze fraud pattern outputs.

## Methods Used

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Correlation analysis
* Principal Component Analysis (PCA)
* Neural Network classification
* Support Vector Machine (SVM)

## Main Findings

* Predictor variables showed very low correlation with the fraud target.
* Neural network predictions were mostly uniform because the model could not identify strong patterns.
* SVM produced similar limited results.
* EDA showed clearer descriptive trends than predictive modeling.

## Roadblocks

The main limitation was weak feature signal in the dataset. Improving model performance would require major feature engineering or substantial changes to the dataset, which would significantly alter the original data.

## Dependencies

* pandas
* numpy
* scikit-learn
* tensorflow
* matplotlib
* seaborn

## Running the Project

1. Run `preprocess.py`
2. Run `NN_model.py`
3. Run `NN_pattern.py`
4. Run `SVM_model.py`
5. Run `SVM_pattern.py`
