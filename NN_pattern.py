import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv("cleaned_bankFraud.csv")

# Preprocessing (match NN training)
df["Transaction_Time"] = pd.to_timedelta(df["Transaction_Time"]).dt.total_seconds()
df = df.drop(columns=["Transaction_Description", "Transaction_Location"])

categorical_cols = [
    "Gender", "State", "Bank_Branch", "Account_Type",
    "Transaction_Type", "Merchant_Category",
    "Transaction_Device", "Device_Type", "Transaction_Currency"
]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features and target
X = df_encoded.drop(columns=["Is_Fraud"])
y = df_encoded["Is_Fraud"]

# Load trained model
model = load_model("fraud_neural_network_model.h5")

# Predicted probabilities
df_encoded["Pred_Prob"] = model.predict(X).flatten()

# -----------------------------
# 1. Fraud by Age and Gender (stacked)
# -----------------------------
df["AgeGroup"] = pd.cut(df["Age"], bins=[18,25,35,45,55,65,75])
age_gender_fraud = df.groupby(["AgeGroup", "Gender"])["Is_Fraud"].mean().unstack()

age_gender_fraud.plot(kind="bar", stacked=True, figsize=(8,6))
plt.title("Fraud Rate by Age Group and Gender")
plt.ylabel("Fraud Rate")
plt.xlabel("Age Group")
plt.legend(title="Gender")
plt.show()

# -----------------------------
# 2. Fraud by Time of Day (grouped)
# -----------------------------
df["Hour"] = pd.to_datetime(df["Transaction_Time"], unit="s").dt.hour

def period_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df["Period"] = df["Hour"].apply(period_of_day)
period_fraud = df.groupby("Period")["Is_Fraud"].mean().reindex(["Morning","Afternoon","Evening","Night"])
period_fraud.plot(kind="bar", figsize=(6,4))
plt.title("Fraud Rate by Period of Day")
plt.ylabel("Fraud Rate")
plt.show()

df_encoded["Period"] = df["Period"]
period_pred = df_encoded.groupby("Period")["Pred_Prob"].mean().reindex(["Morning","Afternoon","Evening","Night"])
period_pred.plot(kind="bar", figsize=(6,4))
plt.title("Predicted Fraud Probability by Period of Day")
plt.ylabel("Predicted Fraud Probability")
plt.show()

# -----------------------------
# 3. Fraud by Month
# -----------------------------
month_fraud = df.groupby("Transaction_Month")["Is_Fraud"].mean()
month_fraud.plot(kind="bar", figsize=(8,4))
plt.title("Fraud Rate by Month")
plt.ylabel("Fraud Rate")
plt.xlabel("Month")
plt.show()

df_encoded["Transaction_Month"] = df["Transaction_Month"]
month_pred = df_encoded.groupby("Transaction_Month")["Pred_Prob"].mean()
month_pred.plot(kind="bar", figsize=(8,4))
plt.title("Predicted Fraud Probability by Month")
plt.ylabel("Predicted Fraud Probability")
plt.xlabel("Month")
plt.show()

# -----------------------------
# 4. Fraud by Day of Month
# -----------------------------
day_fraud = df.groupby("Transaction_Day")["Is_Fraud"].mean()
day_fraud.plot(kind="bar", figsize=(12,4))
plt.title("Fraud Rate by Day of Month")
plt.ylabel("Fraud Rate")
plt.xlabel("Day")
plt.show()

df_encoded["Transaction_Day"] = df["Transaction_Day"]
day_pred = df_encoded.groupby("Transaction_Day")["Pred_Prob"].mean()
day_pred.plot(kind="bar", figsize=(12,4))
plt.title("Predicted Fraud Probability by Day of Month")
plt.ylabel("Predicted Fraud Probability")
plt.xlabel("Day")
plt.show()