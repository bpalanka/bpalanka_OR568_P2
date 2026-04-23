#Preprocessing.py 
import pandas as pd
import numpy as np
import kagglehub
import os
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Load dataset
# -----------------------------
path = kagglehub.dataset_download("orangelmendez/bank-fraud")
file_path = os.path.join(path, "new_bank_fraud_detection.csv")

df = pd.read_csv(file_path)
df = df.iloc[:, 1:]

# -----------------------------
# Basic cleaning
# -----------------------------
df = df.drop(columns=["Transaction_ID", "Merchant_ID"], errors="ignore")

df["Gender"] = df["Gender"].map({"Female": 1, "Male": 0})
df["Is_Fraud"] = df["Is_Fraud"].astype(int)

# -----------------------------
# Date features
# -----------------------------
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")

df["Transaction_Year"] = df["Transaction_Date"].dt.year
df["Transaction_Month"] = df["Transaction_Date"].dt.month
df["Transaction_Day"] = df["Transaction_Date"].dt.day
df["Transaction_DayOfWeek"] = df["Transaction_Date"].dt.dayofweek

df = df.drop(columns=["Transaction_Date"])

# -----------------------------
# Time conversion
# -----------------------------
if "Transaction_Time" in df.columns:
    df["Transaction_Time"] = pd.to_timedelta(df["Transaction_Time"]).dt.total_seconds()

# -----------------------------
# Encoding
# -----------------------------
categorical_cols = [
    "State", "Bank_Branch", "Account_Type",
    "Transaction_Type", "Merchant_Category",
    "Transaction_Device", "Device_Type",
    "Transaction_Currency"
]

df = pd.get_dummies(
    df,
    columns=[c for c in categorical_cols if c in df.columns],
    drop_first=True
)

# -----------------------------
# VISUALIZATION (POST-CLEANING)
# -----------------------------
print("Dataset shape:", df.shape)

# 1. Weekday vs Weekend
df["Is_Weekend"] = df["Transaction_DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)

weekend_dist = df.groupby("Is_Weekend")["Is_Fraud"].mean()

plt.figure()
plt.bar(["Weekday", "Weekend"], weekend_dist.values)
plt.title("Fraud Rate: Weekday vs Weekend")
plt.ylabel("Fraud Rate")
plt.show()

# 2. Day of Week (Mon–Sun)
day_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

df["DayName"] = df["Transaction_DayOfWeek"].map(day_map)

dow_dist = df.groupby("DayName")["Is_Fraud"].mean()

order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_dist = dow_dist.reindex(order)

plt.figure(figsize=(8,4))
plt.bar(dow_dist.index, dow_dist.values)
plt.title("Fraud Rate by Day of Week")
plt.ylabel("Fraud Rate")
plt.xticks(rotation=45)
plt.show()

# 3. Gender
gender_dist = df.groupby("Gender")["Is_Fraud"].mean()

plt.figure()
plt.bar(["Male", "Female"], gender_dist.values)
plt.title("Fraud Rate by Gender")
plt.ylabel("Fraud Rate")
plt.show()

# 4. Device Type
if "Device_Type" in df.columns:
    device_dist = df.groupby("Device_Type")["Is_Fraud"].mean()

    plt.figure(figsize=(8,4))
    plt.bar(device_dist.index.astype(str), device_dist.values)
    plt.title("Fraud Rate by Device Type")
    plt.ylabel("Fraud Rate")
    plt.xticks(rotation=45)
    plt.show()

# 5. Time of Day
if "Transaction_Time" in df.columns:
    df["Hour"] = pd.to_datetime(df["Transaction_Time"], unit="s").dt.hour

    def period(h):
        if h < 12:
            return "Morning"
        elif h < 17:
            return "Afternoon"
        elif h < 21:
            return "Evening"
        return "Night"

    df["Period"] = df["Hour"].apply(period)

    time_dist = df.groupby("Period")["Is_Fraud"].mean().reindex(
        ["Morning", "Afternoon", "Evening", "Night"]
    )

    plt.figure()
    plt.bar(time_dist.index, time_dist.values)
    plt.title("Fraud Rate by Time of Day")
    plt.ylabel("Fraud Rate")
    plt.show()

# -----------------------------
# Save dataset + feature order
# -----------------------------
X = df.drop(columns=["Is_Fraud"])
joblib.dump(X.columns.tolist(), "feature_order.pkl")

df.to_csv("cleaned_bankFraud.csv", index=False)

print("Preprocessing complete")