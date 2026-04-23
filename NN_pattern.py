#Neural Network Pattern 
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load dataset + model artifacts
# -----------------------------
df = pd.read_csv("cleaned_bankFraud.csv")

model = load_model("fraud_neural_network.h5")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")

# -----------------------------
# Align features correctly
# -----------------------------
X = df[feature_order]
X_scaled = scaler.transform(X)

df["Pred_Prob"] = model.predict(X_scaled).flatten()

# -----------------------------
# 1. Weekday vs Weekend (PREDICTION)
# -----------------------------
df["Is_Weekend"] = df["Transaction_DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)

weekend_pred = df.groupby("Is_Weekend")["Pred_Prob"].mean()

plt.figure()
plt.bar(["Weekday", "Weekend"], weekend_pred.values)
plt.title("Predicted Fraud Risk: Weekday vs Weekend")
plt.ylabel("Average Fraud Probability")
plt.show()

# -----------------------------
# 2. Day of Week (Mon–Sun)
# -----------------------------
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

dow_pred = df.groupby("DayName")["Pred_Prob"].mean()

order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_pred = dow_pred.reindex(order)

plt.figure(figsize=(8,4))
plt.bar(dow_pred.index, dow_pred.values)
plt.title("Predicted Fraud Risk by Day of Week")
plt.ylabel("Average Fraud Probability")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 3. Gender
# -----------------------------
gender_pred = df.groupby("Gender")["Pred_Prob"].mean()

plt.figure()
plt.bar(["Male", "Female"], gender_pred.values)
plt.title("Predicted Fraud Risk by Gender")
plt.ylabel("Average Fraud Probability")
plt.show()

# -----------------------------
# 4. Device Type
# -----------------------------
if "Device_Type" in df.columns:
    device_pred = df.groupby("Device_Type")["Pred_Prob"].mean()

    plt.figure(figsize=(8,4))
    plt.bar(device_pred.index.astype(str), device_pred.values)
    plt.title("Predicted Fraud Risk by Device Type")
    plt.ylabel("Average Fraud Probability")
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------
# 5. Time of Day
# -----------------------------
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

time_pred = df.groupby("Period")["Pred_Prob"].mean().reindex(
    ["Morning", "Afternoon", "Evening", "Night"]
)

plt.figure()
plt.bar(time_pred.index, time_pred.values)
plt.title("Predicted Fraud Risk by Time of Day")
plt.ylabel("Average Fraud Probability")
plt.show()