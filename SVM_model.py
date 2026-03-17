# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# Load cleaned dataset
df = pd.read_csv("cleaned_bankFraud.csv")

# Separate predictors and target
X = df.drop(columns=["Is_Fraud", "Transaction_Description", "Transaction_Location"])
y = df["Is_Fraud"]

# Numeric columns
numeric_cols = ["Age", "Transaction_Time", "Transaction_Amount", "Account_Balance"]

# Convert Transaction_Time to seconds
X["Transaction_Time"] = pd.to_timedelta(X["Transaction_Time"]).dt.total_seconds()

# One-hot encode categorical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Scale numeric columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
svm_model = SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=42
)

print("Training SVM...")
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(svm_model, "fraud_svm_model.pkl")
print("Model saved: fraud_svm_model.pkl")