# =========================
# FRAUD DETECTION NN   
# =========================

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
    roc_curve, f1_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("cleaned_bankFraud.csv")
df.columns = df.columns.str.strip()

# Remove EDA-only columns
df = df.drop(columns=["DayName", "Period"], errors="ignore")

# =========================
# TARGET + FEATURES
# =========================
y = df["Is_Fraud"].values
X = df.drop(columns=["Is_Fraud"])

# Keep numeric only
X = X.select_dtypes(include=[np.number]).fillna(0)

# =========================
# MODEL BUILDER
# =========================
def build_model(input_dim):
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.2),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================
# CROSS VALIDATION
# =========================
print("\nRunning 5-Fold Cross Validation...\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_scores = []
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    model = build_model(X_tr.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )

    y_val_probs = model.predict(X_val).flatten()
    y_val_pred = (y_val_probs > 0.5).astype(int)

    roc = roc_auc_score(y_val, y_val_probs)
    f1 = f1_score(y_val, y_val_pred)

    roc_scores.append(roc)
    f1_scores.append(f1)

    print(f"ROC AUC: {roc:.4f}, F1: {f1:.4f}\n")

print("Average ROC AUC:", np.mean(roc_scores))
print("Average F1 Score:", np.mean(f1_scores))

# =========================
# FINAL TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save artifacts
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_order.pkl")

# =========================
# FINAL MODEL TRAINING
# =========================
model = build_model(X_train.shape[1])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# TEST SET EVALUATION
# =========================
y_probs = model.predict(X_test).flatten()

# Threshold optimization (F1)
prec, rec, thr = precision_recall_curve(y_test, y_probs)
f1 = (2 * prec * rec) / (prec + rec + 1e-8)

best_thr = thr[np.argmax(f1[:-1])]
print("\nBest Threshold:", best_thr)

y_pred = (y_probs > best_thr).astype(int)

# Metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_probs)
print("\nTest ROC AUC:", roc_auc)

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================
# PRECISION-RECALL CURVE
# =========================
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# =========================
# SAVE MODEL
# =========================
model.save("fraud_neural_network.h5")

print("\nModel saved as fraud_neural_network.h5")