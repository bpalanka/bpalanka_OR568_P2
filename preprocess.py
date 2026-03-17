# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# Download latest version from Kaggle
path = kagglehub.dataset_download("orangelmendez/bank-fraud")
print("Path to dataset files:", path)

# Load dataset 
file_path = os.path.join(path, "new_bank_fraud_detection.csv")
df = pd.read_csv(file_path)

# Drop first column automatically
df = df.iloc[:, 1:]

# Inspect the data
print(df.info())
print(df.head())

# Remove identifier columns if they exist
cols_to_drop = ['Transaction_ID', 'Merchant_ID']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Convert Gender to dummy (1 = Female, 0 = Male)
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)

# Convert target to categorical
df['Is_Fraud'] = df['Is_Fraud'].astype('category')

# Convert categorical variables
factor_vars = [
    "State", "Bank_Branch", "Account_Type",
    "Transaction_Type", "Merchant_Category",
    "Transaction_Device", "Device_Type",
    "Transaction_Currency"
]

for col in factor_vars:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Handle date column
if 'Transaction_Date' in df.columns:
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], format='mixed', dayfirst=False)

    df['Transaction_Year'] = df['Transaction_Date'].dt.year
    df['Transaction_Month'] = df['Transaction_Date'].dt.month
    df['Transaction_Day'] = df['Transaction_Date'].dt.day

    df = df.drop(columns=['Transaction_Date'])

print(df.head())

# Separate predictors and target
X = df.drop(columns=['Is_Fraud'])
y = df['Is_Fraud']

# Numeric predictors only
X_num = X.select_dtypes(include=[np.number])

# Check class balance
print(y.value_counts())
print(y.value_counts(normalize=True))

# Check skewness
skew_values = X_num.apply(skew)
print(skew_values)

# Example variable skewness
if 'Transaction_Amount' in X_num.columns:
    print("Skewness of Transaction_Amount:", skew(X_num['Transaction_Amount']))

# Remove zero variance predictors
vt = VarianceThreshold()
X_num_var = vt.fit_transform(X_num)
X_num = X_num[X_num.columns[vt.get_support(indices=True)]]

# Box-Cox transformation + scaling
pt = PowerTransformer(method='box-cox', standardize=True)
X_trans = pd.DataFrame(
    pt.fit_transform(X_num + 1e-6),
    columns=X_num.columns
)

# Histogram before/after transformation
if 'Transaction_Amount' in X_num.columns:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(X_num['Transaction_Amount'], bins=30)
    plt.title("Original Transaction Amount")

    plt.subplot(1, 2, 2)
    plt.hist(X_trans['Transaction_Amount'], bins=30)
    plt.title("Transformed Transaction Amount")

    print("Skewness before:", skew(X_num['Transaction_Amount']))
    print("Skewness after:", skew(X_trans['Transaction_Amount']))

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_trans)

percent_variance = pca.explained_variance_ratio_ * 100
print("Variance explained by first 5 components:", percent_variance[:5])

plt.figure()
plt.plot(np.arange(1, len(percent_variance)+1), percent_variance, marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Percentage of Variance")
plt.title("PCA - Fraud Dataset")

# Correlation matrix
corr_matrix = X_trans.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix")

# Remove highly correlated predictors
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
X_filtered = X_trans.drop(columns=to_drop)

# Correlation with target
cor_with_target = X_trans.corrwith(y.cat.codes)
print("Correlation with target (Is_Fraud):")
print(cor_with_target.sort_values(ascending=False))

# Boxplots
selected_cols = ['Transaction_Amount', 'Account_Balance', 'Age']

plt.figure(figsize=(15, 5))
for i, col in enumerate([c for c in selected_cols if c in X_num.columns], 1):
    plt.subplot(1, len(selected_cols), i)
    sns.boxplot(x=y, y=X_num[col])
    plt.title(col)

# Histograms
plt.figure(figsize=(15, 5))
for i, col in enumerate([c for c in selected_cols if c in X_num.columns], 1):
    plt.subplot(1, len(selected_cols), i)
    plt.hist(X_num[col], bins=30)
    plt.title(col)

# Export cleaned CSV
df.to_csv("cleaned_bankFraud.csv", index=False)

print("Cleaned file saved as cleaned_bankFraud.csv")