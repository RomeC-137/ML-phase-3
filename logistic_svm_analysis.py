# 📦 Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, log_loss
import warnings
warnings.filterwarnings("ignore")

# 📂 Load and Prepare Data
df = pd.read_csv("student-mat.csv")
df.columns = df.columns.str.strip()
df["pass"] = (df["G3"] >= 10).astype(int)
df = df.drop(columns=["G1", "G2", "G3"])
df = pd.get_dummies(df, drop_first=True)

# 🎯 Features and Target
X = df.drop("pass", axis=1)
y = df["pass"]

# 🔀 Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚖️ Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
log_likelihood = -log_loss(y_train, log_reg.predict_proba(X_train_scaled), normalize=False)
avg_log_loss = log_loss(y_train, log_reg.predict_proba(X_train_scaled))

log_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_log),
    "Precision": precision_score(y_test, y_pred_log),
    "Recall": recall_score(y_test, y_pred_log),
    "Log-Likelihood": log_likelihood,
    "Average Log Loss": avg_log_loss,
    "Confusion Matrix": confusion_matrix(y_test, y_pred_log).tolist()
}

# 🤖 SVM Model
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

svm_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_svm),
    "Precision": precision_score(y_test, y_pred_svm),
    "Recall": recall_score(y_test, y_pred_svm),
    "Confusion Matrix": confusion_matrix(y_test, y_pred_svm).tolist()
}

# 📊 Final Summary
print("\n🔍 Logistic Regression Results:")
for k, v in log_metrics.items():
    print(f"{k}: {v}")

print("\n🤖 SVM Results:")
for k, v in svm_metrics.items():
    print(f"{k}: {v}")
