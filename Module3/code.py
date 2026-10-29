"""
Loan Default Prediction System
A Comparative Study of Classification Algorithms
------------------------------------------------
Author: Ojas Sharma  |  Registration No.: RA2411028030022

Reproduces the report: load & clean -> class balance -> correlations ->
train/test split & scaling -> Logistic Regression / Random Forest / SVM / XGBoost
-> model comparison (metrics + ROC) -> best-model confusion matrix
-> feature importance -> conclusion.

Files expected in the same folder:
    Module 3 Project Data.csv   Index, Employed, Bank Balance, Annual Salary, Defaulted?

Run:   python code.py
Needs: pip install pandas numpy matplotlib seaborn scikit-learn xgboost
Output: figures in ./figures/, model comparison in ./model_results.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

DATA_FILE = "Module 3 Project Data.csv"
SHOW_PLOTS = False            # True = also open each figure in a window
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

if not SHOW_PLOTS:
    matplotlib.use("Agg")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"  saved {OUT_DIR}/{name}")


# ---------------------------------------------------------------- 1. Load & clean
print("1. Loading and cleaning data")
df = pd.read_csv(DATA_FILE)
df = df.drop(columns=["Index"])
df = df.rename(columns={"Bank Balance": "Bank_Balance",
                        "Annual Salary": "Annual_Salary",
                        "Defaulted?": "Defaulted"})
print(df.head())
df.info()
print("Missing values per column:\n", df.isnull().sum())

# ---------------------------------------------------------------- 2. Target
print("\n2. Class distribution")
print(df["Defaulted"].value_counts(normalize=True))

plt.figure(figsize=(5, 4))
sns.countplot(x="Defaulted", data=df)
plt.title("Class Distribution: Loan Default")
plt.xlabel("Defaulted (0 = No, 1 = Yes)")
plt.ylabel("Count")
save("01_class_distribution.png")

# ---------------------------------------------------------------- 3. Correlations
print("\n3. Feature correlations")
plt.figure(figsize=(5, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
save("02_correlation_heatmap.png")

# ---------------------------------------------------------------- 4. Split & scale
print("\n4. Train/test split and scaling")
X = df.drop(columns=["Defaulted"])
y = df["Defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Train:", X_train.shape, " Test:", X_test.shape)

# ---------------------------------------------------------------- 5. Models
print("\n5. Training the four models")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss", random_state=42
    ),
}

# Distance/gradient-based models get scaled features; tree models use raw values
needs_scaling = {"Logistic Regression", "SVM (RBF)"}

results = {}
roc_data = {}

for name, model in models.items():
    if name in needs_scaling:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "cm": confusion_matrix(y_test, y_pred),
    }
    roc_data[name] = roc_curve(y_test, y_prob)[:2] + (roc_auc_score(y_test, y_prob),)

    print(name)
    print(results[name]["cm"])
    print()

# ---------------------------------------------------------------- 6. Compare
print("6. Model comparison")
results_df = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "cm"}
                           for k, v in results.items()}).T
results_df = results_df.sort_values("roc_auc", ascending=False)
print(results_df.round(4))
results_df.to_csv("model_results.csv")

results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", figsize=(8, 4.5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.legend(loc="lower right", fontsize=8)
save("03_model_comparison.png")

plt.figure(figsize=(6, 5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend(loc="lower right", fontsize=8)
save("04_roc_curves.png")

best_model_name = results_df["roc_auc"].idxmax()
print("\nBest model (by ROC-AUC):", best_model_name)

# ---------------------------------------------------------------- 7. Best model
print("\n7. Confusion matrix of the best model")
best_cm = results[best_model_name]["cm"]

plt.figure(figsize=(4.5, 4))
sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
save("05_best_confusion_matrix.png")

# ---------------------------------------------------------------- 8. Importance
print("\n8. Feature importance")
for name, fname in [("Random Forest", "06_importance_random_forest.png"),
                    ("XGBoost", "07_importance_xgboost.png")]:
    importances = pd.Series(models[name].feature_importances_,
                            index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(5, 4))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Feature Importance - {name}")
    plt.xlabel("Importance")
    save(fname)
    print(f"{name}:\n{importances.round(4)}\n")

# ---------------------------------------------------------------- 9. Conclusion
best = results[best_model_name]
print("9. Conclusion")
print(f"- Defaults make up only {y.mean():.1%} of customers, so accuracy alone is misleading.")
print(f"- {best_model_name} has the best ROC-AUC ({best['roc_auc']:.2f}) "
      f"with recall {best['recall']:.0%}.")
print("- Bank balance is the strongest driver of default risk, followed by annual salary;")
print("  employment status adds little once balance and salary are known.")
