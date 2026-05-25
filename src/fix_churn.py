"""
RetailPulse — Stage 4a FIX: Churn model without leakage
The fix: remove 'recency' from features since it directly encodes the churn label.
The model must predict churn from BEHAVIORAL signals only.
"""

import os, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model      import LogisticRegression
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import shap

os.makedirs("models",         exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv("data/rfm_features.csv")

# ── KEY FIX: remove recency — it leaks the churn label ──────
# Recency = days since last purchase. Churn = recency >= 90.
# Including recency is like handing the model the answer sheet.
# Real-world scenario: you want to predict churn BEFORE 90 days
# passes, using only purchase behavior signals.
FEATURES = [
    "frequency",            # how often they buy
    "monetary",             # how much they spend
    "unique_products",      # product variety
    "avg_unit_price",       # price sensitivity
    "lifespan_days",        # how long they've been a customer
    "avg_order_value",      # spend per visit
    "purchase_rate",        # orders per active day
    "revenue_per_product",  # spend breadth
]
TARGET = "churned"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Features used: {FEATURES}\n")

# ── Logistic Regression baseline ────────────────────────────
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
print(f"Logistic Regression AUC: {lr_auc:.4f}")

# ── XGBoost ──────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)

xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_proba)
print(f"XGBoost AUC            : {xgb_auc:.4f}")
print(classification_report(y_test, xgb_preds,
      target_names=["Active", "Churned"]))

cv_scores = cross_val_score(xgb, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
print(f"5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Save
joblib.dump({"model": xgb, "scaler": scaler, "features": FEATURES},
            "models/churn_model.pkl")
print("\nModel saved: models/churn_model.pkl")

# ── Charts ───────────────────────────────────────────────────
# ROC curve
fig, ax = plt.subplots(figsize=(7, 5))
for name, proba in [("Logistic Regression", lr_proba), ("XGBoost", xgb_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, linewidth=2,
            label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
ax.plot([0,1],[0,1], "--", color="gray", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Churn (no leakage)", fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig("outputs/charts/churn_roc.png"); plt.close()

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, xgb_preds),
    display_labels=["Active", "Churned"]
).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — XGBoost", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/charts/churn_confusion.png"); plt.close()

# SHAP
print("Computing SHAP values...")
explainer = shap.TreeExplainer(xgb)
shap_vals = explainer.shap_values(X_test)
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (no leakage)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/charts/churn_shap.png", bbox_inches="tight"); plt.close()

# Probability distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(xgb_proba[y_test==0], bins=40, alpha=0.6, color="#22C55E", label="Active")
ax.hist(xgb_proba[y_test==1], bins=40, alpha=0.6, color="#EF4444", label="Churned")
ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Predicted Churn Probability")
ax.set_ylabel("Count")
ax.set_title("Churn Probability Distribution (no leakage)",
             fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig("outputs/charts/churn_prob_dist.png"); plt.close()

print(f"\nFixed churn model complete.")
print(f"  XGBoost AUC : {xgb_auc:.4f}  (realistic — no leakage)")
print(f"  Baseline AUC: {lr_auc:.4f}")
print(f"  Lift over baseline: +{(xgb_auc - lr_auc)*100:.1f}%")