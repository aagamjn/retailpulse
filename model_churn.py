"""
RetailPulse — Stage 4a: Churn Prediction
XGBoost classifier on RFM features + SHAP explainability
Run: python model_churn.py
Outputs:
  models/churn_model.pkl
  outputs/charts/churn_*.png
"""

import os, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

os.makedirs("models",          exist_ok=True)
os.makedirs("outputs/charts",  exist_ok=True)
sns.set_theme(style="whitegrid")

# ── Load features ────────────────────────────────────────────
df = pd.read_csv("data/rfm_features.csv")

FEATURES = [
    "recency", "frequency", "monetary",
    "unique_products", "avg_unit_price",
    "lifespan_days", "avg_order_value",
    "purchase_rate", "revenue_per_product"
]
TARGET = "churned"

X = df[FEATURES]
y = df[TARGET]

# ── Train / test split (stratified) ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ── Baseline: Logistic Regression ───────────────────────────
print("\n--- Logistic Regression (baseline) ---")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
lr_preds = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
print(f"AUC-ROC : {lr_auc:.4f}")
print(classification_report(y_test, lr_preds,
      target_names=["Active", "Churned"]))


# ── Champion: XGBoost ────────────────────────────────────────
print("\n--- XGBoost ---")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)

xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_proba)
print(f"AUC-ROC : {xgb_auc:.4f}")
print(classification_report(xgb_preds, y_test,
      target_names=["Active", "Churned"]))

# 5-fold CV
cv_scores = cross_val_score(xgb, X, y, cv=StratifiedKFold(5),
                             scoring="roc_auc")
print(f"5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ── Save model ───────────────────────────────────────────────
joblib.dump({"model": xgb, "scaler": scaler, "features": FEATURES},
            "models/churn_model.pkl")
print("\nModel saved: models/churn_model.pkl")


# ══ CHARTS ══════════════════════════════════════════════════

# Chart A: ROC Curve comparison
fig, ax = plt.subplots(figsize=(7, 5))
for name, proba in [("Logistic Regression", lr_proba), ("XGBoost", xgb_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"--", color="gray", label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Churn Prediction", fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig("outputs/charts/churn_roc.png"); plt.close()

# Chart B: Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, xgb_preds),
    display_labels=["Active", "Churned"]
).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — XGBoost", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/charts/churn_confusion.png"); plt.close()

# Chart C: SHAP feature importance
print("\nComputing SHAP values...")
explainer  = shap.TreeExplainer(xgb)
shap_vals  = explainer.shap_values(X_test)
fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/charts/churn_shap.png", bbox_inches="tight"); plt.close()

# Chart D: Churn probability distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(xgb_proba[y_test==0], bins=40, alpha=0.6, color="#22C55E", label="Active")
ax.hist(xgb_proba[y_test==1], bins=40, alpha=0.6, color="#EF4444", label="Churned")
ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Threshold 0.5")
ax.set_xlabel("Predicted Churn Probability")
ax.set_ylabel("Count")
ax.set_title("Churn Probability Distribution", fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig("outputs/charts/churn_prob_dist.png"); plt.close()

print("\nStage 4a complete — Churn model done.")
print(f"  XGBoost AUC : {xgb_auc:.4f}")
print(f"  Baseline AUC: {lr_auc:.4f}")
print(f"  Improvement : +{(xgb_auc - lr_auc)*100:.1f}% over baseline")
print("  Charts saved to outputs/charts/churn_*.png")