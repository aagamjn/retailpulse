"""
RetailPulse — Stage 4c: Customer Segmentation
K-Means clustering on RFM space
Run: python model_segmentation.py
Outputs:
  models/segmentation_model.pkl
  data/rfm_with_clusters.csv
  outputs/charts/seg_*.png
"""

import os, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing  import StandardScaler
from sklearn.cluster        import KMeans
from sklearn.metrics        import silhouette_score

os.makedirs("models",         exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
sns.set_theme(style="whitegrid")

# ── Load features ────────────────────────────────────────────
df = pd.read_csv("data/rfm_features.csv")
RFM_COLS = ["recency", "frequency", "monetary"]

X = df[RFM_COLS].copy()

# Log-transform monetary + frequency (heavy right skew)
X["frequency"] = np.log1p(X["frequency"])
X["monetary"]  = np.log1p(X["monetary"])

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── Find optimal K via elbow + silhouette ───────────────────
print("Finding optimal K...")
inertias, sil_scores = [], []
K_range = range(2, 9)

for k in K_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, lbl))
    print(f"  K={k}  inertia={km.inertia_:,.0f}  silhouette={sil_scores[-1]:.4f}")

best_k = K_range[np.argmax(sil_scores)]
print(f"\nBest K by silhouette: {best_k}")


# ── Final model ──────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_scaled)


# ── Name clusters by RFM centroid ───────────────────────────
centers = pd.DataFrame(
    scaler.inverse_transform(km_final.cluster_centers_),
    columns=["recency", "frequency_log", "monetary_log"]
)
centers["frequency"] = np.expm1(centers["frequency_log"]).round(1)
centers["monetary"]  = np.expm1(centers["monetary_log"]).round(0)
centers["cluster"]   = range(best_k)

# Sort by monetary desc to assign names
centers_sorted = centers.sort_values("monetary", ascending=False).reset_index(drop=True)
NAMES = ["Champions", "Loyal Customers", "At Risk", "Hibernating", "Lost"][: best_k]
name_map = dict(zip(centers_sorted["cluster"], NAMES))
df["cluster_name"] = df["cluster"].map(name_map)

print("\nCluster profiles:")
profile = df.groupby("cluster_name")[RFM_COLS + ["churned"]].agg({
    "recency":   "mean",
    "frequency": "mean",
    "monetary":  "mean",
    "churned":   "mean"
}).round(2)
profile.columns = ["Avg Recency", "Avg Frequency", "Avg Monetary", "Churn Rate"]
print(profile.to_string())

# Save
df.to_csv("data/rfm_with_clusters.csv", index=False)
joblib.dump({"model": km_final, "scaler": scaler, "name_map": name_map},
            "models/segmentation_model.pkl")
print("\nSaved: data/rfm_with_clusters.csv")
print("Saved: models/segmentation_model.pkl")


# ══ CHARTS ══════════════════════════════════════════════════

COLORS = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"][:best_k]

# Chart A: Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(K_range), inertias, marker="o", color="#6366F1", linewidth=2)
axes[0].set_title("Elbow Method", fontsize=12, fontweight="bold")
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
axes[1].plot(list(K_range), sil_scores, marker="s", color="#10B981", linewidth=2)
axes[1].axvline(best_k, color="red", linestyle="--", linewidth=1, label=f"Best K={best_k}")
axes[1].set_title("Silhouette Score", fontsize=12, fontweight="bold")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Score"); axes[1].legend()
plt.suptitle("Optimal K Selection", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/charts/seg_optimal_k.png"); plt.close()

# Chart B: RFM scatter — Recency vs Monetary
fig, ax = plt.subplots(figsize=(9, 6))
for i, (name, grp) in enumerate(df.groupby("cluster_name")):
    ax.scatter(grp["recency"], np.log1p(grp["monetary"]),
               alpha=0.4, s=20, label=name, color=COLORS[i % len(COLORS)])
ax.set_xlabel("Recency (days since last purchase)")
ax.set_ylabel("log(Monetary £)")
ax.set_title("Customer Segments: Recency vs Monetary", fontsize=13, fontweight="bold")
ax.legend(title="Segment"); plt.tight_layout()
plt.savefig("outputs/charts/seg_scatter.png"); plt.close()

# Chart C: Cluster size
seg_counts = df["cluster_name"].value_counts()
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(seg_counts.index, seg_counts.values,
              color=COLORS[:len(seg_counts)])
ax.bar_label(bars, padding=3)
ax.set_title("Customer Count per Segment", fontsize=13, fontweight="bold")
ax.set_ylabel("Customers"); plt.tight_layout()
plt.savefig("outputs/charts/seg_counts.png"); plt.close()

# Chart D: RFM heatmap per cluster
heat = df.groupby("cluster_name")[RFM_COLS].mean().round(1)
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax)
ax.set_title("RFM Heatmap by Segment", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/charts/seg_rfm_heatmap.png"); plt.close()

print(f"\nStage 4c complete — Segmentation done.")
print(f"  Optimal clusters : {best_k}")
print(f"  Charts saved to outputs/charts/seg_*.png")