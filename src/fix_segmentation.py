"""
RetailPulse — Stage 4c FIX: Segmentation with forced K=4
K=2 is statistically optimal but business-useless.
We force K=4 for a richer, actionable customer story.
Interview answer: "Silhouette favored K=2, but K=4 gives
distinct actionable segments — Champions, Loyal, At Risk,
Lost — which map directly to marketing strategies."
"""

import os, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster       import KMeans
from sklearn.metrics       import silhouette_score

os.makedirs("models",         exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv("data/rfm_features.csv")
RFM_COLS = ["recency", "frequency", "monetary"]

X = df[RFM_COLS].copy()
X["frequency"] = np.log1p(X["frequency"])
X["monetary"]  = np.log1p(X["monetary"])

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Force K=4 — business decision over statistical optimum
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, df["cluster"])
print(f"K=4 silhouette score: {sil:.4f}")

# ── Name clusters by RFM centroid ranking ───────────────────
centers = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_),
    columns=["recency_scaled", "freq_log", "mon_log"]
)
centers["recency"]   = centers["recency_scaled"]
centers["frequency"] = np.expm1(centers["freq_log"]).round(1)
centers["monetary"]  = np.expm1(centers["mon_log"]).round(0)
centers["cluster"]   = range(K)

# Rank: low recency + high monetary = best
centers["rank_score"] = (
    -centers["recency"] +
     centers["monetary"] / centers["monetary"].max() * 100 +
     centers["frequency"] * 5
)
centers_sorted = centers.sort_values("rank_score", ascending=False).reset_index(drop=True)

SEGMENT_NAMES  = ["Champions", "Loyal Customers", "At Risk", "Lost"]
SEGMENT_ACTIONS = [
    "Reward with loyalty perks, ask for reviews",
    "Upsell premium products, increase frequency",
    "Send win-back campaigns, offer discounts",
    "Minimal spend — low re-engagement ROI"
]
name_map   = dict(zip(centers_sorted["cluster"], SEGMENT_NAMES))
action_map = dict(zip(SEGMENT_NAMES, SEGMENT_ACTIONS))

df["cluster_name"]   = df["cluster"].map(name_map)
df["action"]         = df["cluster_name"].map(action_map)

# ── Profile ──────────────────────────────────────────────────
print("\nSegment profiles:")
profile = df.groupby("cluster_name").agg(
    customers   = ("customer_id", "count"),
    avg_recency = ("recency",     "mean"),
    avg_freq    = ("frequency",   "mean"),
    avg_monetary= ("monetary",    "mean"),
    churn_rate  = ("churned",     "mean")
).round(2)
print(profile.to_string())

# Save
df.to_csv("data/rfm_with_clusters.csv", index=False)
joblib.dump({"model": km, "scaler": scaler, "name_map": name_map},
            "models/segmentation_model.pkl")

# ── Charts ───────────────────────────────────────────────────
COLORS = ["#6366F1", "#10B981", "#F59E0B", "#EF4444"]

# Chart A: Scatter — Recency vs Monetary
fig, ax = plt.subplots(figsize=(9, 6))
for i, (name, grp) in enumerate(df.groupby("cluster_name")):
    ax.scatter(grp["recency"], np.log1p(grp["monetary"]),
               alpha=0.4, s=20, label=name, color=COLORS[i % 4])
ax.set_xlabel("Recency (days since last purchase)")
ax.set_ylabel("log(Monetary £)")
ax.set_title("Customer Segments: Recency vs Monetary (K=4)",
             fontsize=13, fontweight="bold")
ax.legend(title="Segment"); plt.tight_layout()
plt.savefig("outputs/charts/seg_scatter.png"); plt.close()

# Chart B: Segment sizes
seg_counts = df["cluster_name"].value_counts().reindex(SEGMENT_NAMES)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(seg_counts.index, seg_counts.values, color=COLORS)
ax.bar_label(bars, padding=3)
ax.set_title("Customer Count per Segment (K=4)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Customers"); plt.tight_layout()
plt.savefig("outputs/charts/seg_counts.png"); plt.close()

# Chart C: RFM heatmap
heat = df.groupby("cluster_name")[RFM_COLS].mean().round(1)
heat = heat.reindex(SEGMENT_NAMES)
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax)
ax.set_title("RFM Heatmap by Segment (K=4)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/charts/seg_rfm_heatmap.png"); plt.close()

# Chart D: Churn rate per segment
churn_by_seg = df.groupby("cluster_name")["churned"].mean().reindex(SEGMENT_NAMES) * 100
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(churn_by_seg.index, churn_by_seg.values, color=COLORS)
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_title("Churn Rate per Segment", fontsize=13, fontweight="bold")
ax.set_ylabel("Churn Rate (%)"); ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig("outputs/charts/seg_churn_rate.png"); plt.close()

print(f"\nFixed segmentation complete.")
print(f"  K=4 silhouette: {sil:.4f}")
print(f"  Segments: {SEGMENT_NAMES}")
print(f"  Charts saved to outputs/charts/seg_*.png")