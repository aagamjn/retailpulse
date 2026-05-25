"""
RetailPulse — Stage 3: Feature Engineering
Builds RFM features, churn labels, and time-series aggregates.
Run: python feature_engineering.py
Outputs:
  data/rfm_features.csv        — churn model input
  data/timeseries_weekly.csv   — forecasting model input
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# ── CONFIG ──────────────────────────────────────────────────
DB_USER     = "root"
DB_PASSWORD = "aagam"
DB_HOST     = "localhost"
DB_NAME     = "retailpulse"
SNAPSHOT    = "2011-12-10"   # day after last transaction = "today"
CHURN_DAYS  = 90             # no purchase in 90 days = churned
# ────────────────────────────────────────────────────────────

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

import os; os.makedirs("data", exist_ok=True)


# ════════════════════════════════════════════════════════════
# PART A — RFM + Churn features
# ════════════════════════════════════════════════════════════

print("Building RFM features...")

rfm_query = """
    SELECT
        i.customer_id,
        DATEDIFF(%s, MAX(i.invoice_date))           AS recency,
        COUNT(DISTINCT i.invoice_no)                AS frequency,
        ROUND(SUM(ii.quantity * ii.unit_price), 2)  AS monetary,
        COUNT(DISTINCT ii.stock_code)               AS unique_products,
        ROUND(AVG(ii.unit_price), 2)                AS avg_unit_price,
        MAX(i.invoice_date)                         AS last_purchase_date,
        MIN(i.invoice_date)                         AS first_purchase_date
    FROM invoices i
    JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
    GROUP BY i.customer_id
"""
df = pd.read_sql(rfm_query, engine, params=(SNAPSHOT,))
print(f"  Customers loaded: {len(df):,}")


# ── Derived features ────────────────────────────────────────
# Customer lifespan in days
df["lifespan_days"] = (
    pd.to_datetime(df["last_purchase_date"]) -
    pd.to_datetime(df["first_purchase_date"])
).dt.days

# Average order value
df["avg_order_value"] = (df["monetary"] / df["frequency"]).round(2)

# Purchase rate: orders per active day (avoid div/0)
df["purchase_rate"] = (
    df["frequency"] / (df["lifespan_days"] + 1)
).round(4)

# Revenue per product variety
df["revenue_per_product"] = (
    df["monetary"] / df["unique_products"]
).round(2)


# ── Churn label ─────────────────────────────────────────────
# Decision: customer who hasn't purchased in last 90 days = churned (1)
# We only label customers whose last purchase was in the
# "observation window" (not brand new customers in last 90 days)
df["churned"] = (df["recency"] >= CHURN_DAYS).astype(int)

churn_rate = df["churned"].mean() * 100
print(f"  Churn threshold: {CHURN_DAYS} days")
print(f"  Churned customers: {df['churned'].sum():,} ({churn_rate:.1f}%)")
print(f"  Active customers:  {(df['churned']==0).sum():,} ({100-churn_rate:.1f}%)")


# ── RFM Scores (1-5 scale, for segmentation) ────────────────
# Recency: lower is better → reverse rank
df["r_score"] = pd.qcut(df["recency"],   q=5, labels=[5,4,3,2,1]).astype(int)
df["f_score"] = pd.qcut(df["frequency"].rank(method="first"),
                         q=5, labels=[1,2,3,4,5]).astype(int)
df["m_score"] = pd.qcut(df["monetary"].rank(method="first"),
                         q=5, labels=[1,2,3,4,5]).astype(int)

df["rfm_score"] = df["r_score"] + df["f_score"] + df["m_score"]


# ── Segment labels (for K-Means context) ────────────────────
def segment(row):
    if row["rfm_score"] >= 13:
        return "Champion"
    elif row["rfm_score"] >= 10:
        return "Loyal"
    elif row["rfm_score"] >= 7:
        return "At Risk"
    elif row["rfm_score"] >= 4:
        return "Hibernating"
    else:
        return "Lost"

df["segment"] = df.apply(segment, axis=1)

seg_counts = df["segment"].value_counts()
print("\n  Segment distribution:")
for seg, cnt in seg_counts.items():
    print(f"    {seg:<15} {cnt:>5} customers  ({cnt/len(df)*100:.1f}%)")


# ── Save ─────────────────────────────────────────────────────
feature_cols = [
    "customer_id",
    "recency", "frequency", "monetary",
    "unique_products", "avg_unit_price",
    "lifespan_days", "avg_order_value",
    "purchase_rate", "revenue_per_product",
    "r_score", "f_score", "m_score", "rfm_score",
    "segment", "churned"
]
df[feature_cols].to_csv("data/rfm_features.csv", index=False)
print(f"\n  Saved: data/rfm_features.csv  ({len(df):,} rows, {len(feature_cols)} cols)")


# ════════════════════════════════════════════════════════════
# PART B — Weekly time-series for forecasting
# ════════════════════════════════════════════════════════════

print("\nBuilding weekly time-series features...")

weekly_query = """
    SELECT
        DATE_FORMAT(
            DATE_SUB(i.invoice_date, INTERVAL WEEKDAY(i.invoice_date) DAY),
        '%%Y-%%m-%%d')                                     AS week_start,
        ROUND(SUM(ii.quantity * ii.unit_price), 2)      AS revenue,
        COUNT(DISTINCT i.invoice_no)                    AS orders,
        COUNT(DISTINCT i.customer_id)                   AS customers,
        SUM(ii.quantity)                                AS units_sold
    FROM invoices i
    JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
    GROUP BY week_start
    ORDER BY week_start
"""
ts = pd.read_sql(weekly_query, engine)
ts["week_start"] = pd.to_datetime(ts["week_start"])
ts = ts.sort_values("week_start").reset_index(drop=True)

# Lag features (Prophet doesn't need these, XGBoost does)
for lag in [1, 2, 4, 8]:
    ts[f"revenue_lag_{lag}w"] = ts["revenue"].shift(lag)

# Rolling averages
ts["revenue_roll4"]  = ts["revenue"].shift(1).rolling(4).mean().round(2)
ts["revenue_roll8"]  = ts["revenue"].shift(1).rolling(8).mean().round(2)

# Week number and year for seasonality
ts["week_num"] = ts["week_start"].dt.isocalendar().week.astype(int)
ts["year"]     = ts["week_start"].dt.year

ts.to_csv("data/timeseries_weekly.csv", index=False)
print(f"  Saved: data/timeseries_weekly.csv  ({len(ts)} weeks)")
print(f"  Date range: {ts['week_start'].min().date()} → {ts['week_start'].max().date()}")


print("\nStage 3 complete. Features ready for modeling.")
print("  → data/rfm_features.csv      (churn + segmentation model)")
print("  → data/timeseries_weekly.csv (sales forecasting model)")