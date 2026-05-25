"""
RetailPulse — Stage 2 EDA
Run: python eda.py
Generates 10 charts saved to outputs/charts/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sqlalchemy import create_engine

# ── CONFIG ──────────────────────────────────────────────────
from dotenv import load_dotenv
import os
load_dotenv()

DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_NAME     = os.getenv("DB_NAME")
CHART_DIR   = "outputs/charts"
# ────────────────────────────────────────────────────────────

os.makedirs(CHART_DIR, exist_ok=True)
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130


def q(sql):
    return pd.read_sql(sql, engine)


# ─── 1. Monthly Revenue Trend ───────────────────────────────
print("Chart 1: Monthly revenue trend...")
df = q("""
    SELECT DATE_FORMAT(i.invoice_date,'%%Y-%%m') AS month,
           ROUND(SUM(ii.quantity*ii.unit_price),2) AS revenue
    FROM invoices i JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
    GROUP BY month ORDER BY month
""")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["month"], df["revenue"], marker="o", linewidth=2, color="#2563EB")
ax.fill_between(range(len(df)), df["revenue"], alpha=0.1, color="#2563EB")
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["month"], rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Revenue")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_monthly_revenue.png"); plt.close()


# ─── 2. Top 10 Countries by Revenue ─────────────────────────
print("Chart 2: Revenue by country...")
df = q("""
    SELECT c.country,
           ROUND(SUM(ii.quantity*ii.unit_price),2) AS revenue
    FROM invoices i
    JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
    JOIN customers c ON i.customer_id=c.customer_id
    WHERE c.country != 'United Kingdom'
    GROUP BY c.country ORDER BY revenue DESC LIMIT 10
""")
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(df["country"][::-1], df["revenue"][::-1], color="#7C3AED")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
ax.set_title("Top 10 Countries by Revenue (excl. UK)", fontsize=13, fontweight="bold")
ax.set_xlabel("Revenue")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_revenue_by_country.png"); plt.close()


# ─── 3. Top 10 Products by Quantity ─────────────────────────
print("Chart 3: Top products...")
df = q("""
    SELECT p.description, SUM(ii.quantity) AS qty
    FROM invoice_items ii JOIN products p ON ii.stock_code=p.stock_code
    GROUP BY p.description ORDER BY qty DESC LIMIT 10
""")
df["description"] = df["description"].str[:35]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(df["description"][::-1], df["qty"][::-1], color="#059669")
ax.set_title("Top 10 Products by Quantity Sold", fontsize=13, fontweight="bold")
ax.set_xlabel("Units Sold")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_top_products.png"); plt.close()


# ─── 4. Orders by Day of Week ────────────────────────────────
print("Chart 4: Orders by day of week...")
df = q("""
    SELECT DAYNAME(invoice_date) AS day,
           DAYOFWEEK(invoice_date) AS day_num,
           COUNT(DISTINCT invoice_no) AS orders
    FROM invoices
    GROUP BY day, day_num ORDER BY day_num
""")
day_order = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
df["day"] = pd.Categorical(df["day"], categories=day_order, ordered=True)
df = df.sort_values("day")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(df["day"], df["orders"], color="#F59E0B")
ax.set_title("Orders by Day of Week", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Orders")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_orders_by_day.png"); plt.close()


# ─── 5. Purchase Frequency Distribution ─────────────────────
print("Chart 5: Purchase frequency...")
df = q("""
    SELECT COUNT(DISTINCT invoice_no) AS orders, COUNT(*) AS customers
    FROM invoices GROUP BY customer_id
""")
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df["orders"], bins=50, color="#DC2626", edgecolor="white")
ax.set_xlim(0, 60)
ax.set_title("Customer Purchase Frequency Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Orders per Customer")
ax.set_ylabel("Number of Customers")
pct_one = (df["orders"] == 1).mean() * 100
ax.axvline(1, color="black", linestyle="--", linewidth=1)
ax.text(2, ax.get_ylim()[1]*0.85, f"{pct_one:.1f}% bought only once", fontsize=10)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_purchase_frequency.png"); plt.close()


# ─── 6. Average Order Value by Month ────────────────────────
print("Chart 6: AOV trend...")
df = q("""
    SELECT DATE_FORMAT(i.invoice_date,'%%Y-%%m') AS month,
           ROUND(SUM(ii.quantity*ii.unit_price)/COUNT(DISTINCT i.invoice_no),2) AS aov
    FROM invoices i JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
    GROUP BY month ORDER BY month
""")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["month"], df["aov"], marker="s", color="#0891B2", linewidth=2)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["month"], rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.0f}"))
ax.set_title("Average Order Value (AOV) by Month", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_aov_trend.png"); plt.close()


# ─── 7. New vs Returning Customers ──────────────────────────
print("Chart 7: New vs returning...")
df = q("""
    WITH fp AS (
        SELECT customer_id, DATE_FORMAT(MIN(invoice_date),'%%Y-%%m') AS first_month
        FROM invoices GROUP BY customer_id
    )
    SELECT DATE_FORMAT(i.invoice_date,'%%Y-%%m') AS month,
           COUNT(DISTINCT CASE WHEN DATE_FORMAT(i.invoice_date,'%%Y-%%m')=fp.first_month
                               THEN i.customer_id END) AS new_customers,
           COUNT(DISTINCT CASE WHEN DATE_FORMAT(i.invoice_date,'%%Y-%%m')>fp.first_month
                               THEN i.customer_id END) AS returning_customers
    FROM invoices i JOIN fp ON i.customer_id=fp.customer_id
    GROUP BY month ORDER BY month
""")
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(df["month"], df["new_customers"],       label="New",       color="#6366F1")
ax.bar(df["month"], df["returning_customers"], label="Returning",
       bottom=df["new_customers"],             color="#A5B4FC")
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df["month"], rotation=45, ha="right", fontsize=8)
ax.set_title("New vs Returning Customers by Month", fontsize=13, fontweight="bold")
ax.set_ylabel("Customers"); ax.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/07_new_vs_returning.png"); plt.close()


# ─── 8. Revenue by Hour of Day ───────────────────────────────
print("Chart 8: Revenue by hour...")
df = q("""
    SELECT HOUR(i.invoice_date) AS hour,
           ROUND(SUM(ii.quantity*ii.unit_price),2) AS revenue
    FROM invoices i JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
    GROUP BY hour ORDER BY hour
""")
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(df["hour"], df["revenue"], color="#10B981")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
ax.set_title("Revenue by Hour of Day", fontsize=13, fontweight="bold")
ax.set_xlabel("Hour (24h)"); ax.set_ylabel("Revenue")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_revenue_by_hour.png"); plt.close()


# ─── 9. RFM Distribution Histograms ─────────────────────────
print("Chart 9: RFM distributions...")
df = q("""
    SELECT i.customer_id,
           DATEDIFF('2011-12-10', MAX(i.invoice_date)) AS recency,
           COUNT(DISTINCT i.invoice_no)                AS frequency,
           ROUND(SUM(ii.quantity*ii.unit_price),2)     AS monetary
    FROM invoices i JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
    GROUP BY i.customer_id
""")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(df["recency"],   bins=50, color="#EF4444"); axes[0].set_title("Recency (days)")
axes[1].hist(df["frequency"], bins=50, color="#3B82F6"); axes[1].set_title("Frequency (orders)")
axes[2].hist(df["monetary"].clip(upper=5000), bins=50, color="#10B981")
axes[2].set_title("Monetary (£, clipped 5K)")
fig.suptitle("RFM Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/09_rfm_distributions.png"); plt.close()


# ─── 10. Monthly Revenue YoY Comparison ─────────────────────
print("Chart 10: YoY comparison...")
df = q("""
    SELECT YEAR(i.invoice_date)  AS yr,
           MONTH(i.invoice_date) AS mo,
           ROUND(SUM(ii.quantity*ii.unit_price),2) AS revenue
    FROM invoices i JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
    GROUP BY yr, mo ORDER BY yr, mo
""")
pivot = df.pivot(index="mo", columns="yr", values="revenue").fillna(0)
fig, ax = plt.subplots(figsize=(10, 4))
for yr in pivot.columns:
    ax.plot(pivot.index, pivot[yr], marker="o", label=str(yr), linewidth=2)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
ax.set_title("Monthly Revenue — Year-over-Year", fontsize=13, fontweight="bold")
ax.legend(title="Year"); ax.set_ylabel("Revenue")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/10_yoy_comparison.png"); plt.close()


print(f"\nStage 2 complete. 10 charts saved to {CHART_DIR}/")