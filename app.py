"""
RetailPulse — Stage 5: Streamlit Dashboard
Run: streamlit run app.py
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

# ── CONFIG ───────────────────────────────────────────────────


from dotenv import load_dotenv
import os
load_dotenv()

DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_NAME     = os.getenv("DB_NAME")

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load assets ──────────────────────────────────────────────
@st.cache_resource
def load_engine():
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    )

@st.cache_resource
def load_models():
    churn = joblib.load("models/churn_model.pkl")
    seg   = joblib.load("models/segmentation_model.pkl")
    return churn, seg

@st.cache_data
def load_data():
    engine = load_engine()
    rfm = pd.read_csv("data/rfm_with_clusters.csv")
    ts  = pd.read_csv("data/timeseries_weekly.csv", parse_dates=["week_start"])

    revenue_monthly = pd.read_sql("""
        SELECT DATE_FORMAT(i.invoice_date,'%%Y-%%m') AS month,
               ROUND(SUM(ii.quantity*ii.unit_price),2) AS revenue,
               COUNT(DISTINCT i.customer_id) AS customers
        FROM invoices i
        JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
        GROUP BY month ORDER BY month
    """, engine)

    top_countries = pd.read_sql("""
        SELECT c.country,
               ROUND(SUM(ii.quantity*ii.unit_price),2) AS revenue
        FROM invoices i
        JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
        JOIN customers c ON i.customer_id=c.customer_id
        WHERE c.country != 'United Kingdom'
        GROUP BY c.country ORDER BY revenue DESC LIMIT 10
    """, engine)

    return rfm, ts, revenue_monthly, top_countries

churn_bundle, seg_bundle = load_models()
rfm, ts, revenue_monthly, top_countries = load_data()

churn_model     = churn_bundle["model"]
churn_features  = churn_bundle["features"]
seg_model       = seg_bundle["model"]
seg_scaler      = seg_bundle["scaler"]
seg_name_map    = seg_bundle["name_map"]

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shop.png", width=60)
st.sidebar.title("RetailPulse")
st.sidebar.caption("Retail Analytics Platform")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔮 Churn Predictor", "👥 Segments", "📈 Forecast"]
)

SEGMENT_COLORS = {
    "Champions":       "#6366F1",
    "Loyal Customers": "#10B981",
    "At Risk":         "#F59E0B",
    "Lost":            "#EF4444",
}

# ════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📦 RetailPulse — Business Overview")
    st.caption("Online Retail II dataset · 2009–2011 · UK e-commerce")

    # KPI cards
    total_rev  = rfm["monetary"].sum()
    total_cust = len(rfm)
    churn_rate = rfm["churned"].mean() * 100
    avg_order  = rfm["avg_order_value"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue",    f"£{total_rev/1e6:.2f}M")
    k2.metric("Total Customers",  f"{total_cust:,}")
    k3.metric("Churn Rate",       f"{churn_rate:.1f}%")
    k4.metric("Avg Order Value",  f"£{avg_order:.2f}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Monthly Revenue Trend")
        fig = px.line(
            revenue_monthly, x="month", y="revenue",
            markers=True,
            color_discrete_sequence=["#6366F1"]
        )
        fig.update_layout(
            xaxis_title="Month", yaxis_title="Revenue (£)",
            yaxis_tickformat="£,.0f",
            margin=dict(l=0,r=0,t=10,b=0), height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top 10 Countries by Revenue")
        fig = px.bar(
            top_countries.sort_values("revenue"),
            x="revenue", y="country", orientation="h",
            color_discrete_sequence=["#10B981"]
        )
        fig.update_layout(
            xaxis_tickformat="£,.0f",
            margin=dict(l=0,r=0,t=10,b=0), height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Distribution")
    seg_counts = rfm["cluster_name"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    seg_counts["Color"] = seg_counts["Segment"].map(SEGMENT_COLORS)
    fig = px.pie(
        seg_counts, names="Segment", values="Customers",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        hole=0.45
    )
    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=320)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — Churn Predictor
# ════════════════════════════════════════════════════════════
elif page == "🔮 Churn Predictor":
    st.title("🔮 Customer Churn Predictor")
    st.caption("Enter customer behavior metrics to predict churn probability")

    st.info(
        "**Model:** XGBoost · **AUC-ROC:** 0.78 · **CV AUC:** 0.79 ± 0.02  \n"
        "Recency excluded to prevent data leakage — model predicts from behavioral signals only.",
        icon="ℹ️"
    )

    col1, col2 = st.columns(2)
    with col1:
        frequency         = st.slider("Purchase Frequency (orders)",        1, 100, 5)
        monetary          = st.number_input("Total Spend (£)",               10.0, 100000.0, 500.0, step=50.0)
        unique_products   = st.slider("Unique Products Purchased",           1, 200, 20)
        avg_unit_price    = st.number_input("Avg Unit Price (£)",            0.1, 500.0, 3.5, step=0.5)
    with col2:
        lifespan_days     = st.slider("Customer Lifespan (days)",            1, 730, 180)
        avg_order_value   = st.number_input("Avg Order Value (£)",           1.0, 5000.0, 150.0, step=10.0)
        purchase_rate     = st.number_input("Purchase Rate (orders/day)",    0.001, 1.0, 0.03, step=0.001, format="%.3f")
        revenue_per_prod  = st.number_input("Revenue per Product (£)",       0.1, 5000.0, 25.0, step=1.0)

    if st.button("Predict Churn Probability", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "frequency":           frequency,
            "monetary":            monetary,
            "unique_products":     unique_products,
            "avg_unit_price":      avg_unit_price,
            "lifespan_days":       lifespan_days,
            "avg_order_value":     avg_order_value,
            "purchase_rate":       purchase_rate,
            "revenue_per_product": revenue_per_prod,
        }])

        prob = churn_model.predict_proba(input_df)[0][1]
        pred = "Churned" if prob >= 0.5 else "Active"

        st.markdown("---")
        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            if pred == "Churned":
                st.error(f"⚠️ **Prediction: {pred}**")
                st.error(f"Churn Probability: **{prob*100:.1f}%**")
                st.markdown("**Recommended action:** Send a win-back campaign with a discount offer.")
            else:
                st.success(f"✅ **Prediction: {pred}**")
                st.success(f"Churn Probability: **{prob*100:.1f}%**")
                st.markdown("**Recommended action:** Upsell with a loyalty reward or premium product.")

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#EF4444" if prob >= 0.5 else "#10B981"},
                    "steps": [
                        {"range": [0,  40], "color": "#DCFCE7"},
                        {"range": [40, 60], "color": "#FEF9C3"},
                        {"range": [60,100], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75, "value": 50
                    }
                },
                title={"text": "Churn Risk"}
            ))
            fig.update_layout(height=260, margin=dict(l=20,r=20,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — Segments
# ════════════════════════════════════════════════════════════
elif page == "👥 Segments":
    st.title("👥 Customer Segmentation")
    st.caption("K-Means clustering on RFM space · K=4 (business-driven selection)")

    ACTIONS = {
        "Champions":       "Reward loyalty, request reviews, offer early access",
        "Loyal Customers": "Upsell premium, increase frequency with bundles",
        "At Risk":         "Win-back campaign — personalised discount within 2 weeks",
        "Lost":            "Low re-engagement ROI — minimal spend, survey only",
    }

    cols = st.columns(4)
    for i, (seg, action) in enumerate(ACTIONS.items()):
        count      = (rfm["cluster_name"] == seg).sum()
        churn_rate = rfm[rfm["cluster_name"] == seg]["churned"].mean() * 100
        with cols[i]:
            st.markdown(
                f"""<div style='background:{SEGMENT_COLORS[seg]}22;
                border:1.5px solid {SEGMENT_COLORS[seg]};
                border-radius:10px;padding:14px;text-align:center'>
                <b style='color:{SEGMENT_COLORS[seg]};font-size:15px'>{seg}</b><br>
                <span style='font-size:22px;font-weight:600'>{count:,}</span>
                <span style='font-size:12px;color:gray'> customers</span><br>
                <span style='font-size:13px'>Churn: {churn_rate:.0f}%</span><br>
                <span style='font-size:11px;color:gray'>{action}</span>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Recency vs Monetary by Segment")
        fig = px.scatter(
            rfm.sample(min(2000, len(rfm))),
            x="recency", y="monetary",
            color="cluster_name",
            color_discrete_map=SEGMENT_COLORS,
            opacity=0.5, log_y=True,
            labels={"recency": "Recency (days)", "monetary": "Monetary £ (log)"}
        )
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("RFM Profile by Segment")
        profile = rfm.groupby("cluster_name")[["recency","frequency","monetary"]].mean().round(1)
        profile = profile.reindex(["Champions","Loyal Customers","At Risk","Lost"])
        fig = px.bar(
            profile.reset_index().melt(id_vars="cluster_name"),
            x="cluster_name", y="value", color="variable",
            barmode="group",
            color_discrete_sequence=["#6366F1","#10B981","#F59E0B"],
            labels={"cluster_name": "Segment", "value": "Avg Value"}
        )
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — Forecast
# ════════════════════════════════════════════════════════════
elif page == "📈 Forecast":
    st.title("📈 Sales Forecast")
    st.caption("Prophet model · 8-week horizon · MAPE 14.74%")

    st.info(
        "**Winner:** Prophet (MAPE 14.74%) vs XGBoost (MAPE 15.98%)  \n"
        "Forecast trained on 88 weeks, evaluated on last 8 weeks of data.",
        icon="ℹ️"
    )

    st.subheader("Weekly Revenue — Historical + Forecast")
    train_ts = ts.iloc[:-8]
    test_ts  = ts.iloc[-8:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_ts["week_start"], y=train_ts["revenue"],
        name="Historical", line=dict(color="#6366F1", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=test_ts["week_start"], y=test_ts["revenue"],
        name="Actual (test)", line=dict(color="#10B981", width=2)
    ))
    '''fig.add_vline(
        x=str(test_ts["week_start"].iloc[0]),
        line_dash="dot", line_color="gray",
        annotation_text="Train/Test split"
    )'''
    split_date = test_ts["week_start"].iloc[0]
    fig.add_shape(
        type="line",
        x0=split_date, x1=split_date,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(dash="dot", color="gray", width=1.5)
    )
    fig.add_annotation(
        x=split_date, y=1,
        xref="x", yref="paper",
        text="Train/Test split",
        showarrow=False,
        yanchor="bottom",
        font=dict(size=11, color="gray")
    )
    fig.update_layout(
        yaxis_tickformat="£,.0f",
        xaxis_title="Week", yaxis_title="Weekly Revenue (£)",
        height=380, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Prophet MAPE",  "14.74%", delta="-1.24% vs XGBoost", delta_color="inverse")
    c2.metric("Prophet RMSE",  "£52,225")
    c3.metric("XGBoost MAPE",  "15.98%")

    st.subheader("Weekly Revenue Summary (Last 8 Weeks)")
    st.dataframe(
        test_ts[["week_start","revenue","orders","customers"]]
        .rename(columns={
            "week_start": "Week",
            "revenue":    "Revenue (£)",
            "orders":     "Orders",
            "customers":  "Unique Customers"
        })
        .set_index("Week")
        .style.format({"Revenue (£)": "£{:,.0f}"}),
        use_container_width=True
    )