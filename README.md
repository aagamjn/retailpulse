# 📦 RetailPulse

> End-to-end retail analytics platform — churn prediction, sales forecasting, and customer segmentation on 1M+ real transactions.

🚀 **[Live Demo →](https://your-app.streamlit.app)** ← replace with your Streamlit Cloud URL after deployment

---

## Results

| Model | Metric | Score |
|---|---|---|
| XGBoost Churn Classifier | AUC-ROC | **0.78** |
| XGBoost Churn Classifier | 5-fold CV AUC | **0.79 ± 0.02** |
| Prophet Sales Forecast | MAPE (8-week) | **14.74%** |
| K-Means Segmentation | Silhouette Score | **0.36** |

**Business findings:**
- 51.1% of customers churned (no purchase in 90 days)
- Champions (22% of customers) generate 3× avg revenue vs Loyal tier
- Lost segment has 100% churn rate — confirmed independently by both supervised and unsupervised models
- Peak revenue hours: 10am–2pm; lowest: weekends

---

## Pipeline

```
Raw CSV (1M rows)
    │
    ▼
1. ETL + MySQL          → 5,878 customers · 36,969 invoices · 805K line items
    │
    ▼
2. SQL Queries + EDA    → 10 business queries · 10 charts
    │
    ▼
3. Feature Engineering  → RFM · churn label · lag features
    │
    ▼
4. Modeling
   ├── Churn Prediction  → XGBoost (AUC 0.78) + SHAP explainability
   ├── Sales Forecast    → Prophet (MAPE 14.74%) vs XGBoost (15.98%)
   └── Segmentation      → K-Means K=4 (Champions · Loyal · At Risk · Lost)
    │
    ▼
5. Streamlit App        → Live interactive dashboard
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Database | MySQL · SQLAlchemy |
| Data | Python · pandas · numpy |
| ML | XGBoost · scikit-learn · Prophet · SHAP |
| Visualisation | matplotlib · seaborn · plotly |
| App | Streamlit |

---

## Project Structure

```
retailpulse/
│
├── data/
│   ├── online_retail_II.csv          # raw dataset (not committed)
│   ├── rfm_features.csv              # engineered features
│   ├── rfm_with_clusters.csv         # features + segment labels
│   └── timeseries_weekly.csv         # weekly aggregates for forecasting
│
├── models/
│   ├── churn_model.pkl               # XGBoost churn classifier
│   ├── forecast_prophet.pkl          # Prophet forecasting model
│   ├── forecast_xgb.pkl              # XGBoost forecasting model
│   └── segmentation_model.pkl        # K-Means segmentation
│
├── outputs/
│   └── charts/                       # all generated charts (PNG)
│
├── etl_load.py                       # Stage 1: ingest CSV → MySQL
├── business_queries.sql              # Stage 2: 10 SQL business queries
├── eda.py                            # Stage 2: exploratory analysis + charts
├── feature_engineering.py            # Stage 3: RFM + churn label + lag features
├── fix_churn.py                      # Stage 4a: churn model (no leakage)
├── model_forecast.py                 # Stage 4b: sales forecasting
├── fix_segmentation.py               # Stage 4c: K=4 customer segmentation
├── app.py                            # Stage 5: Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## Quickstart

**1. Clone and install**
```bash
git clone https://github.com/aagamjn/retailpulse
cd retailpulse
pip install -r requirements.txt
```

**2. Set up MySQL**
- Create database `retailpulse` in MySQL Workbench
- Run `business_queries.sql` to create tables and views

**3. Add dataset**
- Download [Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) from Kaggle
- Place CSV at `data/online_retail_II.csv`

**4. Run the pipeline**
```bash
python etl_load.py              # load data into MySQL
python eda.py                   # generate EDA charts
python feature_engineering.py   # build features
python fix_churn.py             # train churn model
python model_forecast.py        # train forecast models
python fix_segmentation.py      # train segmentation
streamlit run app.py            # launch dashboard
```

---

## Key Design Decisions

**Why remove recency from churn features?**
Recency directly encodes the churn label (churn = recency ≥ 90 days). Including it gives AUC=1.0 — a textbook case of data leakage. The model uses only behavioral signals: frequency, monetary, lifespan, purchase rate, product variety.

**Why K=4 when silhouette peaked at K=2?**
K=2 is statistically optimal but business-useless. K=4 produces Champions, Loyal, At Risk, and Lost — four segments that map directly to distinct marketing strategies. The slight silhouette drop (0.42 → 0.36) is the cost of interpretability.

**Why Prophet over XGBoost for forecasting?**
Prophet won on MAPE (14.74% vs 15.98%) and handles yearly seasonality natively without manual lag engineering. XGBoost is included as a comparison baseline.

---

## Dataset

[Online Retail II — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
Chen, D., Sain, S.L., & Guo, K. (2012). Data mining for the online retail industry.
*Journal of Database Marketing and Customer Strategy Management*, 19(3), 197–208.

---

## Author

**Aagam Jain** · B.Tech ECE · SVNIT Surat  
[GitHub](https://github.com/aagamjn) · [LinkedIn](https://linkedin.com/in/yourprofile)