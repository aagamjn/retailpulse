"""
RetailPulse — Stage 4b: Sales Forecasting
Prophet baseline + XGBoost on lag features
Run: python model_forecast.py
Outputs:
  models/forecast_prophet.pkl
  models/forecast_xgb.pkl
  outputs/charts/forecast_*.png
"""

import os, joblib, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

os.makedirs("models",         exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)

# ── Load data ────────────────────────────────────────────────
ts = pd.read_csv("data/timeseries_weekly.csv", parse_dates=["week_start"])
ts = ts.dropna().reset_index(drop=True)      # drop rows with NaN lags
print(f"Weeks available after dropping NaN lags: {len(ts)}")

FORECAST_HORIZON = 8    # predict 8 weeks ahead

# ── Train / test split (last 8 weeks = test) ─────────────────
train = ts.iloc[:-FORECAST_HORIZON].copy()
test  = ts.iloc[-FORECAST_HORIZON:].copy()
print(f"Train: {len(train)} weeks  |  Test: {len(test)} weeks")


# ════════════════════════════════════════════════════════════
# MODEL A — Prophet
# ════════════════════════════════════════════════════════════
print("\n--- Prophet ---")
from prophet import Prophet

prophet_df = train[["week_start", "revenue"]].rename(
    columns={"week_start": "ds", "revenue": "y"}
)
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.1
)
m.fit(prophet_df)

future   = m.make_future_dataframe(periods=FORECAST_HORIZON, freq="W")
forecast = m.predict(future)

prophet_preds = forecast.tail(FORECAST_HORIZON)["yhat"].values
prophet_preds = np.clip(prophet_preds, 0, None)   # no negative revenue

prophet_mape = mean_absolute_percentage_error(test["revenue"], prophet_preds) * 100
prophet_rmse = np.sqrt(mean_squared_error(test["revenue"], prophet_preds))
print(f"Prophet MAPE : {prophet_mape:.2f}%")
print(f"Prophet RMSE : £{prophet_rmse:,.0f}")

joblib.dump(m, "models/forecast_prophet.pkl")


# ════════════════════════════════════════════════════════════
# MODEL B — XGBoost on lag features
# ════════════════════════════════════════════════════════════
print("\n--- XGBoost Forecaster ---")

LAG_FEATURES = [
    "revenue_lag_1w", "revenue_lag_2w",
    "revenue_lag_4w", "revenue_lag_8w",
    "revenue_roll4",  "revenue_roll8",
    "week_num",       "year"
]

X_train = train[LAG_FEATURES]
y_train = train["revenue"]
X_test  = test[LAG_FEATURES]
y_test  = test["revenue"]

xgb = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

xgb_mape = mean_absolute_percentage_error(y_test, xgb_preds) * 100
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
print(f"XGBoost MAPE : {xgb_mape:.2f}%")
print(f"XGBoost RMSE : £{xgb_rmse:,.0f}")

joblib.dump(xgb, "models/forecast_xgb.pkl")


# ══ CHARTS ══════════════════════════════════════════════════

# Chart A: Prophet full forecast
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train["week_start"], train["revenue"],
        color="#6366F1", linewidth=1.5, label="Historical")
ax.plot(test["week_start"],  test["revenue"],
        color="#10B981", linewidth=2,   label="Actual (test)")
ax.plot(test["week_start"],  prophet_preds,
        color="#F59E0B", linewidth=2, linestyle="--", label=f"Prophet (MAPE {prophet_mape:.1f}%)")
ax.plot(test["week_start"],  xgb_preds,
        color="#EF4444", linewidth=2, linestyle="--", label=f"XGBoost (MAPE {xgb_mape:.1f}%)")
ax.axvline(test["week_start"].iloc[0], color="gray",
           linestyle=":", linewidth=1.5, label="Train/Test split")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
ax.set_title("8-Week Sales Forecast — Prophet vs XGBoost", fontsize=13, fontweight="bold")
ax.set_xlabel("Week"); ax.set_ylabel("Weekly Revenue")
ax.legend(); plt.tight_layout()
plt.savefig("outputs/charts/forecast_comparison.png"); plt.close()

# Chart B: Model comparison bar
fig, ax = plt.subplots(figsize=(6, 4))
models = ["Prophet", "XGBoost"]
mapes  = [prophet_mape, xgb_mape]
colors = ["#F59E0B", "#EF4444"]
bars = ax.bar(models, mapes, color=colors, width=0.4)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11)
ax.set_ylabel("MAPE (lower is better)")
ax.set_title("Forecast Model Comparison", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(mapes) * 1.3)
plt.tight_layout()
plt.savefig("outputs/charts/forecast_mape.png"); plt.close()

# Chart C: XGBoost feature importance
fig, ax = plt.subplots(figsize=(7, 4))
feat_imp = pd.Series(xgb.feature_importances_, index=LAG_FEATURES).sort_values()
feat_imp.plot(kind="barh", ax=ax, color="#6366F1")
ax.set_title("XGBoost Feature Importance (Forecasting)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/charts/forecast_feature_importance.png"); plt.close()

print("\nStage 4b complete — Forecasting models done.")
print(f"  Prophet  MAPE: {prophet_mape:.2f}%  |  RMSE: £{prophet_rmse:,.0f}")
print(f"  XGBoost  MAPE: {xgb_mape:.2f}%  |  RMSE: £{xgb_rmse:,.0f}")
winner = "Prophet" if prophet_mape < xgb_mape else "XGBoost"
print(f"  Winner: {winner}")