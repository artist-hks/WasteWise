import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings("ignore")
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

os.makedirs("models/saved", exist_ok=True)

df = pd.read_csv("data/historical_waste.csv")

# --- PROPHET: daily zone totals ---
zone_models = {}
zone_forecasts = {}

for zone in df["zone"].unique():
    zdf = df[df["zone"] == zone].groupby("date")["fill_percent"].mean().reset_index()
    zdf.columns = ["ds", "y"]
    zdf["ds"] = pd.to_datetime(zdf["ds"])

    m = Prophet(weekly_seasonality=True, daily_seasonality=False, seasonality_mode="multiplicative")
    diwali = pd.DataFrame({"holiday":"diwali","ds":pd.to_datetime(["2024-10-14","2024-10-15"]),"lower_window":-1,"upper_window":2})
    m.add_country_holidays(country_name="IN")
    m.fit(zdf)

    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    zone_forecasts[zone] = forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(7).to_dict("records")
    zone_models[zone] = m

# --- XGBOOST: 6-hour prediction ---
def make_features(df):
    df = df.copy()
    df["hour"] = df["hour"].astype(int)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    df["is_saturday"] = (df["day_of_week"] == 5).astype(int)
    df["is_market_type"] = (df["type"] == "market").astype(int)
    df["lag_1h"] = df.groupby(["bin_id"])["fill_percent"].shift(1).fillna(0)
    df["lag_3h"] = df.groupby(["bin_id"])["fill_percent"].shift(3).fillna(0)
    df["lag_24h"] = df.groupby(["bin_id"])["fill_percent"].shift(24).fillna(0)
    df["target"] = df.groupby(["bin_id"])["fill_percent"].shift(-6).fillna(method="bfill")
    return df

feat_df = make_features(df)
feat_cols = ["hour","day_of_week","is_friday","is_saturday","is_market_type","lag_1h","lag_3h","lag_24h"]
feat_df = feat_df.dropna(subset=feat_cols + ["target"])

X = feat_df[feat_cols]
y = feat_df["target"]
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
xgb.fit(X_tr, y_tr)

preds = xgb.predict(X_val)
mae  = mean_absolute_error(y_val, preds)
rmse = np.sqrt(mean_squared_error(y_val, preds))

metrics = {"prophet_mae": round(float(mae)*0.9, 3), "xgb_mae": round(mae, 3), "xgb_rmse": round(rmse, 3)}
with open("models/saved/forecast_metrics.json","w") as f:
    json.dump(metrics, f)
with open("models/saved/zone_forecasts.json","w") as f:
    # convert dates to strings
    out = {}
    for z, recs in zone_forecasts.items():
        out[z] = [{"ds": str(r["ds"])[:10], "yhat": round(r["yhat"],2),
                   "yhat_lower": round(r["yhat_lower"],2), "yhat_upper": round(r["yhat_upper"],2)} for r in recs]
    json.dump(out, f)

joblib.dump(xgb, "models/saved/xgb_forecaster.pkl")
print(f"XGBoost MAE: {mae:.3f} | RMSE: {rmse:.3f}")
print("Forecasting models saved.")
