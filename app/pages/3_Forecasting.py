import streamlit as st
import pandas as pd
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.chart_utils import forecast_chart, historical_trend

st.set_page_config(page_title="Forecasting", layout="wide")
st.markdown("""<style>.stApp{background:#0F172A;color:#E2E8F0}.stSidebar{background:#1E293B}
div[data-testid="metric-container"]{background:#1E293B;border:1px solid #334155;border-radius:12px;padding:16px}</style>""", unsafe_allow_html=True)
st.title("📈 Waste Generation Forecasting")

@st.cache_data
def load_forecast():
    with open("models/saved/zone_forecasts.json") as f: return json.load(f)
@st.cache_data
def load_metrics():
    with open("models/saved/forecast_metrics.json") as f: return json.load(f)

try:
    forecasts = load_forecast()
    metrics   = load_metrics()
except:
    st.error("Run `python models/waste_forecaster.py` first.")
    st.stop()

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Prophet MAE",  f"{metrics.get('prophet_mae','—')}")
c2.metric("XGBoost MAE",  f"{metrics.get('xgb_mae','—')}")
c3.metric("XGBoost RMSE", f"{metrics.get('xgb_rmse','—')}")

st.markdown("---")
st.subheader("7-Day Zone Forecasts")

tabs = st.tabs(list(forecasts.keys()))
for tab, (zone, recs) in zip(tabs, forecasts.items()):
    with tab:
        fig = forecast_chart(recs, zone)
        st.plotly_chart(fig, use_container_width=True)
        forecast_df = pd.DataFrame(recs)
        forecast_df.columns = ["Date","Forecast (avg fill %)","Lower Bound","Upper Bound"]
        st.dataframe(forecast_df.round(2), use_container_width=True)

st.markdown("---")
st.subheader("Scenario Simulator")
col1, col2, col3 = st.columns(3)
with col1: temp = st.slider("Temperature (°C)", 20, 45, 32)
with col2: is_festival = st.checkbox("Festival Day 🎆", False)
with col3: is_market   = st.checkbox("Market Day 🛒", False)

if st.button("🔮 Run Scenario Prediction"):
    mult = 1.0
    if is_festival: mult *= 1.65
    if is_market:   mult *= 1.35
    if temp > 38:   mult *= 1.1
    st.success(f"Predicted waste multiplier: **{mult:.2f}x** baseline")
    st.info(f"High-waste zones to watch: {'All zones' if mult > 1.4 else 'Malviya Nagar, Civil Lines'}")
