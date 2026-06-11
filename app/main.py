import streamlit as st
import pandas as pd
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="WasteWise | Municipal Intelligence",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #0F172A; color: #E2E8F0; }
.stSidebar { background-color: #1E293B; }
.stMetric { background: #1E293B; border-radius: 12px; padding: 16px; border: 1px solid #334155; }
section[data-testid="stSidebar"] { background-color: #1E293B; }
div[data-testid="metric-container"] {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 12px; padding: 16px;
}
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { color: #F8FAFC; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ♻️ WasteWise")
    st.markdown("**Municipal Waste Intelligence**")
    st.markdown("---")
    st.markdown("**🏙️ City:** Jaipur, Rajasthan")
    st.markdown("---")
    st.markdown("**System Status**")
    st.success("🟢 Data Pipeline: Active")
    st.success("🟢 Forecaster: Ready")
    st.success("🟢 Route Engine: Ready")
    st.markdown("---")
    st.caption("WasteWise v1.0 | SMART Cities Mission")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/historical_waste.csv")
    with open("data/bins_metadata.json") as f:
        bins = json.load(f)
    return df, bins

try:
    df, bins = load_data()
    today_df = df[df["date"] == df["date"].max()]
    latest   = today_df.groupby("bin_id")["fill_percent"].last().reset_index()
    bins_df  = pd.DataFrame(bins).merge(latest, on="bin_id", how="left")
    bins_df["fill_percent"] = bins_df["fill_percent"].fillna(
        bins_df["fill_percent"].mean() if bins_df["fill_percent"].notna().any() else 50
    )
except Exception as e:
    st.warning(f"Run `python data/simulate_data.py` first. Error: {e}")
    bins_df = pd.DataFrame()
    df = pd.DataFrame()

# Hero
st.markdown("# ♻️ WasteWise")
st.markdown("#### Predictive Waste Intelligence for Smart Cities — Jaipur, Rajasthan")
st.markdown("---")

# KPI Cards
if not bins_df.empty:
    full_bins    = (bins_df["fill_percent"] >= 70).sum()
    overflow     = (bins_df["fill_percent"] >= 95).sum()
    trips_saved  = round((1 - full_bins / len(bins_df)) * 100, 1)
    fuel_saved   = round(full_bins * 0.47, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🗑️ Bins Monitored",   f"{len(bins_df)}")
    c2.metric("🚛 Trips Saved",       f"{trips_saved}%",     delta="vs fixed route")
    c3.metric("⚠️ Overflow Risk",     f"{overflow} bins",    delta_color="inverse")
    c4.metric("⛽ Fuel Saved Today",  f"{fuel_saved} L")

    st.markdown("---")
    st.markdown("### Zone Fill Summary")
    zones = bins_df.groupby("zone")["fill_percent"].mean().reset_index()
    for _, row in zones.iterrows():
        fp = row["fill_percent"]
        color = "#10B981" if fp < 50 else "#F59E0B" if fp < 75 else "#EF4444"
        col1, col2 = st.columns([3,1])
        col1.markdown(f"**{row['zone']}**")
        col2.markdown(f"<span style='color:{color};font-weight:700'>{fp:.1f}%</span>", unsafe_allow_html=True)
        st.progress(int(fp))
else:
    st.info("👋 Welcome! Run the setup scripts to load data. See README.md.")
