import streamlit as st
import pandas as pd
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.chart_utils import fill_donut, heatmap_24h, historical_trend

st.set_page_config(page_title="Live Dashboard", layout="wide")
st.markdown("""<style>
.stApp{background:#0F172A;color:#E2E8F0}
.stSidebar{background:#1E293B}
div[data-testid="metric-container"]{background:#1E293B;border:1px solid #334155;border-radius:12px;padding:16px}
</style>""", unsafe_allow_html=True)

st.title("📊 Live Dashboard")

@st.cache_data
def load_all():
    df = pd.read_csv("data/historical_waste.csv")
    with open("data/bins_metadata.json") as f:
        bins = json.load(f)
    return df, bins

try:
    df, bins = load_all()
except:
    st.error("Data not found. Run: python data/simulate_data.py")
    st.stop()

bins_df = pd.DataFrame(bins)
today = df[df["date"] == df["date"].max()]
latest = today.groupby("bin_id")["fill_percent"].last().reset_index()
bins_df = bins_df.merge(latest, on="bin_id", how="left")
bins_df["fill_percent"] = bins_df["fill_percent"].fillna(50)

# Zone cards
st.subheader("Zone Fill Levels")
zones_data = bins_df.groupby("zone")["fill_percent"].agg(["mean","count"]).reset_index()
cols = st.columns(5)
zone_colors = {"Malviya Nagar":"#10B981","Mansarovar":"#3B82F6","Vaishali Nagar":"#8B5CF6",
               "Civil Lines":"#F59E0B","Sodala":"#EF4444"}
for i, (_, row) in enumerate(zones_data.iterrows()):
    fp = row["mean"]
    badge_color = "#10B981" if fp < 50 else "#F59E0B" if fp < 75 else "#EF4444"
    with cols[i]:
        st.markdown(f"""
        <div style='background:#1E293B;border:1px solid #334155;border-radius:10px;padding:14px;text-align:center'>
          <div style='font-size:12px;color:#94A3B8'>{row['zone']}</div>
          <div style='font-size:2rem;font-weight:700;color:{badge_color}'>{fp:.0f}%</div>
          <div style='font-size:11px;color:#64748B'>{int(row['count'])} bins</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# Charts row 1
c1, c2 = st.columns(2)
with c1:
    st.subheader("Bin Fill Distribution")
    cats = {"empty":0,"low":0,"medium":0,"full":0,"overflow":0}
    for fp in bins_df["fill_percent"]:
        if fp < 25:   cats["empty"] += 1
        elif fp < 50: cats["low"] += 1
        elif fp < 75: cats["medium"] += 1
        elif fp < 95: cats["full"] += 1
        else:         cats["overflow"] += 1
    st.plotly_chart(fill_donut(cats), use_container_width=True)

with c2:
    st.subheader("24-Hour Fill Heatmap")
    today_full = df[df["date"] == df["date"].max()]
    st.plotly_chart(heatmap_24h(today_full), use_container_width=True)

# Historical trend
st.subheader("30-Day Historical Trend")
st.plotly_chart(historical_trend(df), use_container_width=True)

# Top bins table
st.subheader("⚠️ Bins Needing Collection (>70% Full)")
urgent = bins_df[bins_df["fill_percent"] >= 70].sort_values("fill_percent", ascending=False)
if not urgent.empty:
    disp = urgent[["bin_id","zone","fill_percent","type","capacity_liters"]].rename(columns={
        "bin_id":"Bin ID","zone":"Zone","fill_percent":"Fill %","type":"Type","capacity_liters":"Capacity (L)"
    })
    disp["Fill %"] = disp["Fill %"].round(1)
    st.dataframe(disp.style.background_gradient(subset=["Fill %"], cmap="RdYlGn_r"), use_container_width=True)
else:
    st.success("All bins under 70% — no urgent collections needed.")
