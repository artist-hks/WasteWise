import streamlit as st
import pandas as pd
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.map_utils import create_base_map, add_bin_markers, add_truck_routes, add_depot_marker
from models.route_optimizer import optimize_routes
from streamlit_folium import st_folium

st.set_page_config(page_title="Route Map", layout="wide")
st.markdown("""<style>.stApp{background:#0F172A;color:#E2E8F0}.stSidebar{background:#1E293B}
div[data-testid="metric-container"]{background:#1E293B;border:1px solid #334155;border-radius:12px;padding:16px}</style>""", unsafe_allow_html=True)
st.title("🗺️ Route Optimization Map")

@st.cache_data
def load_data():
    df = pd.read_csv("data/historical_waste.csv")
    with open("data/bins_metadata.json") as f: bins = json.load(f)
    return df, bins

df, bins = load_data()
today = df[df["date"] == df["date"].max()]
latest = today.groupby("bin_id")["fill_percent"].last().reset_index()
bins_df = pd.DataFrame(bins).merge(latest, on="bin_id", how="left")
bins_df["fill_percent"] = bins_df["fill_percent"].fillna(50)
bins_list = bins_df.to_dict("records")

# Sidebar controls
with st.sidebar:
    st.markdown("### Map Controls")
    threshold = st.slider("Collection threshold (%)", 50, 95, 70)
    show_all  = st.checkbox("Show all bins", value=True)
    show_routes = st.checkbox("Show truck routes", value=True)
    route_mode = st.radio("Route type", ["Smart Route", "Fixed Route"])

result = optimize_routes(bins_list, fill_threshold=threshold)
summary = result["summary"]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Bins to Collect",   summary["bins_to_collect"])
c2.metric("Bins Skipped",      summary["bins_skipped"])
c3.metric("Distance Saved",    f"{summary['fixed_distance_km'] - summary['smart_distance_km']:.1f} km")
c4.metric("Fuel Saved",        f"{summary['fuel_saved_liters']} L")

st.markdown("---")

# Build map
m = create_base_map()
display_bins = bins_list if show_all else [b for b in bins_list if b.get("fill_percent",0) >= threshold]
m = add_bin_markers(m, display_bins)
if show_routes:
    routes = result["smart_routes"] if route_mode == "Smart Route" else result["fixed_routes"]
    m = add_truck_routes(m, routes)
m = add_depot_marker(m)

st_folium(m, width=None, height=520)

# Route details table
st.markdown("### Truck Route Summary")
routes_data = result["smart_routes"] if route_mode == "Smart Route" else result["fixed_routes"]
rows = []
truck_colors = {"truck_1":"🔵","truck_2":"🟠","truck_3":"🟣"}
for tid, rdata in routes_data.items():
    rows.append({
        "Truck": f"{truck_colors.get(tid,'')} {tid}",
        "Bins Collected": len(rdata["bins"]),
        "Distance (km)": rdata["total_distance_km"],
        "Est. Time (min)": round(rdata["total_distance_km"] / 0.5 + len(rdata["bins"]) * 3),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)
