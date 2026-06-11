import folium
import json

TRUCK_COLORS = {"truck_1": "#3B82F6", "truck_2": "#F59E0B", "truck_3": "#8B5CF6"}
DEPOT = [26.9124, 75.7873]

def get_bin_color(fill_pct):
    if fill_pct < 50:  return "#10B981"
    if fill_pct < 75:  return "#F59E0B"
    if fill_pct < 95:  return "#EF4444"
    return "#7F1D1D"

def create_base_map(center=DEPOT, zoom=12):
    return folium.Map(location=center, zoom_start=zoom,
                      tiles="CartoDB positron", control_scale=True)

def add_bin_markers(m, bins):
    for b in bins:
        fp = b.get("fill_percent", 0)
        color = get_bin_color(fp)
        popup_html = f"""
        <div style='font-family:sans-serif;font-size:13px;min-width:160px'>
          <b>{b['bin_id']}</b><br>
          Zone: {b['zone']}<br>
          Fill: <b style='color:{color}'>{fp:.1f}%</b><br>
          Type: {b.get('type','—')}<br>
          Capacity: {b.get('capacity_liters',300)}L
        </div>"""
        folium.CircleMarker(
            location=[b["lat"], b["lng"]],
            radius=7 if fp >= 70 else 5,
            color=color, fill=True, fill_color=color, fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{b['bin_id']} — {fp:.0f}%"
        ).add_to(m)
    return m

def add_truck_routes(m, routes):
    for truck_id, route in routes.items():
        coords = route.get("route_coords", [])
        if len(coords) < 2: continue
        color = TRUCK_COLORS.get(truck_id, "#888")
        folium.PolyLine(
            coords, color=color, weight=3, opacity=0.8,
            tooltip=f"{truck_id}: {len(route['bins'])} bins, {route['total_distance_km']} km"
        ).add_to(m)
    return m

def add_depot_marker(m):
    folium.Marker(
        DEPOT,
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
        popup="<b>Municipal Depot</b><br>3 trucks dispatched",
        tooltip="Depot"
    ).add_to(m)
    return m
