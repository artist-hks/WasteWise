import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json

DARK = dict(
    paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
    font=dict(color="#E2E8F0", family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40)
)
GRID = dict(gridcolor="#1E293B", linecolor="#334155", zerolinecolor="#334155")

def fill_donut(zone_summary):
    labels = ["Empty (<25%)", "Low (25-50%)", "Medium (50-75%)", "Full (75-95%)", "Overflow (>95%)"]
    vals   = [zone_summary.get(k, 0) for k in ["empty","low","medium","full","overflow"]]
    colors = ["#6EE7B7","#93C5FD","#FCD34D","#FCA5A5","#DC2626"]
    fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.55,
                           marker=dict(colors=colors, line=dict(color="#0F172A", width=2))))
    fig.update_layout(**DARK, title="Bin Fill Distribution")
    return fig

def heatmap_24h(df):
    zones = df["zone"].unique()
    hours = list(range(24))
    z_data = []
    for zone in zones:
        row = []
        for h in hours:
            v = df[(df["zone"]==zone) & (df["hour"]==h)]["fill_percent"].mean()
            row.append(round(v, 1) if not pd.isna(v) else 0)
        z_data.append(row)
    fig = go.Figure(go.Heatmap(
        z=z_data, x=[f"{h}:00" for h in hours], y=list(zones),
        colorscale="RdYlGn_r", zmin=0, zmax=100,
        colorbar=dict(title="Fill %", tickfont=dict(color="#E2E8F0"))
    ))
    fig.update_layout(**DARK, title="24-Hour Fill Level Heatmap", xaxis=GRID, yaxis=GRID)
    return fig

def historical_trend(df):
    daily = df.groupby(["date","zone"])["fill_percent"].mean().reset_index()
    fig = go.Figure()
    colors_map = {"Malviya Nagar":"#10B981","Mansarovar":"#3B82F6",
                  "Vaishali Nagar":"#8B5CF6","Civil Lines":"#F59E0B","Sodala":"#EF4444"}
    for zone in daily["zone"].unique():
        zd = daily[daily["zone"]==zone]
        fig.add_trace(go.Scatter(x=zd["date"], y=zd["fill_percent"],
                                  mode="lines", name=zone,
                                  line=dict(color=colors_map.get(zone,"#888"), width=2)))
    fig.add_vline(x="2024-10-15", line_dash="dash", line_color="#EF4444",
                  annotation_text="Diwali", annotation_font_color="#EF4444")
    fig.update_layout(**DARK, title="30-Day Historical Waste Trend",
                      xaxis=dict(**GRID, title="Date"), yaxis=dict(**GRID, title="Avg Fill %"))
    return fig

def forecast_chart(forecast_records, zone_name):
    dates  = [r["ds"] for r in forecast_records]
    yhat   = [r["yhat"] for r in forecast_records]
    y_low  = [r["yhat_lower"] for r in forecast_records]
    y_high = [r["yhat_upper"] for r in forecast_records]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates+dates[::-1], y=y_high+y_low[::-1],
                              fill="toself", fillcolor="rgba(59,130,246,0.15)",
                              line=dict(color="rgba(0,0,0,0)"), name="Confidence"))
    fig.add_trace(go.Scatter(x=dates, y=yhat, mode="lines+markers",
                              line=dict(color="#3B82F6", width=2), name="Forecast"))
    fig.update_layout(**DARK, title=f"7-Day Forecast — {zone_name}",
                      xaxis=dict(**GRID), yaxis=dict(**GRID, title="Avg Fill %"))
    return fig

def training_history_chart(history):
    epochs = list(range(1, len(history.get("class_out_accuracy", []))+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history.get("class_out_accuracy",[]),
                              name="Train Acc", line=dict(color="#10B981")))
    fig.add_trace(go.Scatter(x=epochs, y=history.get("val_class_out_accuracy",[]),
                              name="Val Acc", line=dict(color="#10B981", dash="dash")))
    fig.add_trace(go.Scatter(x=epochs, y=history.get("class_out_loss",[]),
                              name="Train Loss", line=dict(color="#EF4444"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=epochs, y=history.get("val_class_out_loss",[]),
                              name="Val Loss", line=dict(color="#EF4444", dash="dash"), yaxis="y2"))
    fig.update_layout(**DARK, title="CNN Training History",
                      yaxis=dict(**GRID, title="Accuracy"),
                      yaxis2=dict(title="Loss", overlaying="y", side="right",
                                  gridcolor="#1E293B", color="#E2E8F0"))
    return fig
