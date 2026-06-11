import pandas as pd
import numpy as np
import json, os, random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

ZONES = {
    "Malviya Nagar":  {"center": [26.8535, 75.8069], "num_bins": 22},
    "Mansarovar":     {"center": [26.8490, 75.7770], "num_bins": 20},
    "Vaishali Nagar": {"center": [26.9124, 75.7382], "num_bins": 18},
    "Civil Lines":    {"center": [26.9260, 75.8235], "num_bins": 20},
    "Sodala":         {"center": [26.9010, 75.7690], "num_bins": 20},
}

# Generate bin locations
bins = []
bin_id = 1
for zone_name, zone_info in ZONES.items():
    clat, clng = zone_info["center"]
    for _ in range(zone_info["num_bins"]):
        btype = random.choice(["residential","residential","commercial","market"])
        capacity = random.choice([200, 300, 500])
        bins.append({
            "bin_id": f"BIN_{bin_id:03d}",
            "zone": zone_name,
            "lat": clat + random.uniform(-0.012, 0.012),
            "lng": clng + random.uniform(-0.012, 0.012),
            "type": btype,
            "capacity_liters": capacity
        })
        bin_id += 1

os.makedirs("data", exist_ok=True)
with open("data/bins_metadata.json", "w") as f:
    json.dump(bins, f, indent=2)

# Generate 30-day hourly fill data
start_date = datetime(2024, 10, 1)
records = []

for day in range(30):
    current_date = start_date + timedelta(days=day)
    is_friday   = current_date.weekday() == 4
    is_saturday = current_date.weekday() == 5
    is_diwali   = (day == 14)  # Day 15 = festival spike

    day_mult = 1.0
    if is_friday:   day_mult = 1.35
    if is_saturday: day_mult = 1.45
    if is_diwali:   day_mult = 1.65

    for b in bins:
        fill = 5.0  # reset at 6 AM
        for hour in range(24):
            # Fill rate by type and hour
            if b["type"] == "residential":
                rate = 5 * (1 / (1 + np.exp(-0.3 * (hour - 14))))
            elif b["type"] == "commercial":
                rate = 7 * (1 / (1 + np.exp(-0.4 * (hour - 12))))
            else:
                rate = 10 * (1 / (1 + np.exp(-0.5 * (hour - 11))))

            fill += rate * day_mult * random.uniform(0.85, 1.15)
            fill = min(fill, 100.0)

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "hour": hour,
                "bin_id": b["bin_id"],
                "zone": b["zone"],
                "fill_percent": round(fill, 2),
                "type": b["type"]
            })

        # Reset at end of day (collection)
        fill = 5.0

df = pd.DataFrame(records)
df.to_csv("data/historical_waste.csv", index=False)
print(f"Generated {len(bins)} bins, {len(df)} hourly records.")
print("Files saved: data/bins_metadata.json, data/historical_waste.csv")
