import json
import math
import numpy as np
from itertools import permutations

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

DEPOT = {"lat": 26.9124, "lng": 75.7873, "bin_id": "DEPOT"}

def nearest_neighbor_route(bins_to_collect, truck_capacity=2000):
    """Greedy nearest-neighbor VRP for 3 trucks."""
    if not bins_to_collect:
        return {f"truck_{i+1}": {"bins":[], "total_distance_km":0, "route_coords":[], "load_liters":0} for i in range(3)}
    
    trucks = {f"truck_{i+1}": {"bins":[], "total_distance_km":0.0, "route_coords":[[DEPOT["lat"],DEPOT["lng"]]], "load_liters":0} for i in range(1,4)}
    unvisited = bins_to_collect.copy()
    truck_keys = list(trucks.keys())
    
    current_positions = {t: DEPOT for t in truck_keys}
    
    while unvisited:
        for tk in truck_keys:
            if not unvisited: break
            curr = current_positions[tk]
            nearest = min(unvisited, key=lambda b: haversine(curr["lat"], curr["lng"], b["lat"], b["lng"]))
            dist = haversine(curr["lat"], curr["lng"], nearest["lat"], nearest["lng"])
            
            est_load = nearest.get("capacity_liters", 300) * nearest.get("fill_percent", 80) / 100
            if trucks[tk]["load_liters"] + est_load > truck_capacity:
                # Return to depot and reset
                d_back = haversine(curr["lat"], curr["lng"], DEPOT["lat"], DEPOT["lng"])
                trucks[tk]["total_distance_km"] += d_back
                trucks[tk]["route_coords"].append([DEPOT["lat"], DEPOT["lng"]])
                trucks[tk]["load_liters"] = 0
                current_positions[tk] = DEPOT
            
            trucks[tk]["bins"].append(nearest["bin_id"])
            trucks[tk]["total_distance_km"] += dist
            trucks[tk]["route_coords"].append([nearest["lat"], nearest["lng"]])
            trucks[tk]["load_liters"] += est_load
            current_positions[tk] = nearest
            unvisited.remove(nearest)
    
    # Return all trucks to depot
    for tk in truck_keys:
        curr = current_positions[tk]
        d_back = haversine(curr["lat"], curr["lng"], DEPOT["lat"], DEPOT["lng"])
        trucks[tk]["total_distance_km"] += d_back
        trucks[tk]["route_coords"].append([DEPOT["lat"], DEPOT["lng"]])
        trucks[tk]["total_distance_km"] = round(trucks[tk]["total_distance_km"], 2)
    
    return trucks

def optimize_routes(bins_data, fill_threshold=70):
    """Main function: optimize collection routes."""
    bins_to_collect = [b for b in bins_data if b.get("fill_percent", 0) >= fill_threshold]
    all_bins = bins_data
    
    smart_routes = nearest_neighbor_route(bins_to_collect)
    fixed_routes  = nearest_neighbor_route(all_bins)
    
    smart_dist = sum(t["total_distance_km"] for t in smart_routes.values())
    fixed_dist  = sum(t["total_distance_km"] for t in fixed_routes.values())
    saving_pct  = round((1 - smart_dist / fixed_dist) * 100, 1) if fixed_dist > 0 else 0
    fuel_saved  = round((fixed_dist - smart_dist) * 0.08, 1)  # 0.08L per km truck avg
    
    return {
        "smart_routes": smart_routes,
        "fixed_routes": fixed_routes,
        "summary": {
            "bins_to_collect": len(bins_to_collect),
            "bins_skipped": len(all_bins) - len(bins_to_collect),
            "smart_distance_km": round(smart_dist, 2),
            "fixed_distance_km": round(fixed_dist, 2),
            "efficiency_saving_pct": saving_pct,
            "fuel_saved_liters": fuel_saved,
        }
    }
