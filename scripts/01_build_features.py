"""Build candidate-level feature tables for the training and student-test partitions.

Output: cache/<partition>.pkl  (one row per GPS observation x candidate road arc)
Also writes results/data/dataset_inventory.json (Section 2.2 of the report).
"""
import json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.data import load_trip, trip_ids
from pathsnap.candidates import build_candidate_table, FEATURES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = "/home/claude/work/student_folder/Student folder"
CACHE = os.path.join(ROOT, "cache")
OUT = os.path.join(ROOT, "results", "data")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

inventory = {}
for partition in ["train", "student_test"]:
    frames, stats = [], []
    t0 = time.time()
    for tid in trip_ids(DATA, partition):
        trip = load_trip(DATA, partition, tid)
        cand = build_candidate_table(trip, K=10, kq=60)
        frames.append(cand)
        pts = cand.groupby("point")["label"].max()
        stats.append({
            "trip_id": tid,
            "points": int(len(trip.xy)),
            "arcs": int(len(trip.arc_from)),
            "nodes": int(len(trip.node_xy)),
            "route_arcs": int(len(trip.route)),
            "candidates": int(len(cand)),
            "candidate_miss_points": int((pts == 0).sum()),
            "dt_eq_1s_pct": float(100 * np.mean(np.diff(trip.t) == 1)),
            "gaps_gt_30s": int(np.sum(np.diff(trip.t) > 30)),
            "max_gap_s": float(np.diff(trip.t).max()),
            "lon_min": float(trip.lonlat[:, 0].min()), "lon_max": float(trip.lonlat[:, 0].max()),
            "lat_min": float(trip.lonlat[:, 1].min()), "lat_max": float(trip.lonlat[:, 1].max()),
        })
    df = pd.concat(frames, ignore_index=True)
    df.to_pickle(os.path.join(CACHE, f"{partition}.pkl"))
    st = pd.DataFrame(stats)
    st.to_csv(os.path.join(OUT, f"trip_stats_{partition}.csv"), index=False)
    inventory[partition] = {
        "trips": int(len(st)),
        "observations": int(st["points"].sum()),
        "candidate_rows": int(len(df)),
        "arcs_total_per_trip_median": float(st["arcs"].median()),
        "nodes_per_trip_median": float(st["nodes"].median()),
        "route_arcs_median": float(st["route_arcs"].median()),
        "candidate_miss_rate_pct": float(100 * st["candidate_miss_points"].sum() / st["points"].sum()),
        "dt_eq_1s_pct": float(np.average(st["dt_eq_1s_pct"], weights=st["points"])),
        "gaps_gt_30s": int(st["gaps_gt_30s"].sum()),
        "max_gap_s": float(st["max_gap_s"].max()),
        "build_seconds": round(time.time() - t0, 1),
    }
    print(partition, inventory[partition], flush=True)

with open(os.path.join(OUT, "dataset_inventory.json"), "w") as f:
    json.dump(inventory, f, indent=2)
print("features:", FEATURES)
