"""Independent spatial check to distinguish reference mismatch from alignment cost."""
from pathlib import Path
import numpy as np
import pandas as pd
import shapely
from .dataset_loader import load_trip
from src.candidates.candidate_utils import RoadNetwork
from src.preprocessing.gps_cleaning import clean_track


def diagnose(root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    summary = pd.read_csv(root / "results/member1/processing_summary.csv", dtype={"trip_id": str})
    rows = []
    for row in summary.itertuples(index=False):
        trip = load_trip(root / "data/raw", row.partition, row.trip_id, include_reference=True)
        network = RoadNetwork.from_tables(trip.nodes, trip.arcs)
        track, _, _ = clean_track(trip.track, network.projection)
        labels = pd.read_csv(root / "data/processed" / row.partition / row.trip_id / "reference_labels.csv.gz")
        points = shapely.points(track[["x_m", "y_m"]].to_numpy())
        tree = shapely.STRtree(network.geometries[trip.route.arc_id.to_numpy()])
        pairs, distances = tree.query_nearest(points, all_matches=False, return_distance=True)
        nearest = np.empty(len(track))
        nearest[pairs[0]] = distances
        aligned = labels.reference_distance_m.to_numpy()
        rows.append({"partition": row.partition, "trip_id": row.trip_id,
                     "observations": len(track), "nearest_reference_over_50m": int((nearest > 50).sum()),
                     "aligned_reference_over_50m": int((aligned > 50).sum()),
                     "additional_far_after_alignment": int(((aligned > 50) & (nearest <= 50)).sum()),
                     "median_nearest_reference_distance_m": float(np.median(nearest)),
                     "median_aligned_reference_distance_m": float(np.median(aligned))})
    result = pd.DataFrame(rows)
    result.to_csv(root / "results/member1/reference_alignment_diagnostics.csv", index=False)
    print(result.groupby("partition")[["nearest_reference_over_50m", "aligned_reference_over_50m",
                                        "additional_far_after_alignment"]].sum().to_string(), flush=True)
    return result


if __name__ == "__main__":
    diagnose()
