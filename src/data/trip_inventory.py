"""Full-dataset quality and topology inventory; no training/model selection."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .dataset_loader import load_trip, list_trips, DEVELOPMENT_PARTITIONS


def inventory_trip(trip):
    track, nodes, arcs = trip.track, trip.nodes, trip.arcs
    endpoints = arcs.to_numpy(dtype=int)
    graph = coo_matrix((np.ones(len(arcs)), endpoints.T), shape=(len(nodes), len(nodes))).tocsr()
    weak, component = connected_components(graph, directed=True, connection="weak")
    strong, _ = connected_components(graph, directed=True, connection="strong")
    used_nodes = np.unique(endpoints)
    dt = track.timestamp_s.diff().dropna().to_numpy()
    route = trip.route.arc_id.to_numpy(dtype=int) if trip.route is not None else np.array([], dtype=int)
    route_breaks = int(np.sum(endpoints[route[:-1], 1] != endpoints[route[1:], 0])) if len(route) > 1 else 0
    return {"partition": trip.partition, "trip_id": trip.trip_id,
            "gps_observations": len(track), "nodes": len(nodes), "arcs": len(arcs),
            "route_arcs": len(route), "missing_values": int(track.isna().sum().sum()),
            "invalid_coordinates": int((~track.longitude.between(-180, 180) |
                                        ~track.latitude.between(-90, 90)).sum()),
            "duplicate_timestamps": int(np.sum(dt == 0)), "backward_timestamps": int(np.sum(dt < 0)),
            "one_second_steps": int(np.sum(np.isclose(dt, 1))), "time_steps": len(dt),
            "gaps_over_10s": int(np.sum(dt > 10)), "max_gap_s": float(dt.max()) if len(dt) else 0,
            "min_longitude": float(track.longitude.min()), "max_longitude": float(track.longitude.max()),
            "min_latitude": float(track.latitude.min()), "max_latitude": float(track.latitude.max()),
            "duration_s": float(track.timestamp_s.iloc[-1] - track.timestamp_s.iloc[0]),
            "weak_components": int(weak), "strong_components": int(strong),
            "largest_weak_component_nodes": int(np.bincount(component).max()),
            "isolated_nodes": int(len(nodes) - len(used_nodes)),
            "self_loops": int(np.sum(endpoints[:, 0] == endpoints[:, 1])),
            "duplicate_arcs": int(arcs.duplicated().sum()), "reference_route_breaks": route_breaks}


def build_inventory(raw_root):
    records = []
    seen = set()
    for partition in DEVELOPMENT_PARTITIONS:
        for trip_id in list_trips(raw_root, partition):
            if trip_id in seen:
                raise ValueError(f"Trip appears in multiple partitions: {trip_id}")
            seen.add(trip_id)
            records.append(inventory_trip(load_trip(raw_root, partition, trip_id, include_reference=True)))
    return pd.DataFrame(records)
