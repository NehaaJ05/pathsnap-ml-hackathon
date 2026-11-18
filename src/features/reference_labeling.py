"""OFFLINE DEVELOPMENT ONLY: approximate observation labels from a route sequence.

The reference is an ordered list of arc occurrences, not timestamped GPS labels.
Monotone dynamic alignment respects that order, restarting after GPS gaps. Its
labels remain proxies, and distance/ambiguity gates explicitly mark uncertainty.
Nothing in this module is called by the inference feature extractor.
"""
import numpy as np
import pandas as pd
from src.candidates.candidate_utils import project_to_segments
from src.preprocessing.coordinate_utils import angular_difference


def align_reference(track, network, route_arc_ids, max_distance_m=50.0,
                    ambiguity_margin_m=3.0, heading_weight=0.1,
                    max_backpointer_mb=512):
    route = np.asarray(route_arc_ids, dtype=np.int64).reshape(-1)
    if not len(route) or route.min() < 0 or route.max() >= len(network.endpoints):
        raise ValueError("Invalid reference route arc IDs")
    endpoints = network.nodes_xy[network.endpoints[route]]
    rows = []
    for _, group in track.groupby("sequence_id", sort=False):
        n, r = len(group), len(route)
        if n * r * 4 > max_backpointer_mb * 1024**2:
            raise MemoryError("Reference alignment exceeds configured memory budget; split long sequences")
        back = np.empty((n, r), dtype=np.int32)
        previous = np.zeros(r)
        xy = group[["x_m", "y_m"]].to_numpy()
        headings = group.heading_deg.to_numpy()
        valid = group.heading_valid.to_numpy()
        # prefix minimum permits staying on the same occurrence or moving forward.
        for i in range(n):
            distance, _, _ = project_to_segments(xy[i], endpoints[:, 0], endpoints[:, 1])
            angle = angular_difference(headings[i], network.bearings_deg[route]) if valid[i] else 0.0
            emission = distance + heading_weight * angle
            if i:
                prefix_min = np.minimum.accumulate(previous)
                new_min = np.r_[True, previous[1:] < prefix_min[:-1]]
                indices = np.maximum.accumulate(np.where(new_min, np.arange(r), 0))
                back[i] = indices
                previous = prefix_min + emission
            else:
                back[i] = np.arange(r)
                previous = emission
        chosen = np.empty(n, dtype=int)
        chosen[-1] = int(np.argmin(previous))
        for i in range(n - 1, 0, -1):
            chosen[i - 1] = back[i, chosen[i]]
        del back
        for i, (_, obs) in enumerate(group.iterrows()):
            distance, _, _ = project_to_segments(xy[i], endpoints[:, 0], endpoints[:, 1])
            angle = angular_difference(headings[i], network.bearings_deg[route]) if valid[i] else np.zeros(r)
            cost = distance + heading_weight * angle
            j = chosen[i]
            # Repeated occurrences of the same arc are the same classification label.
            alternatives = cost[route != route[j]]
            margin = float(alternatives.min() - cost[j]) if len(alternatives) else np.inf
            usable = bool(distance[j] <= max_distance_m and margin >= ambiguity_margin_m
                          and not obs.speed_outlier)
            rows.append({"observation_id": int(obs.observation_id),
                         "reference_arc_id": int(route[j]), "route_position": int(j),
                         "reference_distance_m": float(distance[j]),
                         "local_cost_margin_m": margin, "label_usable": usable,
                         "label_status": "usable_proxy" if usable else
                         ("far_from_reference" if distance[j] > max_distance_m else
                          "speed_outlier" if obs.speed_outlier else "ambiguous_alignment")})
    return pd.DataFrame(rows).sort_values("observation_id").reset_index(drop=True)


def attach_candidate_coverage(labels, candidates):
    labels = labels.copy()
    pairs = set(zip(candidates.observation_id, candidates.candidate_arc_id))
    labels["candidate_hit"] = [(o, a) in pairs for o, a in
                               zip(labels.observation_id, labels.reference_arc_id)]
    labels["training_eligible"] = labels.label_usable & labels.candidate_hit
    return labels


def build_training_table(features, labels):
    """Explicit offline join; callers select FEATURE_COLUMNS, never all columns."""
    if len(features[["partition", "trip_id"]].drop_duplicates()) != 1:
        raise ValueError("Join labels within one partition/trip, then concatenate training tables")
    merged = features.merge(labels, on="observation_id", validate="many_to_one")
    merged["target"] = pd.array(np.where(merged.training_eligible,
                                        (merged.candidate_arc_id == merged.reference_arc_id).astype(int),
                                        np.nan), dtype="Int64")
    return merged
