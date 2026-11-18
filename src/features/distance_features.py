"""Distance and segment-position features in projected metres."""
import numpy as np


def distance_features(candidates, network):
    arc = candidates.candidate_arc_id.to_numpy(dtype=int)
    length = network.lengths_m[arc]
    return {
        "distance_m": candidates.distance_m.to_numpy(),
        "fraction_along": candidates.fraction_along.to_numpy(),
        "segment_length_m": length,
        "distance_to_start_m": candidates.fraction_along.to_numpy() * length,
        "distance_to_end_m": (1.0 - candidates.fraction_along.to_numpy()) * length,
        "log1p_distance_m": np.log1p(candidates.distance_m.to_numpy()),
    }
