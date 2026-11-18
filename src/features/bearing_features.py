"""Directed angular compatibility. Missing heading has an explicit flag."""
import numpy as np
from src.preprocessing.coordinate_utils import angular_difference


def bearing_features(observations, candidates, network):
    arc = candidates.candidate_arc_id.to_numpy(dtype=int)
    valid = observations.heading_valid.to_numpy(dtype=bool)
    angle = angular_difference(observations.heading_deg.to_numpy(), network.bearings_deg[arc])
    return {"heading_difference_deg": np.where(valid, angle, 0.0),
            "heading_cosine": np.where(valid, np.cos(np.radians(angle)), 0.0),
            "heading_valid": valid.astype(int)}
