"""Single source of truth for the training and inference feature schema."""
import numpy as np
import pandas as pd
from .distance_features import distance_features
from .bearing_features import bearing_features
from .connectivity_features import connectivity_features

FEATURE_COLUMNS = [
    "distance_m", "fraction_along", "segment_length_m", "distance_to_start_m",
    "distance_to_end_m", "log1p_distance_m", "heading_difference_deg", "heading_cosine",
    "heading_valid", "dt_s", "step_distance_m", "speed_mps", "gap_before", "speed_outlier",
    "candidate_rank", "candidate_count", "is_fallback", "start_in_degree", "end_out_degree",
    "previous_nearest_available", "same_as_previous_nearest", "connected_from_previous_nearest",
    "same_weak_component_as_previous",
]
IDENTIFIER_COLUMNS = ["partition", "trip_id", "observation_id", "candidate_arc_id"]
SCHEMA_VERSION = "member1-v1"


def extract_features(track, candidates, network, trip_id, partition):
    """No route/label argument is accepted. IDs are metadata, never model inputs."""
    obs = track.set_index("observation_id").loc[candidates.observation_id].reset_index()
    frame = pd.DataFrame({"partition": partition, "trip_id": str(trip_id),
                          "observation_id": candidates.observation_id.to_numpy(),
                          "candidate_arc_id": candidates.candidate_arc_id.to_numpy()})
    values = distance_features(candidates, network)
    values.update(bearing_features(obs, candidates, network))
    for name in ["dt_s", "step_distance_m", "speed_mps", "gap_before", "speed_outlier"]:
        values[name] = obs[name].to_numpy()
    for name in ["candidate_rank", "is_fallback"]:
        values[name] = candidates[name].to_numpy()
    values["candidate_count"] = candidates.groupby("observation_id").observation_id.transform("size").to_numpy()
    values.update(connectivity_features(obs, candidates, network, track))
    for name in FEATURE_COLUMNS:
        frame[name] = np.asarray(values[name], dtype=np.float32)
    if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Nonfinite inference feature")
    return frame[IDENTIFIER_COLUMNS + FEATURE_COLUMNS]
