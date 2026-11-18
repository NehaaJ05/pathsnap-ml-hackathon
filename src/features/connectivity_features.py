"""Cheap graph features; no reference routes and no decoder implementation."""
import numpy as np


def connectivity_features(observations, candidates, network, track):
    arc = candidates.candidate_arc_id.to_numpy(dtype=int)
    start, end = network.endpoints[arc].T
    # Previous GPS observation's geometrically nearest candidate, not its label.
    nearest = candidates.loc[candidates.candidate_rank == 1].set_index("observation_id").candidate_arc_id
    previous_ids = track.observation_id.shift(1).to_numpy()
    previous = nearest.reindex(previous_ids).fillna(-1).to_numpy(dtype=int)
    previous_map = dict(zip(track.observation_id, previous))
    prev_arc = np.array([previous_map[int(i)] for i in candidates.observation_id], dtype=np.int64)
    valid = (prev_arc >= 0) & ~observations.gap_before.to_numpy(dtype=bool)
    safe = np.maximum(prev_arc, 0)
    prev_end = network.endpoints[safe, 1]
    return {"start_in_degree": network.in_degree[start],
            "end_out_degree": network.out_degree[end],
            "previous_nearest_available": valid.astype(int),
            "same_as_previous_nearest": (valid & (prev_arc == arc)).astype(int),
            "connected_from_previous_nearest": (valid & (prev_end == start)).astype(int),
            "same_weak_component_as_previous":
                (valid & (network.weak_component[prev_end] == network.weak_component[start])).astype(int)}
