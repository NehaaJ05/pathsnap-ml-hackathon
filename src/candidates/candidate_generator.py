"""Exact radius search using an STRtree; never reads reference routes."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import shapely
from .candidate_utils import project_to_segments


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 50.0
    max_candidates: int = 8
    fallback_nearest: bool = True

    def __post_init__(self):
        if self.radius_m <= 0 or self.max_candidates < 1:
            raise ValueError("Positive radius and candidate limit required")

    def to_dict(self):
        return asdict(self)


class CandidateGenerator:
    def __init__(self, network, config=None):
        self.network = network
        self.config = config or CandidateConfig()
        self.valid_arc_ids = np.flatnonzero(network.lengths_m > 1e-8)
        if not len(self.valid_arc_ids):
            raise ValueError("Network has no nonzero-length road segments")
        self.tree = shapely.STRtree(network.geometries[self.valid_arc_ids])

    def generate(self, track):
        points_xy = track[["x_m", "y_m"]].to_numpy()
        points = shapely.points(points_xy)
        point_index, local_arc = self.tree.query(points, predicate="dwithin",
                                                  distance=self.config.radius_m)
        fallback = np.zeros(len(point_index), dtype=bool)
        missing = np.setdiff1d(np.arange(len(track)), np.unique(point_index))
        if len(missing) and self.config.fallback_nearest:
            # query_nearest(all_matches=True) makes equal-distance ties explicit.
            pair = self.tree.query_nearest(points[missing], all_matches=True)
            point_index = np.r_[point_index, missing[pair[0]]]
            local_arc = np.r_[local_arc, pair[1]]
            fallback = np.r_[fallback, np.ones(pair.shape[1], dtype=bool)]
        arc_id = self.valid_arc_ids[local_arc]
        nodes = self.network.nodes_xy[self.network.endpoints[arc_id]]
        distance, fraction, snapped = project_to_segments(points_xy[point_index], nodes[:, 0], nodes[:, 1])
        frame = pd.DataFrame({"observation_id": track.observation_id.to_numpy()[point_index],
                              "candidate_arc_id": arc_id, "distance_m": distance,
                              "fraction_along": fraction, "snapped_x_m": snapped[:, 0],
                              "snapped_y_m": snapped[:, 1], "is_fallback": fallback})
        frame = frame.sort_values(["observation_id", "distance_m", "candidate_arc_id"], kind="stable")
        frame["candidate_rank"] = frame.groupby("observation_id").cumcount() + 1
        return frame.loc[frame.candidate_rank <= self.config.max_candidates].reset_index(drop=True)
