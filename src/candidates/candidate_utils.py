"""Shared segment geometry and directed-network metadata."""
from dataclasses import dataclass
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.preprocessing.coordinate_utils import LocalProjection, bearing_degrees


@dataclass
class RoadNetwork:
    nodes_xy: np.ndarray
    endpoints: np.ndarray
    geometries: np.ndarray
    lengths_m: np.ndarray
    bearings_deg: np.ndarray
    in_degree: np.ndarray
    out_degree: np.ndarray
    weak_component: np.ndarray
    projection: LocalProjection

    @classmethod
    def from_tables(cls, nodes, arcs, projection=None):
        projection = projection or LocalProjection.from_coordinates(nodes.longitude, nodes.latitude)
        xy = projection.project(nodes.longitude.to_numpy(), nodes.latitude.to_numpy())
        endpoints = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
        if not len(endpoints) or endpoints.min() < 0 or endpoints.max() >= len(xy):
            raise ValueError("Invalid road-network endpoints")
        coords = xy[endpoints]
        delta = coords[:, 1] - coords[:, 0]
        graph = coo_matrix((np.ones(len(endpoints)), (endpoints[:, 0], endpoints[:, 1])),
                           shape=(len(xy), len(xy))).tocsr()
        _, component = connected_components(graph, directed=True, connection="weak")
        return cls(xy, endpoints, shapely.linestrings(coords), np.linalg.norm(delta, axis=1),
                   bearing_degrees(delta[:, 0], delta[:, 1]),
                   np.bincount(endpoints[:, 1], minlength=len(xy)),
                   np.bincount(endpoints[:, 0], minlength=len(xy)), component, projection)


def project_to_segments(points_xy, starts_xy, ends_xy):
    """Vectorized exact point-to-segment projection, including zero-length arcs."""
    vector = ends_xy - starts_xy
    denominator = np.sum(vector * vector, axis=-1)
    fraction = np.divide(np.sum((points_xy - starts_xy) * vector, axis=-1), denominator,
                         out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    fraction = np.clip(fraction, 0.0, 1.0)
    snapped = starts_xy + fraction[..., None] * vector
    return np.linalg.norm(points_xy - snapped, axis=-1), fraction, snapped
