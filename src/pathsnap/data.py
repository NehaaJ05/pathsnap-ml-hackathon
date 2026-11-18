"""Loading and projecting PathSnap trips.

Raw per-trip files (see the Dataset Exploration guide):
  .track  lon, lat, timestamp(s)          -- GPS observations
  .nodes  lon, lat (0-based row = node id) -- road-network nodes
  .arcs   from_node, to_node (row = arc id) -- road segments
  .route  arc ids in travel order          -- reference route (train/student_test only)

Coordinates are projected to metres with an azimuthal-equidistant projection
centred on each trip, so Euclidean distances are physical distances.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer


@dataclass
class Trip:
    trip_id: str
    partition: str
    lonlat: np.ndarray        # (n, 2) raw GPS, lon/lat
    t: np.ndarray             # (n,) seconds
    xy: np.ndarray            # (n, 2) GPS in metres
    node_xy: np.ndarray       # (m, 2) road nodes in metres
    arc_from: np.ndarray      # (k,) start node of each arc
    arc_to: np.ndarray        # (k,) end node of each arc
    route: np.ndarray | None  # (r,) reference arc ids, or None


def trip_ids(data_root: str, partition: str) -> list[str]:
    return sorted(os.listdir(os.path.join(data_root, partition)))


def _project(lonlat: np.ndarray, lon0: float, lat0: float, nodes_lonlat: np.ndarray):
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    gx, gy = tr.transform(lonlat[:, 0], lonlat[:, 1])
    nx, ny = tr.transform(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
    return np.column_stack([gx, gy]), np.column_stack([nx, ny])


def load_trip(data_root: str, partition: str, trip_id: str, with_route: bool = True) -> Trip:
    d = os.path.join(data_root, partition, trip_id)
    track = np.loadtxt(os.path.join(d, f"{trip_id}.track"), ndmin=2)
    nodes = np.loadtxt(os.path.join(d, f"{trip_id}.nodes"), ndmin=2)
    arcs = np.loadtxt(os.path.join(d, f"{trip_id}.arcs"), dtype=np.int64, ndmin=2)
    route = None
    rpath = os.path.join(d, f"{trip_id}.route")
    if with_route and os.path.exists(rpath):
        route = np.loadtxt(rpath, dtype=np.int64, ndmin=1)

    lonlat = track[:, :2]
    lon0, lat0 = float(lonlat[:, 0].mean()), float(lonlat[:, 1].mean())
    xy, node_xy = _project(lonlat, lon0, lat0, nodes[:, :2])
    return Trip(
        trip_id=trip_id,
        partition=partition,
        lonlat=lonlat,
        t=track[:, 2].astype(float),
        xy=xy,
        node_xy=node_xy,
        arc_from=arcs[:, 0],
        arc_to=arcs[:, 1],
        route=route,
    )
