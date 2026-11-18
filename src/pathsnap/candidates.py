"""Candidate road-segment generation, reference labels and per-candidate features.

For every GPS observation we keep the K road arcs nearest to it (point-to-segment
distance). Candidate generation never looks at labels, so it is identical for
training, student-test and evaluator-test trips.

Reference label: a GPS observation's reference arc is the arc of the reference route
that lies nearest to it. This is a documented modelling choice (the .route file gives
an ordered list of arcs, not one label per observation). A candidate is positive when
it equals that reference arc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import Trip

SAMPLE_FRACS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
FEATURES = [
    "d", "d_rel", "rank", "t_param", "dist_start", "dist_end", "arc_len",
    "cos_heading", "heading_deg", "speed", "dt_prev", "dt_next",
    "deg_from", "deg_to", "n_cands",
]


def point_segment(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Distance from points P to segments AB, plus the clipped projection parameter."""
    AB = B - A
    L2 = (AB ** 2).sum(1)
    AP = P - A
    safe = np.where(L2 > 0, L2, 1.0)
    t = np.where(L2 > 0, (AP * AB).sum(1) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    foot = A + t[:, None] * AB
    d = np.linalg.norm(P - foot, axis=1)
    return d, t


def generate_candidates(trip: Trip, K: int = 10, kq: int = 60) -> pd.DataFrame:
    A = trip.node_xy[trip.arc_from]
    B = trip.node_xy[trip.arc_to]
    k = A.shape[0]
    samples = (A[None] * (1 - SAMPLE_FRACS)[:, None, None]
               + B[None] * SAMPLE_FRACS[:, None, None]).reshape(-1, 2)
    arc_of = np.tile(np.arange(k), len(SAMPLE_FRACS))
    tree = cKDTree(samples)
    _, idx = tree.query(trip.xy, k=min(kq, len(arc_of)))
    idx = np.asarray(idx).reshape(len(trip.xy), -1)

    n = len(trip.xy)
    pt = np.repeat(np.arange(n), idx.shape[1])
    arc = arc_of[idx.ravel()]
    d, t = point_segment(trip.xy[pt], A[arc], B[arc])
    df = pd.DataFrame({"point": pt, "arc": arc, "d": d, "t_param": t})
    df = df.sort_values(["point", "d"]).drop_duplicates(["point", "arc"])
    df = df.sort_values(["point", "d"], kind="stable")
    df["rank"] = df.groupby("point").cumcount()
    df = df[df["rank"] < K].reset_index(drop=True)
    return df


def reference_arcs(trip: Trip, chunk: int = 2048):
    """Nearest reference-route arc for each observation (returns arc ids and distances)."""
    if trip.route is None:
        raise ValueError("trip has no reference route")
    r = trip.route
    A = trip.node_xy[trip.arc_from[r]]
    B = trip.node_xy[trip.arc_to[r]]
    n = len(trip.xy)
    best_arc = np.empty(n, dtype=np.int64)
    best_d = np.empty(n)
    for s in range(0, n, chunk):
        P = trip.xy[s:s + chunk]
        m = len(P)
        Pr = np.repeat(P, len(r), axis=0)
        Ar = np.tile(A, (m, 1))
        Br = np.tile(B, (m, 1))
        d, _ = point_segment(Pr, Ar, Br)
        d = d.reshape(m, len(r))
        j = d.argmin(1)
        best_arc[s:s + m] = r[j]
        best_d[s:s + m] = d[np.arange(m), j]
    return best_arc, best_d


def build_candidate_table(trip: Trip, K: int = 10, kq: int = 60) -> pd.DataFrame:
    """Candidate rows with features (and labels when a reference route exists)."""
    cand = generate_candidates(trip, K=K, kq=kq)
    n = len(trip.xy)
    A = trip.node_xy[trip.arc_from]
    B = trip.node_xy[trip.arc_to]
    L = np.linalg.norm(B - A, axis=1)

    # per-observation quantities
    prev_i = np.maximum(np.arange(n) - 1, 0)
    next_i = np.minimum(np.arange(n) + 1, n - 1)
    v = trip.xy[next_i] - trip.xy[prev_i]
    v_norm = np.linalg.norm(v, axis=1)
    dt_prev = np.clip(np.r_[1.0, np.diff(trip.t)], 0, 60)
    dt_prev[0] = 1.0
    dt_next = np.clip(np.r_[np.diff(trip.t), 1.0], 0, 60)
    dt_span = trip.t[next_i] - trip.t[prev_i]
    speed = np.where(dt_span > 0, v_norm / np.where(dt_span > 0, dt_span, 1), 0.0)

    n_nodes = int(max(trip.arc_from.max(), trip.arc_to.max())) + 1
    deg = np.bincount(trip.arc_from, minlength=n_nodes) + np.bincount(trip.arc_to, minlength=n_nodes)

    p = cand["point"].to_numpy()
    a = cand["arc"].to_numpy()
    cand["d_rel"] = cand["d"] - cand.groupby("point")["d"].transform("min")
    cand["n_cands"] = cand.groupby("point")["d"].transform("size")
    cand["arc_len"] = L[a]
    cand["dist_start"] = cand["t_param"] * L[a]
    cand["dist_end"] = (1 - cand["t_param"]) * L[a]
    u = B[a] - A[a]
    un = np.linalg.norm(u, axis=1)
    denom = v_norm[p] * un
    cos_h = np.where(denom > 0, np.abs((v[p] * u).sum(1)) / np.where(denom > 0, denom, 1), 0.5)
    cand["cos_heading"] = cos_h
    cand["heading_deg"] = np.degrees(np.arccos(np.clip(cos_h, 0, 1)))
    cand["speed"] = speed[p]
    cand["dt_prev"] = dt_prev[p]
    cand["dt_next"] = dt_next[p]
    cand["deg_from"] = deg[trip.arc_from[a]]
    cand["deg_to"] = deg[trip.arc_to[a]]

    cand["trip_id"] = trip.trip_id
    cand["t"] = trip.t[p]
    cand["x"] = trip.xy[p, 0]
    cand["y"] = trip.xy[p, 1]
    cand["arc_from"] = trip.arc_from[a]
    cand["arc_to"] = trip.arc_to[a]

    if trip.route is not None:
        ref_arc, ref_d = reference_arcs(trip)
        cand["ref_arc"] = ref_arc[p]
        cand["ref_dist"] = ref_d[p]
        cand["label"] = (a == cand["ref_arc"].to_numpy()).astype(np.int8)
    return cand
