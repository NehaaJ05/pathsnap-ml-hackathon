"""End-to-end inference: raw trip -> candidates -> classifier -> HMM/Viterbi route.

The prediction path never reads the reference route. Scoring against the reference is a
separate function, used only by the evaluation code and the API's server-side evaluation.
"""
from __future__ import annotations

import numpy as np

from .candidates import FEATURES, build_candidate_table, reference_arcs
from .data import Trip
from .hmm import ArcGraph, HMMConfig, decode_trip


def predict_trip(trip: Trip, model, cfg: HMMConfig) -> dict:
    cand = build_candidate_table(_without_route(trip))  # no reference route -> no labels
    cand = cand.reset_index(drop=True)
    n = len(trip.xy)
    probs = model.predict_proba(cand[FEATURES].to_numpy(dtype=np.float64))[:, 1]

    # classifier-only: highest-scoring candidate per observation
    best = cand.assign(p=probs).groupby("point")["p"].idxmax().to_numpy()
    clf_arc = np.full(n, -1, dtype=np.int64)
    clf_arc[cand["point"].to_numpy()[best]] = cand["arc"].to_numpy()[best]

    # classifier + HMM/Viterbi
    graph = ArcGraph(trip.node_xy, trip.arc_from, trip.arc_to)
    sel, breaks = decode_trip(cand, probs, graph, trip.xy, trip.t, cfg)
    hmm_arc = np.full(n, -1, dtype=np.int64)
    ok = sel >= 0
    hmm_arc[np.flatnonzero(ok)] = cand["arc"].to_numpy()[sel[ok]]
    return {"classifier_arcs": clf_arc, "hmm_arcs": hmm_arc, "chain_breaks": int(breaks),
            "observations": n, "candidate_rows": int(len(cand))}


def score_against_reference(trip: Trip, arcs: np.ndarray) -> dict:
    """Observation top-1 and sequence metrics using the reference route. Server-side only."""
    if trip.route is None:
        raise ValueError("no reference route for this trip")
    ref_arc, _ = reference_arcs(trip)
    n = len(arcs)
    correct = float(np.mean(arcs == ref_arc))
    pred_seq = arcs[arcs >= 0]
    p_set, r_set = set(pred_seq.tolist()), set(trip.route.tolist())
    inter = len(p_set & r_set)
    prec = inter / len(p_set) if p_set else 0.0
    rec = inter / len(r_set) if r_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"observation_top1": correct, "route_precision": prec, "route_recall": rec, "route_f1": f1}


def _without_route(trip: Trip) -> Trip:
    from dataclasses import replace
    return replace(trip, route=None)
