"""HMM / Viterbi route reconstruction on top of classifier candidate probabilities.

Hidden states at observation i are the K candidate road arcs kept for that observation.
  * Emission  e(i, c) = log( p_c / sum_c' p_c' )      (classifier probability, normalised per observation)
  * Transition between candidate a (obs i-1) and candidate b (obs i), Newson & Krumm style:
        route distance rd(a, b): distance travelled along the network from the projected
                                 point on a to the projected point on b (directed Dijkstra,
                                 cut off at a bound); if a == b, the distance along that arc.
        gc                       straight-line distance between the two GPS observations.
        log T(a, b) = -|rd - gc| / beta           if b is reachable from a within the bound
        log T(a, b) = -unreachable_penalty        otherwise (invalid transition)
  * Chains are broken (no transition is scored) where the time gap exceeds gap_s, the implied
    speed exceeds max_speed, or an observation has no candidates. Each chain is decoded separately.

beta, unreachable_penalty and gap_s are the decoder parameters. They are tuned on training
trips only (out-of-fold classifier probabilities) and then frozen for the student-test run.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class HMMConfig:
    beta: float = 10.0
    unreachable_penalty: float = 20.0
    gap_s: float = 30.0
    max_speed: float = 40.0        # m/s
    reach_factor: float = 3.0      # Dijkstra bound = reach_factor * gc + reach_margin
    reach_margin: float = 300.0    # metres
    eps: float = 1e-9


class ArcGraph:
    """Directed adjacency over road nodes (arc = edge from arc_from to arc_to)."""

    def __init__(self, node_xy: np.ndarray, arc_from: np.ndarray, arc_to: np.ndarray):
        self.arc_from = arc_from
        self.arc_to = arc_to
        self.L = np.linalg.norm(node_xy[arc_to] - node_xy[arc_from], axis=1)
        order = np.argsort(arc_from, kind="stable")
        counts = np.bincount(arc_from, minlength=node_xy.shape[0])
        self.indptr = np.r_[0, np.cumsum(counts)].tolist()
        self.heads = arc_to[order].tolist()
        self.w = self.L[order].tolist()

    def dijkstra(self, src: int, cutoff: float) -> dict:
        dist = {src: 0.0}
        pq = [(0.0, src)]
        indptr, heads, w = self.indptr, self.heads, self.w
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf):
                continue
            for k in range(indptr[u], indptr[u + 1]):
                v = heads[k]
                nd = d + w[k]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist


def decode_trip(cand: pd.DataFrame, probs: np.ndarray, graph: ArcGraph, xy: np.ndarray,
                t: np.ndarray, cfg: HMMConfig):
    """Viterbi-decode one trip.

    cand:  candidate rows for ONE trip (columns: point, arc, t_param, arc_from, arc_to),
           sorted by point. probs: classifier score for each row.
    Returns: selected row position (into cand) for each observation (-1 if none),
             and the number of broken chains.
    """
    cand = cand.reset_index(drop=True)
    pts = cand["point"].to_numpy()
    arcs = cand["arc"].to_numpy()
    tpar = cand["t_param"].to_numpy()
    n = len(xy)
    starts = np.searchsorted(pts, np.arange(n + 1))  # rows of point i are starts[i]:starts[i+1]

    selected = np.full(n, -1, dtype=np.int64)
    breaks = 0
    chain_points: list[int] = []
    chain_bp: list[np.ndarray] = []
    delta = None
    prev_i = -1

    def finish_chain():
        nonlocal chain_points, chain_bp
        if not chain_points:
            return
        last = chain_points[-1]
        lo, hi = starts[last], starts[last + 1]
        k = int(np.argmax(delta))
        for idx in range(len(chain_points) - 1, -1, -1):
            i = chain_points[idx]
            selected[i] = starts[i] + k
            if idx > 0:
                k = int(chain_bp[idx][k])
        chain_points, chain_bp = [], []

    for i in range(n):
        lo, hi = starts[i], starts[i + 1]
        K = hi - lo
        if K == 0:
            finish_chain()
            breaks += 1
            prev_i = -1
            continue
        p = probs[lo:hi].astype(float)
        e = np.log(p / max(p.sum(), 1e-300) + cfg.eps)

        if prev_i < 0:
            finish_chain()
            delta = e
            chain_points = [i]
            chain_bp = [np.zeros(K, dtype=np.int64)]
            prev_i = i
            continue

        gc = float(np.hypot(*(xy[i] - xy[prev_i])))
        dt = float(t[i] - t[prev_i])
        speed_ok = gc <= cfg.max_speed * max(dt, 1.0) + 50.0
        if dt > cfg.gap_s or dt <= 0 or not speed_ok:
            finish_chain()
            breaks += 1
            delta = e
            chain_points = [i]
            chain_bp = [np.zeros(K, dtype=np.int64)]
            prev_i = i
            continue

        plo, phi = starts[prev_i], starts[prev_i + 1]
        Kp = phi - plo
        cutoff = cfg.reach_factor * gc + cfg.reach_margin
        reach = {}
        for ta_row in range(plo, phi):
            node = graph.arc_to[arcs[ta_row]]
            if node not in reach:
                reach[node] = graph.dijkstra(int(node), cutoff)

        T = np.full((Kp, K), -cfg.unreachable_penalty)
        for ai in range(Kp):
            ra = plo + ai
            a_arc, a_t = arcs[ra], tpar[ra]
            La = graph.L[a_arc]
            for bi in range(K):
                rb = lo + bi
                b_arc, b_t = arcs[rb], tpar[rb]
                Lb = graph.L[b_arc]
                if a_arc == b_arc:
                    rd = abs(b_t - a_t) * La
                else:
                    D = reach[int(graph.arc_to[a_arc])].get(int(graph.arc_from[b_arc]))
                    if D is None:
                        continue
                    rd = (1 - a_t) * La + D + b_t * Lb
                T[ai, bi] = -abs(rd - gc) / cfg.beta

        M = delta[:, None] + T
        bp = np.argmax(M, axis=0)
        delta = M[bp, np.arange(K)] + e
        chain_points.append(i)
        chain_bp.append(bp.astype(np.int64))
        prev_i = i

    finish_chain()
    return selected, breaks
