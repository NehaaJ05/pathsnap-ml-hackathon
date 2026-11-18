"""Evaluation metrics used across models and the HMM decoder.

Definitions (also stated in the report):
  * Candidate-level Macro F1 / Log Loss: binary classification of every
    (observation, candidate) row; macro-averaged F1 over the two classes at threshold 0.5.
  * Observation-level Top-1 Accuracy: for each GPS observation take the candidate with
    the highest score; it is correct iff that candidate is the reference arc.
    Observations whose reference arc is missing from the candidate set count as wrong.
  * Candidate Miss Rate: share of observations whose reference arc is not a candidate.
  * Route Continuity: share of consecutive predicted arcs that are the same arc or share
    at least one network node (i.e. the predicted path does not jump between
    disconnected road segments).
  * Sequence-level: arc-set precision / recall / F1 of the predicted route against the
    reference route (duplicates removed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def candidate_macro_f1(y, p, threshold: float = 0.5) -> float:
    return float(f1_score(y, (np.asarray(p) >= threshold).astype(int), average="macro"))


def candidate_log_loss(y, p) -> float:
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))


def predicted_rows(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """For each observation, the index (into df) of the highest-scoring candidate."""
    tmp = pd.DataFrame({"trip_id": df["trip_id"].values, "point": df["point"].values, "p": p})
    idx = tmp.groupby(["trip_id", "point"])["p"].idxmax()
    return df.iloc[idx.values]


def observation_top1(df: pd.DataFrame, p: np.ndarray) -> float:
    best = predicted_rows(df, p)
    return float(best["label"].mean())


def observation_candidate_miss_rate(df: pd.DataFrame) -> float:
    has_pos = df.groupby(["trip_id", "point"])["label"].max()
    return float(1 - has_pos.mean())


def route_continuity(pred: pd.DataFrame) -> float:
    """pred: rows already chosen per observation (one row per observation, with arc_from/arc_to/arc)."""
    total, ok = 0, 0
    for _, g in pred.sort_values(["trip_id", "point"]).groupby("trip_id", sort=False):
        a = g["arc"].to_numpy()
        f = g["arc_from"].to_numpy()
        t = g["arc_to"].to_numpy()
        same = a[1:] == a[:-1]
        share = (t[:-1] == f[1:]) | (t[:-1] == t[1:]) | (f[:-1] == f[1:]) | (f[:-1] == t[1:])
        ok += int(np.sum(same | share))
        total += len(a) - 1
    return float(ok / total) if total else float("nan")


def sequence_metrics(pred_arcs: np.ndarray, ref_arcs: np.ndarray) -> dict:
    p_set, r_set = set(pred_arcs.tolist()), set(ref_arcs.tolist())
    inter = len(p_set & r_set)
    prec = inter / len(p_set) if p_set else 0.0
    rec = inter / len(r_set) if r_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"seq_precision": prec, "seq_recall": rec, "seq_f1": f1}
