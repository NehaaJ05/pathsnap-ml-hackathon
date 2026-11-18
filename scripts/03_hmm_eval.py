"""Tune the HMM/Viterbi decoder on TRAINING trips only, then compare
classifier-only routes with classifier + HMM routes on the 10 student-test trips.

Tuning uses out-of-fold probabilities from script 02 (so no training trip is scored by a model
that saw it). The student-test set is used only for the final comparison.
"""
import itertools, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import FEATURES
from pathsnap.data import load_trip, trip_ids
from pathsnap.hmm import ArcGraph, HMMConfig, decode_trip
from pathsnap.metrics import route_continuity, sequence_metrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = "/home/claude/work/student_folder/Student folder"
CACHE = os.path.join(ROOT, "cache")
RES = os.path.join(ROOT, "results", "metrics")

TUNE_EVERY = 6          # tune on every 6th training trip (about 14 trips) to keep runtime reasonable
BETAS = [2.0, 5.0, 10.0, 20.0, 50.0]
PENALTIES = [5.0, 20.0]


def select_rows(cand: pd.DataFrame, trip_id: str) -> pd.DataFrame:
    return cand[cand["trip_id"] == trip_id].sort_values(["point", "rank"]).reset_index(drop=True)


def classifier_selection(cand: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
    tmp = pd.DataFrame({"point": cand["point"].to_numpy(), "p": probs})
    idx = tmp.groupby("point")["p"].idxmax()
    return idx.to_numpy()


def trip_eval(trip, cand, probs, graph, cfg, sel_method):
    n = len(trip.xy)
    if sel_method == "classifier":
        sel = classifier_selection(cand, probs)
        breaks = 0
        rows_idx = sel
    else:
        sel_pos, breaks = decode_trip(cand, probs, graph, trip.xy, trip.t, cfg)
        rows_idx = sel_pos[sel_pos >= 0]
    pred = cand.iloc[rows_idx].sort_values("point")
    pointwise = float(pred["label"].sum() / n)
    cont = route_continuity(pred)
    seq = sequence_metrics(pred["arc"].to_numpy(), trip.route) if trip.route is not None else {}
    return {"pointwise_top1": pointwise, "route_continuity": cont, "chain_breaks": int(breaks),
            **seq}, pred


def main():
    cfg_name = json.load(open(os.path.join(RES, "final_model.json")))["name"]
    print("selected classifier:", cfg_name, flush=True)
    train = pd.read_pickle(os.path.join(CACHE, "train.pkl"))
    test = pd.read_pickle(os.path.join(CACHE, "student_test.pkl"))
    oof = np.load(os.path.join(CACHE, f"oof_{cfg_name}.npy"))
    test_p = np.load(os.path.join(CACHE, f"test_{cfg_name}.npy"))
    train = train.assign(oof=oof)
    test = test.assign(p=test_p)

    # ---- tuning on training trips (out-of-fold probabilities) ----
    tune_ids = trip_ids(DATA, "train")[::TUNE_EVERY]
    grid = list(itertools.product(BETAS, PENALTIES))
    tune_rows = []
    trip_cache = {}
    for tid in tune_ids:
        trip = load_trip(DATA, "train", tid)
        cand = select_rows(train, tid)
        graph = ArcGraph(trip.node_xy, trip.arc_from, trip.arc_to)
        trip_cache[tid] = (trip, cand, cand["oof"].to_numpy(), graph)
    print(f"tuning on {len(tune_ids)} training trips, {len(grid)} grid points", flush=True)
    t0 = time.time()
    for beta, pen in grid:
        cfg = HMMConfig(beta=beta, unreachable_penalty=pen)
        accs, conts, seqf = [], [], []
        for tid in tune_ids:
            trip, cand, probs, graph = trip_cache[tid]
            r, _ = trip_eval(trip, cand, probs, graph, cfg, "hmm")
            accs.append(r["pointwise_top1"]); conts.append(r["route_continuity"]); seqf.append(r["seq_f1"])
        row = {"beta": beta, "unreachable_penalty": pen,
               "tune_pointwise_top1": float(np.mean(accs)), "tune_continuity": float(np.mean(conts)),
               "tune_seq_f1": float(np.mean(seqf))}
        tune_rows.append(row)
        print(row, flush=True)
    base_clf = []
    for tid in tune_ids:
        trip, cand, probs, graph = trip_cache[tid]
        r, _ = trip_eval(trip, cand, probs, graph, None, "classifier")
        base_clf.append(r["pointwise_top1"])
    tune_df = pd.DataFrame(tune_rows).sort_values("tune_pointwise_top1", ascending=False)
    tune_df.to_csv(os.path.join(RES, "hmm_tuning_grid.csv"), index=False)
    best = tune_df.iloc[0]
    cfg = HMMConfig(beta=float(best["beta"]), unreachable_penalty=float(best["unreachable_penalty"]))
    print("tuning took %.1fs; best:" % (time.time() - t0), cfg, flush=True)
    print("classifier-only pointwise top1 on tuning trips: %.4f" % np.mean(base_clf), flush=True)

    # ---- student-test comparison (frozen decoder configuration) ----
    rows, preds = [], {}
    for tid in trip_ids(DATA, "student_test"):
        trip = load_trip(DATA, "student_test", tid)
        cand = select_rows(test, tid)
        probs = cand["p"].to_numpy()
        graph = ArcGraph(trip.node_xy, trip.arc_from, trip.arc_to)
        clf, clf_pred = trip_eval(trip, cand, probs, graph, cfg, "classifier")
        hmm, hmm_pred = trip_eval(trip, cand, probs, graph, cfg, "hmm")
        rows.append({"trip_id": tid, "points": len(trip.xy),
                     **{f"clf_{k}": v for k, v in clf.items()},
                     **{f"hmm_{k}": v for k, v in hmm.items()}})
        preds[tid] = {"classifier": clf_pred[["point", "arc", "arc_from", "arc_to", "x", "y", "label"]],
                      "hmm": hmm_pred[["point", "arc", "arc_from", "arc_to", "x", "y", "label"]]}
        print(tid, "clf top1 %.3f -> hmm top1 %.3f" % (clf["pointwise_top1"], hmm["pointwise_top1"]), flush=True)
    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(os.path.join(RES, "hmm_student_test_per_trip.csv"), index=False)
    pd.to_pickle(preds, os.path.join(CACHE, "student_test_predictions.pkl"))

    summary = {
        "decoder_config": cfg.__dict__,
        "tuning_trips": tune_ids,
        "student_test_mean": {c: float(per_trip[c].mean()) for c in per_trip.columns if c not in ("trip_id", "points")},
    }
    with open(os.path.join(RES, "hmm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["student_test_mean"], indent=2), flush=True)


if __name__ == "__main__":
    main()
