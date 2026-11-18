"""Turn result files into Markdown tables for the report (so numbers are never copied by hand)."""
import json, os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
M = os.path.join(ROOT, "results", "metrics")
D = os.path.join(ROOT, "results", "data")
OUT = os.path.join(ROOT, "results", "tables")
os.makedirs(OUT, exist_ok=True)


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def main():
    comp = pd.read_csv(os.path.join(M, "model_comparison.csv"), index_col=0)
    te = pd.read_csv(os.path.join(M, "student_test_by_model.csv")).set_index("model")
    fam = {"LogisticRegression": "Linear", "GaussianNB": "Probabilistic", "DecisionTree": "Tree",
           "RandomForest": "Ensemble (bagging)", "HistGradientBoosting": "Boosting", "MLP": "Neural network"}
    rows = []
    for name in comp.index:
        r = comp.loc[name]; t = te.loc[name]
        rows.append({
            "Model": name, "Family": fam[name],
            "CV Macro F1": f"{r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f}",
            "Top-1 (CV)": f"{pct(r.cv_top1_mean)}%",
            "Route continuity (test)": pct(t.route_continuity),
            "Test Top-1": pct(t.obs_top1_acc) + "%",
            "Test Macro F1": f"{t.cand_macro_f1:.4f}",
            "Test Log Loss": f"{t.cand_log_loss:.4f}",
            "Inference (s / 1k obs)": f"{t.infer_seconds_per_1k_obs:.4f}",
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_comparison_table.csv"), index=False)
    with open(os.path.join(OUT, "model_comparison_table.md"), "w") as f:
        f.write(md(pd.DataFrame(rows)))

    if os.path.exists(os.path.join(M, "hmm_summary.json")):
        h = json.load(open(os.path.join(M, "hmm_summary.json")))
        pt = pd.read_csv(os.path.join(M, "hmm_student_test_per_trip.csv"), dtype={"trip_id": str})
        per = pt[["trip_id", "points", "clf_pointwise_top1", "hmm_pointwise_top1", "clf_seq_f1", "hmm_seq_f1",
                  "clf_route_continuity", "hmm_route_continuity", "hmm_chain_breaks"]].copy()
        for c in ["clf_pointwise_top1", "hmm_pointwise_top1", "clf_seq_f1", "hmm_seq_f1",
                  "clf_route_continuity", "hmm_route_continuity"]:
            per[c] = per[c].map(lambda v: pct(v, 1))
        per["hmm_chain_breaks"] = per["hmm_chain_breaks"].astype(int)
        per.columns = ["Trip", "Observations", "Clf top-1", "HMM top-1", "Clf seq F1", "HMM seq F1",
                       "Clf continuity", "HMM continuity", "HMM chain breaks"]
        with open(os.path.join(OUT, "hmm_per_trip.md"), "w") as f:
            f.write(md(per))
        m = h["student_test_mean"]
        summ = pd.DataFrame([
            {"Method": "Classifier only", "Observation top-1": pct(m["clf_pointwise_top1"]) + "%",
             "Route continuity": pct(m["clf_route_continuity"]) + "%",
             "Sequence precision": pct(m["clf_seq_precision"]) + "%", "Sequence recall": pct(m["clf_seq_recall"]) + "%",
             "Sequence F1": pct(m["clf_seq_f1"]) + "%"},
            {"Method": "Classifier + HMM/Viterbi", "Observation top-1": pct(m["hmm_pointwise_top1"]) + "%",
             "Route continuity": pct(m["hmm_route_continuity"]) + "%",
             "Sequence precision": pct(m["hmm_seq_precision"]) + "%", "Sequence recall": pct(m["hmm_seq_recall"]) + "%",
             "Sequence F1": pct(m["hmm_seq_f1"]) + "%"},
        ])
        with open(os.path.join(OUT, "hmm_summary.md"), "w") as f:
            f.write(md(summ))

    if os.path.exists(os.path.join(M, "hmm_tuning_grid.csv")):
        g = pd.read_csv(os.path.join(M, "hmm_tuning_grid.csv"))
        g = g[["beta", "unreachable_penalty", "tune_pointwise_top1", "tune_seq_f1", "tune_continuity"]].copy()
        g["tune_pointwise_top1"] = g["tune_pointwise_top1"].map(lambda v: pct(v, 2))
        g["tune_seq_f1"] = g["tune_seq_f1"].map(lambda v: pct(v, 2))
        g["tune_continuity"] = g["tune_continuity"].map(lambda v: pct(v, 2))
        g.columns = ["beta (m)", "Unreachable penalty", "Tuning top-1 (%)", "Tuning seq F1 (%)", "Tuning continuity (%)"]
        with open(os.path.join(OUT, "hmm_tuning.md"), "w") as f:
            f.write(md(g))

    stats = pd.read_csv(os.path.join(D, "trip_stats_train.csv"))
    inv = json.load(open(os.path.join(D, "dataset_inventory.json")))
    with open(os.path.join(OUT, "dataset_inventory.json"), "w") as f:
        json.dump(inv, f, indent=2)
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
