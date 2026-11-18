"""Figures (matplotlib, from actual results) and interactive Folium maps.

Outputs
  results/plots/*.png   EDA, model comparison, decoder comparison, MLP training curve
  results/maps/*.html   Folium maps: GPS points, road network, reference route, predicted routes
"""
import json, os, sys
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import FEATURES
from pathsnap.data import load_trip, trip_ids

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = "/home/claude/work/student_folder/Student folder"
CACHE = os.path.join(ROOT, "cache")
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(RES, "plots"); MAPS = os.path.join(RES, "maps")
for d in [PLOTS, MAPS]:
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name))
    plt.close(fig)


def eda_figures():
    st = pd.read_csv(os.path.join(RES, "data", "trip_stats_train.csv"))
    st_t = pd.read_csv(os.path.join(RES, "data", "trip_stats_student_test.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].hist(np.log10(st["points"]), bins=20, alpha=.7, label="train")
    ax[0].hist(np.log10(st_t["points"]), bins=20, alpha=.7, label="student test")
    ax[0].set_xlabel("log10 GPS observations per trip"); ax[0].legend()
    ax[1].scatter(st["arcs"], st["candidate_miss_points"] / st["points"] * 100, s=12, label="train")
    ax[1].scatter(st_t["arcs"], st_t["candidate_miss_points"] / st_t["points"] * 100, s=12, label="student test")
    ax[1].set_xscale("log"); ax[1].set_xlabel("road arcs in trip network (log)")
    ax[1].set_ylabel("candidate miss rate (%)"); ax[1].legend()
    ax[2].bar(["train", "student test"], [st["gaps_gt_30s"].sum(), st_t["gaps_gt_30s"].sum()])
    ax[2].set_ylabel("steps with gap > 30 s")
    save(fig, "eda_trip_overview.png")

    # reference distance distribution (label quality)
    cand = pd.read_pickle(os.path.join(CACHE, "train.pkl"))
    pts = cand.drop_duplicates(["trip_id", "point"])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(pts["ref_dist"].clip(upper=300), bins=60)
    ax.set_xlabel("distance from GPS point to reference route (m, clipped at 300)")
    ax.set_ylabel("observations")
    save(fig, "eda_reference_distance.png")

    # per-trip view of the two label-problem trips
    worst = pts.groupby("trip_id")["ref_dist"].apply(lambda s: (s > 50).mean()).sort_values(ascending=False).head(6)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(worst.index[::-1], worst.values[::-1] * 100)
    ax.set_xlabel("% observations > 50 m from reference route")
    save(fig, "eda_worst_reference_trips.png")


def trip_plot(partition, tid, name):
    trip = load_trip(DATA, partition, tid)
    A = trip.node_xy[trip.arc_from]; B = trip.node_xy[trip.arc_to]
    lo, hi = trip.xy.min(0) - 300, trip.xy.max(0) + 300
    inside = ((A >= lo) & (A <= hi)).all(1) | ((B >= lo) & (B <= hi)).all(1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    segs = np.stack([A[inside], B[inside]], axis=1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors="0.8", linewidths=0.5))
    if trip.route is not None:
        r = trip.route
        rs = np.stack([trip.node_xy[trip.arc_from[r]], trip.node_xy[trip.arc_to[r]]], axis=1)
        ax.add_collection(LineCollection(rs, colors="green", linewidths=1.6, label="reference route"))
    ax.scatter(trip.xy[:, 0], trip.xy[:, 1], s=2, c="tab:blue", label="GPS")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"{partition} trip {tid}")
    save(fig, name)


def model_figures():
    cv = pd.read_csv(os.path.join(RES, "metrics", "cv_folds.csv"))
    comp = pd.read_csv(os.path.join(RES, "metrics", "model_comparison.csv"), index_col=0)
    te = pd.read_csv(os.path.join(RES, "metrics", "student_test_by_model.csv"))
    names = comp.index.tolist()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    g = cv.groupby("model")["cand_macro_f1"].agg(["mean", "std"]).loc[names]
    ax.bar(names, g["mean"], yerr=g["std"], capsize=3)
    ax.set_ylabel("CV macro F1 (candidate level)"); ax.tick_params(axis="x", rotation=20)
    save(fig, "cv_macro_f1_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(te)); w = 0.38
    t2 = te.set_index("model").loc[names]
    ax.bar(x - w / 2, t2["obs_top1_acc"], w, label="Top-1 accuracy")
    ax.bar(x + w / 2, t2["cand_macro_f1"], w, label="Macro F1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20); ax.legend()
    ax.set_title("Student-test performance (10 trips)")
    save(fig, "student_test_top1_f1.png")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(names, t2["cand_log_loss"], color="tab:orange")
    ax.set_ylabel("Log loss (student test)"); ax.tick_params(axis="x", rotation=20)
    save(fig, "student_test_logloss.png")

    sel = json.load(open(os.path.join(RES, "metrics", "final_model.json")))["name"]
    test = pd.read_pickle(os.path.join(CACHE, "student_test.pkl"))
    p = np.load(os.path.join(CACHE, f"test_{sel}.npy"))
    cm = confusion_matrix(test["label"], (p >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{sel}: candidates")
    save(fig, f"confusion_{sel}.png")

    hist_path = os.path.join(RES, "metrics", "mlp_training_history.json")
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(range(1, len(h["loss_curve"]) + 1), h["loss_curve"], label="training loss")
        ax.plot(range(1, len(h["validation_scores"]) + 1), [1 - v for v in h["validation_scores"]],
                label="validation error (1 - accuracy)")
        ax.set_xlabel("epoch"); ax.legend()
        save(fig, "mlp_training_curves.png")


def decoder_figures():
    grid = pd.read_csv(os.path.join(RES, "metrics", "hmm_tuning_grid.csv"))
    piv = grid.pivot(index="unreachable_penalty", columns="beta", values="tune_pointwise_top1")
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for (i, j), v in np.ndenumerate(piv.values):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=7)
    ax.set_xlabel("beta (m)"); ax.set_ylabel("unreachable penalty")
    fig.colorbar(im, ax=ax, label="tuning top-1")
    save(fig, "hmm_tuning_grid.png")

    pt = pd.read_csv(os.path.join(RES, "metrics", "hmm_student_test_per_trip.csv"), dtype={"trip_id": str})
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(pt)); w = 0.38
    ax.bar(x - w / 2, pt["clf_pointwise_top1"], w, label="classifier only")
    ax.bar(x + w / 2, pt["hmm_pointwise_top1"], w, label="classifier + HMM")
    ax.set_xticks(x); ax.set_xticklabels(pt["trip_id"], rotation=45, fontsize=7)
    ax.set_ylabel("observation top-1 accuracy"); ax.legend()
    save(fig, "hmm_vs_classifier_per_trip.png")


def folium_maps(tid, partition="student_test"):
    """Interactive map for one trip: GPS, network, reference route, classifier and HMM routes."""
    trip = load_trip(DATA, partition, tid)
    nodes = np.loadtxt(os.path.join(DATA, partition, tid, f"{tid}.nodes"), ndmin=2)[:, :2]
    arcs = np.loadtxt(os.path.join(DATA, partition, tid, f"{tid}.arcs"), dtype=np.int64, ndmin=2)
    preds = pd.read_pickle(os.path.join(CACHE, "student_test_predictions.pkl"))[tid]
    lon, lat = trip.lonlat[:, 0], trip.lonlat[:, 1]
    pad = 0.002
    box = (lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad)
    A = nodes[arcs[:, 0]]; B = nodes[arcs[:, 1]]
    inside = (((A[:, 0] >= box[0]) & (A[:, 0] <= box[1]) & (A[:, 1] >= box[2]) & (A[:, 1] <= box[3]))
              | ((B[:, 0] >= box[0]) & (B[:, 0] <= box[1]) & (B[:, 1] >= box[2]) & (B[:, 1] <= box[3])))
    idx = np.flatnonzero(inside)[:20000]

    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=15, tiles="OpenStreetMap")
    net = folium.FeatureGroup(name="Road network (within trip extent)", show=True)
    for k in idx:
        folium.PolyLine([(A[k, 1], A[k, 0]), (B[k, 1], B[k, 0])], color="#9e9e9e", weight=1, opacity=.6).add_to(net)
    net.add_to(m)

    def route_layer(name, arc_ids, color, show):
        fg = folium.FeatureGroup(name=name, show=show)
        seq = arc_ids.astype(int)
        for a in seq:
            folium.PolyLine([(nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]),
                             (nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0])],
                            color=color, weight=4, opacity=.85,
                            tooltip=f"arc {a}").add_to(fg)
        fg.add_to(m)

    route_layer("Reference route (ground truth)", trip.route, "#2e7d32", True)
    route_layer("Classifier-only route", preds["classifier"]["arc"].to_numpy(), "#fb8c00", False)
    route_layer("Classifier + HMM route", preds["hmm"]["arc"].to_numpy(), "#c62828", True)

    gps = folium.FeatureGroup(name="Raw GPS observations", show=True)
    for la, lo_ in zip(lat, lon):
        folium.CircleMarker([la, lo_], radius=2, color="#1565c0", fill=True, fill_opacity=.9).add_to(gps)
    gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ("<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px;"
              "border:1px solid #888;font-size:12px'><b>Trip %s</b><br>"
              "<span style='color:#1565c0'>&#9679;</span> GPS observations<br>"
              "<span style='color:#2e7d32'>&#9473;</span> reference route<br>"
              "<span style='color:#fb8c00'>&#9473;</span> classifier-only route<br>"
              "<span style='color:#c62828'>&#9473;</span> classifier + HMM route<br>"
              "<span style='color:#9e9e9e'>&#9473;</span> road network</div>") % tid
    m.get_root().html.add_child(folium.Element(legend))
    out = os.path.join(MAPS, f"{partition}_{tid}.html")
    m.save(out)
    return out


if __name__ == "__main__":
    eda_figures()
    trip_plot("train", "00000010", "trip_train_00000010.png")
    trip_plot("student_test", "00000000", "trip_student_test_00000000.png")
    if os.path.exists(os.path.join(RES, "metrics", "model_comparison.csv")):
        model_figures()
    if os.path.exists(os.path.join(RES, "metrics", "hmm_tuning_grid.csv")):
        decoder_figures()
    if os.path.exists(os.path.join(CACHE, "student_test_predictions.pkl")):
        pt = pd.read_csv(os.path.join(RES, "metrics", "hmm_student_test_per_trip.csv"), dtype={"trip_id": str})
        # map: the trip with the largest HMM gain, plus the first trip for reference
        best = pt.assign(gain=pt["hmm_pointwise_top1"] - pt["clf_pointwise_top1"]).sort_values("gain").iloc[-1]["trip_id"]
        for tid in sorted({str(best), "00000000"}):
            print("map:", folium_maps(tid))
    print("done")
