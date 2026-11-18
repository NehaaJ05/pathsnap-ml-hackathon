"""Train and compare six classifiers (Section 3 of the guide).

  * Cross-validation: GroupKFold(5) by trip on the 80 training trips (no trip is split across folds).
  * Development test: refit on all 80 training trips, score the 10 student-test trips.
  * Saves out-of-fold probabilities (used later to tune the HMM on training data only).
Evaluator-test trips are never read here.
"""
import json, os, sys, time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import FEATURES
from pathsnap.metrics import (candidate_log_loss, candidate_macro_f1, observation_top1,
                              predicted_rows, route_continuity, observation_candidate_miss_rate)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache")
RES = os.path.join(ROOT, "results", "metrics")
MOD = os.path.join(ROOT, "models")
for d in [RES, MOD, os.path.join(ROOT, "results", "logs")]:
    os.makedirs(d, exist_ok=True)

SEED = 42
N_SPLITS = 5


def model_zoo():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300)),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=40, max_depth=14, min_samples_leaf=10, max_samples=0.25,
                                               n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                               early_stopping=True, random_state=SEED),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=15, early_stopping=True,
                                           n_iter_no_change=5, random_state=SEED)),
    }


def score(df, p):
    y = df["label"].to_numpy()
    pred = predicted_rows(df, p)
    return {
        "cand_macro_f1": candidate_macro_f1(y, p),
        "cand_log_loss": candidate_log_loss(y, p),
        "obs_top1_acc": observation_top1(df, p),
        "route_continuity": route_continuity(pred),
    }


def main():
    train = pd.read_pickle(os.path.join(CACHE, "train.pkl"))
    test = pd.read_pickle(os.path.join(CACHE, "student_test.pkl"))
    X = train[FEATURES].to_numpy(dtype=np.float64)
    y = train["label"].to_numpy()
    groups = train["trip_id"].to_numpy()
    Xt = test[FEATURES].to_numpy(dtype=np.float64)
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(X, y, groups))

    cv_rows, test_rows, summary = [], [], {}
    for name, proto in model_zoo().items():
        print(f"== {name}", flush=True)
        oof = np.zeros(len(y))
        fold_scores, fit_times, infer_times = [], [], []
        for k, (tr, va) in enumerate(folds):
            m = clone(proto)
            t0 = time.time(); m.fit(X[tr], y[tr]); fit_t = time.time() - t0
            t0 = time.time(); p = m.predict_proba(X[va])[:, 1]; inf_t = time.time() - t0
            oof[va] = p
            s = score(train.iloc[va], p)
            s.update({"model": name, "fold": k, "fit_seconds": fit_t,
                      "infer_seconds_per_1k_obs": inf_t / (train.iloc[va]["point"].nunique() / 1000)})
            fold_scores.append(s); cv_rows.append(s)
            print(f"   fold {k}: F1={s['cand_macro_f1']:.4f} top1={s['obs_top1_acc']:.4f} "
                  f"LL={s['cand_log_loss']:.4f} fit={fit_t:.1f}s", flush=True)
        np.save(os.path.join(CACHE, f"oof_{name}.npy"), oof.astype(np.float32))

        # refit on all training trips, evaluate on student-test
        final = clone(proto)
        t0 = time.time(); final.fit(X, y); fit_all = time.time() - t0
        t0 = time.time(); pt = final.predict_proba(Xt)[:, 1]; inf_all = time.time() - t0
        ts = score(test, pt)
        ts.update({"model": name, "fit_seconds_full": fit_all,
                   "infer_seconds_per_1k_obs": inf_all / (test["point"].nunique() / 1000),
                   "candidate_miss_rate": observation_candidate_miss_rate(test)})
        test_rows.append(ts)
        joblib.dump(final, os.path.join(MOD, f"{name}.joblib"))
        np.save(os.path.join(CACHE, f"test_{name}.npy"), pt.astype(np.float32))

        cv = pd.DataFrame(fold_scores)
        summary[name] = {
            "cv_macro_f1_mean": cv.cand_macro_f1.mean(), "cv_macro_f1_std": cv.cand_macro_f1.std(),
            "cv_top1_mean": cv.obs_top1_acc.mean(), "cv_top1_std": cv.obs_top1_acc.std(),
            "cv_logloss_mean": cv.cand_log_loss.mean(),
            "fit_seconds_cv_mean": cv.fit_seconds.mean(),
            "test_top1": ts["obs_top1_acc"], "test_macro_f1": ts["cand_macro_f1"],
            "test_logloss": ts["cand_log_loss"], "test_continuity": ts["route_continuity"],
            "test_miss_rate": ts["candidate_miss_rate"],
            "infer_seconds_per_1k_obs": ts["infer_seconds_per_1k_obs"],
            "fit_seconds_full": fit_all,
        }
        if name == "MLP":
            hist = {"loss_curve": [float(v) for v in final.steps[-1][1].loss_curve_],
                    "validation_scores": [float(v) for v in final.steps[-1][1].validation_scores_]}
            with open(os.path.join(RES, "mlp_training_history.json"), "w") as f:
                json.dump(hist, f, indent=2)

    pd.DataFrame(cv_rows).to_csv(os.path.join(RES, "cv_folds.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(RES, "student_test_by_model.csv"), index=False)
    comp = pd.DataFrame(summary).T.sort_values("cv_macro_f1_mean", ascending=False)
    comp.to_csv(os.path.join(RES, "model_comparison.csv"))
    print(comp.round(4).to_string(), flush=True)
    best = comp.index[0]
    with open(os.path.join(RES, "selected_model.json"), "w") as f:
        json.dump({"selected": best, "rule": "highest mean 5-fold CV candidate macro F1 on the 80 training trips"}, f, indent=2)
    print("selected:", best, flush=True)


if __name__ == "__main__":
    main()
