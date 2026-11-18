"""Freeze the final classifier = Bayesian-tuned HistGradientBoosting.

Selection rule (fixed before the search, see README): the tuned model is kept if it has a higher
3-fold group-CV candidate macro F1 on the training trips than the untuned model. It does
(see results/metrics/untuned_cv_3fold.json and bayes_opt_best.json).

This script produces out-of-fold probabilities for the tuned model on the SAME 5-fold split used in
script 02, so the HMM decoder can be re-tuned on training data using the probabilities of the model
that will actually be deployed.
"""
import json, os, shutil, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import FEATURES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache"); RES = os.path.join(ROOT, "results", "metrics"); MOD = os.path.join(ROOT, "models")
SEED = 42
NAME = "HistGradientBoosting_tuned"


def main():
    best = json.load(open(os.path.join(RES, "bayes_opt_best.json")))
    untuned = json.load(open(os.path.join(RES, "untuned_cv_3fold.json")))
    assert best["best_value"] > untuned["cv_macro_f1"], "selection rule not met"
    p = best["best_params"]
    params = dict(learning_rate=p["learning_rate"], max_leaf_nodes=p["max_leaf_nodes"],
                  min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
                  max_iter=p["max_iter"])
    print("tuned params:", params, flush=True)

    train = pd.read_pickle(os.path.join(CACHE, "train.pkl"))
    X = train[FEATURES].to_numpy(dtype=np.float64); y = train["label"].to_numpy(); groups = train["trip_id"].to_numpy()
    oof = np.zeros(len(y))
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        m = HistGradientBoostingClassifier(**params, early_stopping=True, random_state=SEED)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        print("fold", k, "done", flush=True)
    np.save(os.path.join(CACHE, f"oof_{NAME}.npy"), oof.astype(np.float32))

    # the deployed artefact: refit on all 80 training trips with the tuned configuration
    full = HistGradientBoostingClassifier(**params, early_stopping=True, random_state=SEED)
    full.fit(X, y)
    import joblib
    joblib.dump(full, os.path.join(MOD, f"{NAME}.joblib"))
    shutil.copy(os.path.join(MOD, f"{NAME}.joblib"), os.path.join(MOD, "final_model.joblib"))
    json.dump({"name": NAME, "params": params, "selection_rule":
               "tuned model kept if its 3-fold group-CV candidate macro F1 on training trips exceeds the untuned model's",
               "untuned_cv_macro_f1": untuned["cv_macro_f1"], "tuned_cv_macro_f1": best["best_value"]},
              open(os.path.join(RES, "final_model.json"), "w"), indent=2)
    print("final model saved", flush=True)


if __name__ == "__main__":
    main()
