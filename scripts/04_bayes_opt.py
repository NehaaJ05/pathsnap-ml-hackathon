"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model family.

Objective: mean candidate-level Macro F1 over 3 GroupKFold folds of the 80 TRAINING trips.
The student-test set is only used afterwards, to compare tuned and untuned models.
Every trial is written to results/metrics/optuna_trials.csv (the search record).
"""
import json, os, sys, time
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import FEATURES
from pathsnap.metrics import candidate_macro_f1, observation_top1, candidate_log_loss

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache")
RES = os.path.join(ROOT, "results", "metrics")
MOD = os.path.join(ROOT, "models")
SEED = 42
N_TRIALS = int(os.environ.get("N_TRIALS", 20))
N_SPLITS = 3
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build(family, trial):
    if family == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 400, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 400),
            early_stopping=True, random_state=SEED)
    if family == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 20, 80),
            max_depth=trial.suggest_int("max_depth", 6, 18),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 50, log=True),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            max_samples=trial.suggest_float("max_samples", 0.1, 0.4),
            n_jobs=-1, random_state=SEED)
    if family == "DecisionTree":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 4, 16),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            random_state=SEED)
    if family == "MLP":
        h = trial.suggest_categorical("hidden", ["32", "64", "64-32", "128-64"])
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(int(x) for x in h.split("-")),
            alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            max_iter=15, early_stopping=True, n_iter_no_change=5, random_state=SEED))
    if family == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 1e2, log=True), max_iter=300))
    raise ValueError(family)


def main():
    family = json.load(open(os.path.join(RES, "selected_model.json")))["selected"]
    print("optimising family:", family, "trials:", N_TRIALS, flush=True)
    train = pd.read_pickle(os.path.join(CACHE, "train.pkl"))
    X = train[FEATURES].to_numpy(dtype=np.float64)
    y = train["label"].to_numpy()
    groups = train["trip_id"].to_numpy()
    folds = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))

    def objective(trial):
        t0 = time.time()
        scores, top1s, lls = [], [], []
        for tr, va in folds:
            m = build(family, trial)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            scores.append(candidate_macro_f1(y[va], p))
            top1s.append(observation_top1(train.iloc[va], p))
            lls.append(candidate_log_loss(y[va], p))
        trial.set_user_attr("cv_top1", float(np.mean(top1s)))
        trial.set_user_attr("cv_logloss", float(np.mean(lls)))
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: macroF1={np.mean(scores):.4f} top1={np.mean(top1s):.4f} "
              f"({time.time()-t0:.0f}s) {trial.params}", flush=True)
        return float(np.mean(scores))

    # untuned baseline on the SAME 3-fold objective (sklearn defaults plus the zoo settings from script 02)
    untuned_params = {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                      "l2_regularization": 0.0, "max_iter": 200}
    if family == "HistGradientBoosting":
        f1s, t1s, lls = [], [], []
        for tr, va in folds:
            m = HistGradientBoostingClassifier(**untuned_params, early_stopping=True, random_state=SEED)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            f1s.append(candidate_macro_f1(y[va], p)); t1s.append(observation_top1(train.iloc[va], p))
            lls.append(candidate_log_loss(y[va], p))
        json.dump({"params": untuned_params, "cv_macro_f1": float(np.mean(f1s)),
                   "cv_top1": float(np.mean(t1s)), "cv_logloss": float(np.mean(lls))},
                  open(os.path.join(RES, "untuned_cv_3fold.json"), "w"), indent=2)
        print("untuned 3-fold macroF1=%.4f top1=%.4f" % (np.mean(f1s), np.mean(t1s)), flush=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(RES, "optuna_trials.csv"), index=False)

    best = study.best_trial
    out = {"family": family, "objective": "mean 3-fold group-CV candidate macro F1 (training trips only)",
           "n_trials": N_TRIALS, "best_value": best.value, "best_params": best.params,
           "best_cv_top1": best.user_attrs.get("cv_top1"), "best_cv_logloss": best.user_attrs.get("cv_logloss"),
           "search_seconds_total": float(sum(t.user_attrs.get("seconds", 0) for t in study.trials))}
    json.dump(out, open(os.path.join(RES, "bayes_opt_best.json"), "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)

    # refit tuned model on all training trips, evaluate on student-test (for the tuned vs untuned table)
    test = pd.read_pickle(os.path.join(CACHE, "student_test.pkl"))
    Xt = test[FEATURES].to_numpy(dtype=np.float64)
    fixed = optuna.trial.FixedTrial(best.params)
    tuned = build(family, fixed)
    t0 = time.time(); tuned.fit(X, y); fit_s = time.time() - t0
    t0 = time.time(); pt = tuned.predict_proba(Xt)[:, 1]; inf_s = time.time() - t0
    from pathsnap.metrics import route_continuity, predicted_rows, observation_candidate_miss_rate
    res = {"model": f"{family} (tuned)",
           "student_test_top1": observation_top1(test, pt),
           "student_test_macro_f1": candidate_macro_f1(test["label"].to_numpy(), pt),
           "student_test_logloss": candidate_log_loss(test["label"].to_numpy(), pt),
           "student_test_continuity": route_continuity(predicted_rows(test, pt)),
           "fit_seconds_full": fit_s,
           "infer_seconds_per_1k_obs": inf_s / (test["point"].nunique() / 1000)}
    json.dump(res, open(os.path.join(RES, "tuned_student_test.json"), "w"), indent=2)
    import joblib
    joblib.dump(tuned, os.path.join(MOD, f"{family}_tuned.joblib"))
    np.save(os.path.join(CACHE, f"test_{family}_tuned.npy"), pt.astype(np.float32))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
