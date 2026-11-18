"""In-sample (training-trip) scores of each saved model, to judge over/underfitting.

Compares training-trip metrics with the 5-fold CV and student-test metrics from script 02.
"""
import os, sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.candidates import FEATURES
from pathsnap.metrics import candidate_macro_f1, observation_top1, candidate_log_loss

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "cache"); RES = os.path.join(ROOT, "results", "metrics"); MOD = os.path.join(ROOT, "models")

train = pd.read_pickle(os.path.join(CACHE, "train.pkl"))
X = train[FEATURES].to_numpy(dtype=np.float64); y = train["label"].to_numpy()
rows = []
for name in ["LogisticRegression", "GaussianNB", "DecisionTree", "RandomForest", "HistGradientBoosting", "MLP"]:
    m = joblib.load(os.path.join(MOD, f"{name}.joblib"))
    p = m.predict_proba(X)[:, 1]
    rows.append({"model": name,
                 "train_macro_f1": candidate_macro_f1(y, p),
                 "train_top1": observation_top1(train, p),
                 "train_logloss": candidate_log_loss(y, p)})
    print(rows[-1], flush=True)
df = pd.DataFrame(rows).set_index("model")
cv = pd.read_csv(os.path.join(RES, "cv_folds.csv")).groupby("model")[["cand_macro_f1", "obs_top1_acc", "cand_log_loss"]].mean()
cv.columns = ["cv_macro_f1", "cv_top1", "cv_logloss"]
te = pd.read_csv(os.path.join(RES, "student_test_by_model.csv")).set_index("model")[["cand_macro_f1", "obs_top1_acc", "cand_log_loss"]]
te.columns = ["test_macro_f1", "test_top1", "test_logloss"]
out = df.join(cv).join(te)
out["gap_train_minus_cv_top1"] = out["train_top1"] - out["cv_top1"]
out.to_csv(os.path.join(RES, "train_cv_test_gap.csv"))
print(out.round(4).to_string())
