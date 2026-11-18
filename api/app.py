"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
  GET  /health            service status and loaded artifacts
  GET  /trips             evaluator trip IDs only (no labels)
  POST /predict           body {"trip_id": "..."}; runs the full prediction pipeline on the
                          evaluator trip, scores it against the reference route, and returns map
                          layers for the raw GPS trace, predicted route, and reference route.

Configuration (environment variables)
  PATHSNAP_EVAL_ROOT   folder containing the evaluator trips (<root>/evaluator_test/<trip_id>/...)
  PATHSNAP_MODEL       path to the selected classifier (joblib)
  PATHSNAP_HMM_CONFIG  path to the frozen decoder configuration (JSON)
  PATHSNAP_LOG         request log file (default: logs/api_requests.log)

Run:  uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsnap.data import load_trip, trip_ids
from pathsnap.hmm import HMMConfig
from pathsnap.pipeline import predict_trip, score_against_reference

EVAL_ROOT = os.environ.get("PATHSNAP_EVAL_ROOT", os.path.join(ROOT, "evaluator_data"))
PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION", "evaluator_test")
MODEL_PATH = os.environ.get("PATHSNAP_MODEL", os.path.join(ROOT, "models", "final_model.joblib"))
HMM_PATH = os.environ.get("PATHSNAP_HMM_CONFIG", os.path.join(ROOT, "models", "hmm_config.json"))
LOG_PATH = os.environ.get("PATHSNAP_LOG", os.path.join(ROOT, "logs", "api_requests.log"))
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("pathsnap.api")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
for h in [logging.StreamHandler(), logging.FileHandler(LOG_PATH)]:
    h.setFormatter(_fmt)
    logger.addHandler(h)

app = FastAPI(title="PathSnap", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {}


@app.on_event("startup")
def _load():
    STATE["model"] = joblib.load(MODEL_PATH)
    with open(HMM_PATH) as f:
        STATE["cfg"] = HMMConfig(**json.load(f))
    STATE["model_name"] = os.path.basename(MODEL_PATH)
    logger.info("loaded model=%s hmm=%s eval_root=%s", MODEL_PATH, STATE["cfg"], EVAL_ROOT)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE.get("model_name"), "hmm_config": STATE["cfg"].__dict__,
            "eval_root": EVAL_ROOT}


@app.get("/trips")
def trips():
    ids = trip_ids(EVAL_ROOT, PARTITION) if os.path.isdir(os.path.join(EVAL_ROOT, PARTITION)) else []
    logger.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    if req.trip_id not in trip_ids(EVAL_ROOT, PARTITION):
        logger.warning("POST /predict unknown trip_id=%s", req.trip_id)
        raise HTTPException(status_code=404, detail="unknown evaluator trip")
    logger.info("POST /predict trip_id=%s start", req.trip_id)

    # 1. prediction path: raw GPS and road network only
    raw = load_trip(EVAL_ROOT, PARTITION, req.trip_id, with_route=False)
    out = predict_trip(raw, STATE["model"], STATE["cfg"])

    # 2. Load the labelled route for evaluation and for the comparison map.
    ref = load_trip(EVAL_ROOT, PARTITION, req.trip_id, with_route=True)
    clf_scores = score_against_reference(ref, out["classifier_arcs"])
    hmm_scores = score_against_reference(ref, out["hmm_arcs"])

    # All map geometry is returned as [lon, lat].  Leaflet reverses it to [lat, lon].
    nodes_ll = _node_lonlat(EVAL_ROOT, req.trip_id)
    network_arcs = raw_arcs(EVAL_ROOT, req.trip_id)
    predicted_lonlat = _route_segments_lonlat(out["hmm_arcs"], network_arcs, nodes_ll)
    actual_lonlat = _route_segments_lonlat(ref.route, network_arcs, nodes_ll)

    elapsed = time.time() - t0
    result = {
        "trip_id": req.trip_id,
        "observations": out["observations"],
        "chain_breaks": out["chain_breaks"],
        "classifier_only": clf_scores,
        "classifier_plus_hmm": hmm_scores,
        "gps_trace_lonlat": raw.lonlat.astype(float).tolist(),
        "predicted_segments_lonlat": predicted_lonlat,
        "actual_segments_lonlat": actual_lonlat,
        "elapsed_seconds": round(elapsed, 3),
        "served_at": datetime.utcnow().isoformat() + "Z",
    }
    logger.info("POST /predict trip_id=%s done in %.2fs hmm_top1=%.4f clf_top1=%.4f breaks=%d",
                req.trip_id, elapsed, hmm_scores["observation_top1"], clf_scores["observation_top1"],
                out["chain_breaks"])
    return result


_ARC_CACHE: dict = {}


def raw_arcs(root, trip_id):
    key = ("arcs", trip_id)
    if key not in _ARC_CACHE:
        arr = np.loadtxt(os.path.join(root, PARTITION, trip_id, f"{trip_id}.arcs"), dtype=np.int64, ndmin=2)
        _ARC_CACHE[key] = arr
    return _ARC_CACHE[key]


def _node_lonlat(root, trip_id):
    key = ("nodes", trip_id)
    if key not in _ARC_CACHE:
        _ARC_CACHE[key] = np.loadtxt(os.path.join(root, PARTITION, trip_id, f"{trip_id}.nodes"), ndmin=2)[:, :2]
    return _ARC_CACHE[key]


def _route_segments_lonlat(route, arcs, nodes_lonlat):
    """Convert an arc sequence into map-ready line segments, ignoring repeated samples."""
    seq = [int(arc) for arc in route if arc >= 0]
    dedup = [arc for i, arc in enumerate(seq) if i == 0 or arc != seq[i - 1]]
    segments = []
    for arc_id in dedup:
        start, end = arcs[arc_id]
        segments.append([
            [float(nodes_lonlat[start, 0]), float(nodes_lonlat[start, 1])],
            [float(nodes_lonlat[end, 0]), float(nodes_lonlat[end, 1])],
        ])
    return segments
