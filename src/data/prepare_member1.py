"""Reproducible Member 1 entry point. Run from the repository root.

python -m src.data.prepare_member1 --archive "Student folder.zip"
python -m src.data.prepare_member1 --raw-root data/raw --trip-limit 2
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import importlib.metadata
import numpy as np
import pandas as pd
from src.data.dataset_loader import extract_student_archive, load_trip
from src.data.trip_inventory import build_inventory
from src.preprocessing.gps_cleaning import clean_track
from src.candidates.candidate_utils import RoadNetwork
from src.candidates.candidate_generator import CandidateGenerator, CandidateConfig
from src.features.feature_extractor import extract_features, FEATURE_COLUMNS, SCHEMA_VERSION
from src.features.reference_labeling import align_reference, attach_candidate_coverage
from src.visualization.static_plots import plot_inventory, plot_raw_trip, plot_feature_summary
from src.visualization.folium_maps import raw_trip_map


def prepare_trip(trip, config=None, include_labels=True):
    network = RoadNetwork.from_tables(trip.nodes, trip.arcs)
    track, rejected, _ = clean_track(trip.track, network.projection)
    candidates = CandidateGenerator(network, config).generate(track)
    features = extract_features(track, candidates, network, trip.trip_id, trip.partition)
    labels = None
    if include_labels:
        if trip.route is None:
            raise ValueError("Offline labels requested without a supplied reference route")
        labels = attach_candidate_coverage(align_reference(track, network, trip.route.arc_id), candidates)
    return track, rejected, candidates, features, labels, network


def candidate_experiment(track, network, labels):
    """Coverage sensitivity on a training trip only; no evaluator data."""
    records = []
    for radius in (25.0, 50.0, 100.0):
        for limit in (4, 8, 12):
            start = time.perf_counter()
            config = CandidateConfig(radius_m=radius, max_candidates=limit)
            candidates = CandidateGenerator(network, config).generate(track)
            coverage = attach_candidate_coverage(labels, candidates)
            usable = coverage.loc[coverage.label_usable]
            records.append({"radius_m": radius, "max_candidates": limit,
                            "rows": len(candidates), "usable_proxy_observations": len(usable),
                            "candidate_miss_rate": float((~usable.candidate_hit).mean()) if len(usable) else None,
                            "fallback_observations": int(candidates.loc[candidates.is_fallback, "observation_id"].nunique()),
                            "elapsed_s": time.perf_counter() - start})
    return pd.DataFrame(records)


def run(raw_root, output_root, config=None, trip_limit=None):
    start = time.perf_counter()
    raw_root, output_root = Path(raw_root), Path(output_root)
    config = config or CandidateConfig()
    for local_directory in [raw_root, output_root / "data/processed", output_root / "data/features"]:
        local_directory.mkdir(parents=True, exist_ok=True)
        ignore = local_directory / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n!.gitignore\n", encoding="utf-8")
    eda = output_root / "results" / "member1"
    eda.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(raw_root)
    inventory.to_csv(eda / "dataset_inventory.csv", index=False)
    plot_inventory(inventory, eda / "dataset_overview.png")
    selected = inventory if trip_limit is None else inventory.groupby("partition", sort=False).head(trip_limit)
    summaries = []
    map_done = set()
    fingerprint = hashlib.sha256()
    for row in selected.itertuples(index=False):
        began = time.perf_counter()
        trip = load_trip(raw_root, row.partition, row.trip_id, include_reference=True)
        for ext in ("track", "nodes", "arcs", "route"):
            p = raw_root / row.partition / row.trip_id / f"{row.trip_id}.{ext}"
            fingerprint.update(f"{row.partition}/{p.name}".encode())
            fingerprint.update(p.read_bytes())
        track, rejected, candidates, features, labels, network = prepare_trip(trip, config)
        processed = output_root / "data" / "processed" / row.partition / row.trip_id
        feature_dir = output_root / "data" / "features" / row.partition
        processed.mkdir(parents=True, exist_ok=True)
        feature_dir.mkdir(parents=True, exist_ok=True)
        track.to_csv(processed / "observations.csv.gz", index=False)
        rejected.to_csv(processed / "rejected_observations.csv", index=False)
        candidates.to_csv(processed / "candidates.csv.gz", index=False)
        labels.to_csv(processed / "reference_labels.csv.gz", index=False)
        features.to_csv(feature_dir / f"{row.trip_id}.csv.gz", index=False)
        usable = labels.loc[labels.label_usable]
        summary = {"partition": row.partition, "trip_id": row.trip_id,
                   "raw_observations": len(trip.track), "clean_observations": len(track),
                   "rejected_observations": len(rejected), "candidate_rows": len(candidates),
                   "fallback_observations": int(candidates.loc[candidates.is_fallback, "observation_id"].nunique()),
                   "usable_proxy_labels": len(usable), "training_eligible_observations": int(labels.training_eligible.sum()),
                   "candidate_misses": int((~usable.candidate_hit).sum()),
                   "far_reference_observations": int((labels.label_status == "far_from_reference").sum()),
                   "ambiguous_observations": int((labels.label_status == "ambiguous_alignment").sum()),
                   "speed_outliers": int(track.speed_outlier.sum()),
                   "zero_length_arcs": int((network.lengths_m <= 1e-8).sum()),
                   "epsg": network.projection.epsg, "elapsed_s": time.perf_counter() - began}
        (processed / "metadata.json").write_text(json.dumps({**summary, "candidate_config": asdict(config),
            "schema_version": SCHEMA_VERSION, "reference_alignment": "monotone_route_proxy_v1"}, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(f"{row.partition}/{row.trip_id}: {len(track)} GPS, {len(candidates)} candidates, "
              f"{int(labels.training_eligible.sum())} eligible proxy observations", flush=True)
        if row.partition not in map_done:
            name = f"{row.partition}_{row.trip_id}"
            raw_trip_map(trip, eda / f"{name}_raw_map.html")
            plot_raw_trip(trip, track, network, eda / f"{name}_raw_network.png")
            plot_feature_summary(features, labels, eda / f"{name}_features.png")
            if row.partition == "train":
                candidate_experiment(track, network, labels).to_csv(eda / "candidate_sensitivity.csv", index=False)
            map_done.add(row.partition)
    summary = pd.DataFrame(summaries)
    summary.to_csv(eda / "processing_summary.csv", index=False)
    schema = {"version": SCHEMA_VERSION, "feature_columns": FEATURE_COLUMNS,
              "dtype": "float32", "identifier_columns": ["partition", "trip_id", "observation_id", "candidate_arc_id"],
              "target": "separate offline reference_labels.csv.gz join; not included in feature files",
              "candidate_config": asdict(config), "gap_threshold_s": 10.0,
              "label_max_distance_m": 50.0, "label_ambiguity_margin_m": 3.0, "label_heading_weight": 0.1}
    (eda / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    totals = summary.groupby("partition").sum(numeric_only=True).to_dict(orient="index")
    run_manifest = {"schema_version": SCHEMA_VERSION, "processed_trips": len(summary),
                    "inventory_trips": len(inventory), "full_dataset_run": len(summary) == len(inventory),
                    "dataset_sha256": fingerprint.hexdigest(), "candidate_config": asdict(config),
                    "python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("numpy", "pandas", "scipy", "pyproj", "shapely", "folium", "matplotlib")},
                    "elapsed_s": time.perf_counter() - start, "partition_totals": totals}
    (eda / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return inventory, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--trip-limit", type=int, help="Per-partition smoke run; inventory still checks every trip")
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()
    if args.trip_limit is not None and args.trip_limit < 1:
        parser.error("--trip-limit must be positive")
    if args.archive:
        print(f"Validated/extracted {extract_student_archive(args.archive, args.raw_root)} raw files", flush=True)
    run(args.raw_root, args.output_root, CandidateConfig(args.radius_m, args.max_candidates), args.trip_limit)


if __name__ == "__main__":
    main()
