"""Audit all generated features/labels and save machine-readable evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.features.feature_extractor import FEATURE_COLUMNS, IDENTIFIER_COLUMNS
from src.features.reference_labeling import build_training_table


def audit(root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    evidence = root / "results/member1"
    summary = pd.read_csv(evidence / "processing_summary.csv", dtype={"trip_id": str})
    manifest = json.loads((evidence / "run_manifest.json").read_text())
    if not manifest["full_dataset_run"]:
        raise ValueError("Final verification requires a full dataset run")
    records = []
    for row in summary.itertuples(index=False):
        features = pd.read_csv(root / "data/features" / row.partition / f"{row.trip_id}.csv.gz", dtype={"trip_id": str})
        processed = root / "data/processed" / row.partition / row.trip_id
        candidates = pd.read_csv(processed / "candidates.csv.gz")
        labels = pd.read_csv(processed / "reference_labels.csv.gz")
        observations = pd.read_csv(processed / "observations.csv.gz")
        assert features.columns.tolist() == IDENTIFIER_COLUMNS + FEATURE_COLUMNS
        assert np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()
        assert len(features) == len(candidates) == row.candidate_rows
        assert len(labels) == len(observations) == row.clean_observations
        assert features.trip_id.eq(row.trip_id).all() and features.partition.eq(row.partition).all()
        assert not features.duplicated(["observation_id", "candidate_arc_id"]).any()
        pd.testing.assert_frame_equal(features[["observation_id", "candidate_arc_id"]],
                                      candidates[["observation_id", "candidate_arc_id"]], check_dtype=False)
        assert candidates.groupby("observation_id").size().max() <= manifest["candidate_config"]["max_candidates"]
        assert labels.training_eligible.equals(labels.label_usable & labels.candidate_hit)
        table = build_training_table(features, labels)
        eligible = table.loc[table.training_eligible]
        assert eligible.groupby("observation_id").target.sum().eq(1).all()
        assert table.loc[~table.training_eligible, "target"].isna().all()
        assert len(observations.observation_id.unique()) == len(observations)
        assert observations.timestamp_s.is_monotonic_increasing
        with_sequence = labels.merge(observations[["observation_id", "sequence_id"]], on="observation_id", validate="one_to_one")
        assert with_sequence.groupby("sequence_id").route_position.apply(lambda x: x.is_monotonic_increasing).all()
        records.append({"partition": row.partition, "trip_id": row.trip_id, "feature_rows": len(features),
                        "eligible_observations": int(labels.training_eligible.sum()),
                        "training_rows": len(eligible), "positive_rows": int(eligible.target.sum()),
                        "checks_passed": True})
    result = {"all_passed": True, "trips_verified": len(records), "feature_count": len(FEATURE_COLUMNS),
              "candidate_rows_verified": sum(r["feature_rows"] for r in records), "per_trip": records}
    (evidence / "full_data_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_trip"}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    audit()
