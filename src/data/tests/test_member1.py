"""Member 1 invariants: geometry, leakage boundaries, IDs and proxy-label limits."""
import zipfile
import numpy as np
import pandas as pd
import pytest
from src.data.file_parser import parse_file
from src.data.dataset_loader import extract_student_archive, load_trip
from src.preprocessing.coordinate_utils import LocalProjection, angular_difference
from src.preprocessing.gps_cleaning import clean_track
from src.candidates.candidate_utils import RoadNetwork, project_to_segments
from src.candidates.candidate_generator import CandidateGenerator, CandidateConfig
from src.features.feature_extractor import extract_features, FEATURE_COLUMNS
from src.features.reference_labeling import align_reference, attach_candidate_coverage, build_training_table


@pytest.fixture
def road():
    nodes = pd.DataFrame({"longitude": [8., 8.001, 8.002, 8.001],
                          "latitude": [50., 50., 50., 50.001]})
    arcs = pd.DataFrame({"from_node_id": [0, 1, 1, 1, 2], "to_node_id": [1, 2, 0, 3, 2]})
    return RoadNetwork.from_tables(nodes, arcs)


def observation_track(road, longitude=(8.0002, 8.0005, 8.0015), latitude=None, timestamps=None):
    n = len(longitude)
    track = pd.DataFrame({"longitude": longitude, "latitude": latitude or [50.] * n,
                          "timestamp_s": timestamps or list(range(n))})
    return clean_track(track, road.projection)[0]


@pytest.mark.parametrize("suffix,text", [(".track", "8 50\n"), (".arcs", "0 1.5\n"),
                                        (".route", "-1\n"), (".nodes", "181 50\n"),
                                        (".track", "8 50 nan\n")])
def test_parser_rejects_malformed(tmp_path, suffix, text):
    path = tmp_path / ("trip" + suffix)
    path.write_text(text)
    with pytest.raises(ValueError):
        parse_file(path)


def test_parser_keeps_single_row_shape(tmp_path):
    path = tmp_path / "one.route"
    path.write_text("2\n")
    assert parse_file(path).shape == (1, 1)


def test_projection_is_metric_and_round_trips():
    projection = LocalProjection.from_coordinates([8., 8.001], [50., 50.])
    xy = projection.project([8., 8.001], [50., 50.])
    assert 70 < np.linalg.norm(xy[1] - xy[0]) < 73
    lon, lat = projection.inverse.transform(xy[:, 0], xy[:, 1])
    np.testing.assert_allclose(lon, [8., 8.001], atol=1e-9)
    np.testing.assert_allclose(lat, [50., 50.], atol=1e-9)


def test_angular_wrap():
    assert angular_difference(359, 1) == 2


def test_exact_segment_projection_handles_long_and_zero_arcs():
    distance, fraction, snapped = project_to_segments(np.array([[500., 2.], [2., 3.]]),
                                                     np.array([[0., 0.], [0., 0.]]),
                                                     np.array([[1000., 0.], [0., 0.]]))
    np.testing.assert_allclose(distance, [2., np.sqrt(13)])
    np.testing.assert_allclose(fraction, [.5, 0])


def test_cleaning_preserves_original_ids_and_does_not_sort(road):
    raw = pd.DataFrame({"longitude": [8.] * 5, "latitude": [50.] * 5,
                        "timestamp_s": [1., 1., 0., 2., 3.]})
    track, rejected, _ = clean_track(raw, road.projection)
    assert track.observation_id.tolist() == [0, 3, 4]
    assert len(rejected) == 2


def test_gap_resets_motion_and_connectivity(road):
    track = observation_track(road, timestamps=[0., 1., 100.])
    assert track.sequence_id.tolist() == [0, 0, 1]
    assert track.speed_mps.iloc[-1] == 0
    candidates = CandidateGenerator(road).generate(track)
    features = extract_features(track, candidates, road, "trip", "train")
    assert features.loc[features.observation_id == 2, "previous_nearest_available"].eq(0).all()


def test_candidates_have_exact_distances_bounded_count_and_stable_ids(road):
    track = observation_track(road)
    candidates = CandidateGenerator(road, CandidateConfig(max_candidates=2)).generate(track)
    assert candidates.groupby("observation_id").size().max() <= 2
    assert candidates.candidate_arc_id.isin([0, 1, 2, 3]).all()  # zero-length arc 4 excluded
    assert not candidates.duplicated(["observation_id", "candidate_arc_id"]).any()
    assert candidates.distance_m.max() <= 50


def test_nearest_fallback_is_flagged_and_optional(road):
    track = observation_track(road, longitude=(8.01,), latitude=[50.01])
    fallback = CandidateGenerator(road, CandidateConfig(radius_m=1)).generate(track)
    assert len(fallback) and fallback.is_fallback.all() and fallback.distance_m.min() > 1
    empty = CandidateGenerator(road, CandidateConfig(radius_m=1, fallback_nearest=False)).generate(track)
    assert empty.empty


def test_inference_schema_has_no_reference_or_target(road):
    track = observation_track(road)
    candidates = CandidateGenerator(road).generate(track)
    features = extract_features(track, candidates, road, "00000001", "train")
    assert not any("reference" in name or "target" in name for name in features.columns)
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()
    assert list(features.columns[4:]) == FEATURE_COLUMNS
    assert features[FEATURE_COLUMNS].dtypes.eq(np.dtype("float32")).all()


def test_monotone_alignment_and_separate_training_join(road):
    track = observation_track(road)
    candidates = CandidateGenerator(road).generate(track)
    features = extract_features(track, candidates, road, "trip", "train")
    labels = align_reference(track, road, [0, 1])
    assert labels.route_position.is_monotonic_increasing
    assert labels.reference_arc_id.tolist() == [0, 0, 1]
    labels = attach_candidate_coverage(labels, candidates)
    merged = build_training_table(features, labels)
    positives = merged.loc[merged.training_eligible].groupby("observation_id").target.sum()
    assert positives.eq(1).all()
    assert "target" not in features


def test_missing_positive_is_reported_not_injected(road):
    track = observation_track(road)
    candidates = CandidateGenerator(road).generate(track)
    labels = align_reference(track, road, [0, 1])
    candidates = candidates.loc[~((candidates.observation_id == 2) & (candidates.candidate_arc_id == 1))]
    covered = attach_candidate_coverage(labels, candidates)
    assert not covered.loc[covered.observation_id == 2, "candidate_hit"].item()
    assert not covered.loc[covered.observation_id == 2, "training_eligible"].item()


def test_reference_ambiguity_and_distance_are_flagged(road):
    track = observation_track(road, longitude=(8.0005,))
    labels = align_reference(track, road, [0, 2])
    assert not labels.label_usable.any()  # coincident reverse arcs without movement
    track = observation_track(road, longitude=(8.1,), latitude=[50.1])
    labels = align_reference(track, road, [0])
    assert labels.label_status.item() == "far_from_reference"


def test_reference_alignment_memory_guard(road):
    with pytest.raises(MemoryError):
        align_reference(observation_track(road), road, [0, 1], max_backpointer_mb=0)


def test_archive_rejects_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.track", "8 50 0")
    with pytest.raises(ValueError):
        extract_student_archive(archive, tmp_path / "raw")
    assert not (tmp_path / "outside.track").exists()


def test_inference_load_does_not_require_route(tmp_path):
    folder = tmp_path / "train" / "00000001"
    folder.mkdir(parents=True)
    for suffix, content in {"track": "8 50 0\n", "nodes": "8 50\n8.001 50\n", "arcs": "0 1\n"}.items():
        (folder / f"00000001.{suffix}").write_text(content)
    trip = load_trip(tmp_path, "train", "00000001")
    assert trip.route is None
    with pytest.raises(FileNotFoundError):
        load_trip(tmp_path, "train", "00000001", include_reference=True)


def test_evaluator_partition_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_trip(tmp_path, "evaluator_test", "00000001")


def test_long_segment_is_found_even_when_midpoint_is_far():
    nodes = pd.DataFrame({"longitude": [8., 8.02], "latitude": [50., 50.]})
    arcs = pd.DataFrame({"from_node_id": [0], "to_node_id": [1]})
    network = RoadNetwork.from_tables(nodes, arcs)
    track = observation_track(network, longitude=(8.001,), latitude=[50.00005])
    candidates = CandidateGenerator(network, CandidateConfig(radius_m=20, fallback_nearest=False)).generate(track)
    assert candidates.candidate_arc_id.tolist() == [0]
    assert candidates.distance_m.item() < 10


def test_empty_candidates_produce_empty_feature_schema(road):
    track = observation_track(road, longitude=(8.1,), latitude=[50.1])
    candidates = CandidateGenerator(road, CandidateConfig(radius_m=1, fallback_nearest=False)).generate(track)
    features = extract_features(track, candidates, road, "trip", "train")
    assert features.empty
    assert features.columns.tolist()[4:] == FEATURE_COLUMNS


def test_pooled_trip_label_join_is_rejected(road):
    track = observation_track(road)
    candidates = CandidateGenerator(road).generate(track)
    features = extract_features(track, candidates, road, "trip", "train")
    other = features.assign(trip_id="other")
    labels = attach_candidate_coverage(align_reference(track, road, [0, 1]), candidates)
    with pytest.raises(ValueError, match="one partition/trip"):
        build_training_table(pd.concat([features, other]), labels)
