# Member 1 feature specification

Schema version: `member1-v1`. Authoritative ordering:
`src.features.feature_extractor.FEATURE_COLUMNS`. All model inputs are emitted as
float32, finite, with identical extraction code for training and inference.
Exported schema and processing settings: `results/member1/feature_schema.json`.

## Task definition

One row represents a GPS observation / directed-road-segment candidate pair. The
binary development target is 1 for the aligned reference proxy arc and 0 for other
candidates, only when that observation is `training_eligible`. A model ranks the
candidates per observation. Arc IDs are trip-local metadata, not global classes.

## Geometry and candidate policy

WGS84 longitude/latitude is projected to the median road-network coordinate's UTM
zone (`always_xy=True`). Candidates are all nonzero-length segments within 50 m,
ranked by exact point-to-segment distance and then original arc ID, capped at 8.
A Shapely STRtree indexes segment extents, so long arcs are not missed by a
midpoint-only search. When none are inside the radius, nearest nonzero-length
segments are returned and flagged; the same 8-candidate cap and tie order apply.
This fallback is not guaranteed to be a plausible match, and its distance remains
available to the classifier. No reference arc is injected into candidates.

Defaults are an initial reproducible operating point, not a claim of optimality.
The 25/50/100 m and 4/8/12-candidate sensitivity table uses a training trip only.
All-trip coverage is also measured. Member 2 may tune the policy using allowed
training/student-test data and must regenerate consistently afterwards.

## Feature definitions

| Feature | Units / interpretation | Rationale |
| --- | --- | --- |
| `distance_m` | Exact GPS-to-segment distance, m | Spatial fit |
| `fraction_along` | Projection fraction in [0,1] | Position on arc |
| `segment_length_m` | Endpoint-to-endpoint length, m | Segment scale |
| `distance_to_start_m` | Fraction × segment length, m | Remaining/reached geometry context |
| `distance_to_end_m` | (1-fraction) × length, m | Endpoint context |
| `log1p_distance_m` | Log-transformed distance | Reduce influence of extreme fallback distances |
| `heading_difference_deg` | Smallest directed angle, 0–180; 0 when unavailable | Motion/road direction agreement |
| `heading_cosine` | Cosine of angle; 0 when unavailable | Smooth directional agreement |
| `heading_valid` | 0/1 | Distinguish missing direction from perfect agreement |
| `dt_s` | Previous retained time interval; first row 0 | Sampling context |
| `step_distance_m` | Previous GPS displacement; first row 0 | Motion magnitude, including gap displacement |
| `speed_mps` | Displacement/time; 0 at sequence starts | Motion context |
| `gap_before` | 0/1; first observation or gap >10 s | Reset sequential assumptions |
| `speed_outlier` | 0/1; speed >70 m/s | Flag suspicious motion without deleting it |
| `candidate_rank` | 1-based distance order | Relative spatial preference |
| `candidate_count` | Number retained for observation | Local ambiguity |
| `is_fallback` | 0/1 | Search radius contained no road |
| `start_in_degree` | Incoming arc count at candidate start | Directed topology |
| `end_out_degree` | Outgoing arc count at candidate end | Directed topology |
| `previous_nearest_available` | 0/1; false at gaps/missing previous candidates | Availability of local history |
| `same_as_previous_nearest` | Candidate equals previous observation's distance-rank-1 arc | Local persistence |
| `connected_from_previous_nearest` | Previous nearest arc's end equals candidate start | Direct road adjacency |
| `same_weak_component_as_previous` | Candidate start and previous nearest end share a weak component | Coarse graph connectivity |

Connectivity history is based solely on previous **geometric candidates**, never
reference labels or model predictions. Weak connectivity does not imply a legal
directed route; shortest-path transition modelling belongs to Member 3.

## Reference proxy alignment and uncertainty

The offline labeler aligns observations to the ordered reference arc occurrences.
At observation i and reference occurrence j, local cost is projected distance
plus `0.1 × heading_difference_deg` when heading is valid. Dynamic programming
minimizes cumulative local cost with a nondecreasing occurrence index. Any starting
occurrence and forward skips are permitted; alignment restarts after GPS gaps.
This is reference supervision preparation, not the learned HMM decoder.

A usable label must be within 50 m, have at least a 3-unit local-cost margin over
every alternative distinct reference arc, and not be a speed outlier. Repeated
occurrences of the same arc are equivalent for classification. Cost margins mix
distance with scaled angle and are heuristic scores, not calibrated confidence.
Intersecting segments, reverse arcs without reliable heading, GPS noise and a
monotonicity constraint can produce ambiguous or inaccurate proxies. The gates
reduce uncertainty but do not prove correctness. Forward skips are unpenalized;
there is no speed-consistency or timestamped reference guarantee.

Report candidate miss rate **among usable reference proxy observations**, and
report excluded-label counts alongside it. This rate is not independently verified
ground-truth recall. Alignment backpointers have a 512 MiB per-sequence guard.

## Member 2 / Member 3 handoff

```python
from pathlib import Path
import pandas as pd
from src.features.feature_extractor import FEATURE_COLUMNS
from src.features.reference_labeling import build_training_table

trip_id = "00000001"
features = pd.read_csv(Path("data/features/train") / f"{trip_id}.csv.gz",
                       dtype={"trip_id": str})
labels = pd.read_csv(Path("data/processed/train") / trip_id / "reference_labels.csv.gz")
table = build_training_table(features, labels)
eligible = table.loc[table.training_eligible]
X = eligible[FEATURE_COLUMNS]
y = eligible.target.astype(int)
groups = eligible.trip_id
```

Concatenate per-trip tables after this join. Never use all numeric columns as X:
IDs, reference fields, `target`, and eligibility flags must not be model features.
Fit learned preprocessing inside trip-grouped training folds and save it with the
classifier. Student-test trips stay separate from fitting. For inference:

```python
from src.data.dataset_loader import load_trip
from src.data.prepare_member1 import prepare_trip
trip = load_trip("data/raw", "student_test", "00000000", include_reference=False)
track, rejected, candidates, features, labels, network = prepare_trip(trip, include_labels=False)
assert labels is None
X = features[FEATURE_COLUMNS]
```

`prepare_trip` also accepts a Trip object with `partition="inference"` constructed
from API-supplied tables, without using the development dataset loader. Keep a
consistent candidate config and feature schema in the deployed artifacts. The HMM
must retain `(observation_id, candidate_arc_id)` correspondence and handle fallback
and gap flags. Target probability is a match score; normalization/calibration and
emission interpretation are the decoder team's responsibility.
