# Member 1 data dictionary

## Raw inputs

Each trip directory contains four headerless whitespace-separated tables. IDs are
local to a trip; never pool node/arc IDs as global model classes.

| File | Columns | Meaning |
| --- | --- | --- |
| `<trip_id>.track` | `longitude`, `latitude`, `timestamp_s` | WGS84 degrees and seconds; row index is original `observation_id` |
| `<trip_id>.nodes` | `longitude`, `latitude` | WGS84 node coordinates; zero-based row index is `node_id` |
| `<trip_id>.arcs` | `from_node_id`, `to_node_id` | Directed endpoints; zero-based row index is `arc_id` |
| `<trip_id>.route` | `arc_id` | Chronological reference arc occurrences; repeated IDs are possible; these are NOT per-observation labels |

The development archive has `train` and `student_test`. The expected counts are 80
and 10 trips. The code rejects other partitions. Evaluator trips are not provided
in this archive and are excluded from all Member 1 tools.

## Processed observations

Location: `data/processed/<partition>/<trip_id>/observations.csv.gz`.

`observation_id` retains the original raw row number. `x_m`, `y_m` use the trip
road network's median-coordinate UTM zone, recorded as `epsg` in metadata. All
distance features use that same projection. `dt_s`, `step_distance_m`, `speed_mps`
refer to the preceding retained observation. `heading_deg` is clockwise from grid
north. `heading_valid` requires at least 2 metres of motion and a valid time step.
`gap_before` is true for the first observation and after a gap over 10 seconds.
`sequence_id` starts at zero and increments at these gaps. Gap-start speed is zero
and heading is marked unavailable. `speed_outlier` flags speeds over 70 m/s.

Invalid coordinates and nonincreasing timestamps are rejected, not sorted or
interpolated. `rejected_observations.csv` preserves their raw IDs. Non-numeric,
nonfinite, empty or malformed raw tables fail validation rather than silently
changing their shape. Large finite GPS jumps are retained and flagged.

## Candidate geometry

Location: `data/processed/<partition>/<trip_id>/candidates.csv.gz`.

`observation_id`, `candidate_arc_id` identify each candidate. `distance_m` is the
exact projected point-to-segment distance, `fraction_along` is clipped to [0,1],
and `snapped_x_m`, `snapped_y_m` are projection coordinates. `candidate_rank` starts
at 1, sorting by distance then arc ID. `is_fallback` identifies nearest-segment
fallbacks outside the radius. Zero-length arcs are counted but excluded from
candidate generation; arc IDs are never renumbered.

## Reference-derived development labels

Location: `data/processed/<partition>/<trip_id>/reference_labels.csv.gz`.

| Column | Meaning |
| --- | --- |
| `observation_id` | Original GPS row ID |
| `reference_arc_id` | Arc selected by monotone reference-route alignment |
| `route_position` | Occurrence index in the supplied `.route` sequence |
| `reference_distance_m` | Distance to the selected reference segment |
| `local_cost_margin_m` | Alternative distinct arc cost minus selected cost; may be infinite if no alternative exists |
| `label_usable` | Passes distance, ambiguity and speed-outlier gates |
| `label_status` | `usable_proxy`, `far_from_reference`, `ambiguous_alignment`, or `speed_outlier` |
| `candidate_hit` | Reference proxy arc exists in the unmodified candidate set |
| `training_eligible` | `label_usable AND candidate_hit` |

These are **approximate proxy labels**, because reference arcs have no observation
timestamps. A label with `label_usable=false` is not a negative example. Candidate
misses are reported and never repaired by inserting the reference arc.

## Feature tables and join keys

`data/features/<partition>/<trip_id>.csv.gz` contains four metadata columns plus
the 23 inputs defined in `feature_specification.md`. Read `trip_id` as a string to
preserve leading zeros. Group CV by trip, not by candidate row or observation.
Reference labels can be joined on `observation_id` only within the matching
partition/trip file. Pooling before a join requires the complete
`(partition, trip_id, observation_id)` key.

Source material: supplied `Dataset Exploration.pdf` and the development archive.
Actual measured findings are in `eda_findings.md` and `member1_report.md`.
