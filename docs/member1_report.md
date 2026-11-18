# Member 1 completion report

Date: 18 November 2026. Scope: data, EDA, candidates, features and raw-network visualization.

## Result

Completed and locally verified the Member 1 pipeline for all 80 training trips and
10 student-test trips. Generated 1,238,485 candidate-feature rows
with 23 finite inference inputs. Saved reference proxy labels separately and
preserved the original observation, node and arc identifiers.

## Delivered implementation

| Area | Delivered work |
| --- | --- |
| `src/data/` | Strict raw readers, partition-safe loader, safe ZIP import, full inventory, reproducible processing CLI, full-data audit and report generation |
| `src/preprocessing/` | Conservative cleaning, original IDs, trip-local metric projection, causal movement context, gap resets and speed flags |
| `src/candidates/` | Indexed exact segment-radius search, stable ranking, bounded candidate sets and flagged nearest fallback |
| `src/features/` | 23-feature inference schema, graph/motion features, separate monotone reference-proxy alignment and safe per-trip training join |
| `src/visualization/` | Dataset/feature plots, raw GPS/network/reference figures and Folium maps |
| `notebooks/` | Five runnable notebooks with saved executed outputs: inventory, GPS EDA, road network, candidate sensitivity and feature analysis |
| `docs/` | Data dictionary, full feature specification, runbook, EDA findings and this report |

## Verification evidence

- 24 focused tests passed; failures/errors: 0. Evidence: `results/member1/unit_tests.xml`.
- All 90 generated trip tables passed the full-data audit, covering
  1,238,485 candidate rows. Evidence: `full_data_verification.json`.
- All five notebooks executed every code cell with zero errors. Evidence: `notebook_verification.json`.
- Checks include malformed files, metre-based projection, long-road geometry, zero-length arcs,
  timestamp order, GPS gaps, empty candidate sets, reference-free inference schema,
  uncertain labels, missing reference candidates, cross-trip join rejection and evaluator exclusion.
- Dataset processing took 189.0 seconds in this environment, excluding
  dependency installation, notebook execution and later audits. This is not model training time.
- Data fingerprint: `e0cac7c31dadbecfd52f9f519682989864be58c7b0377839de73e054ef7f7ff8`. Dependency versions are in `run_manifest.json`.

## Actual data findings

| partition | clean_observations | candidate_rows | usable_proxy_labels | candidate_misses | training_eligible_observations | proxy_miss_rate_percent | mean_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| student_test | 19160 | 113648 | 14472 | 15 | 14457 | 0.104 | 5.932 |
| train | 208417 | 1124837 | 145069 | 726 | 144343 | 0.5 | 5.397 |

The full EDA narrative, data-quality tables, reference-alignment diagnostics,
candidate sensitivity results and figures are in [eda_findings.md](eda_findings.md).

## Limitations and team handoff

The `.route` files are chronological arc sequences, not per-observation labels.
The implemented labels are uncertain proxies, with explicit exclusions. Monotone
alignment permits forward skips and has no speed penalty; repeated routes and
disconnected references can still create errors. The report separately measures
distance to any reference arc and distance after alignment. No classification,
HMM or final-evaluation accuracy is claimed by this work.

Member 2 should use only `FEATURE_COLUMNS`, join labels within each trip, exclude
ineligible observations, group CV by trip, and fit preprocessing inside folds.
Member 3 should reuse the identical inference extractor and candidate policy;
reference labels must stay out of prediction and UI responses. The integration
team can add Member 1 dependencies from `docs/requirements-member1.txt` to the
project-wide environment and incorporate these EDA sections into its consolidated report.

Existing model-training, optimization, evaluation, HMM, API and UI source files
were not implemented or overwritten. The shared `report/final_report.pdf` placeholder
is not presented as a completed full-project report. No Git commit or push was made.

## Reproduce

See [member1_runbook.md](member1_runbook.md) for installation and the full processing
command, [data_dictionary.md](data_dictionary.md) for file semantics, and
[feature_specification.md](feature_specification.md) for schema and integration examples.

AI assistance: Codex assisted with implementation, documentation and execution.
All numerical findings above come from executed development-data processing.
The team should review, reproduce and understand the implementation before submission.
