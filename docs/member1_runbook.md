# Member 1: run and handoff

## Scope

Implemented: data parsing/inventory, conservative GPS preprocessing, metric
coordinates, spatial candidate generation, 23 inference features, separate offline
reference proxy alignment, five EDA notebooks, EDA plots/maps, and this documentation.
No classifier training, Bayesian optimization, learned HMM, API or UI is implemented
by this workstream. Existing files belonging to those workstreams remain untouched.

## Install and reproduce

From the repository root, using Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r docs/requirements-member1.txt
.\.venv\Scripts\python.exe -m src.data.prepare_member1 --archive "C:\path\to\Student folder.zip"
.\.venv\Scripts\python.exe -m pytest src/data/tests/test_member1.py -q --junitxml=results/member1/unit_tests.xml
.\.venv\Scripts\python.exe -m src.data.audit_member1
.\.venv\Scripts\python.exe -m src.data.reference_diagnostics
.\.venv\Scripts\python.exe -m src.data.execute_notebooks
.\.venv\Scripts\python.exe -m src.data.write_member1_report
```

If data is already extracted, omit `--archive`. Use `--trip-limit 1` for a quick
processing smoke run; note that the inventory still validates every trip. A full
run writes per-trip compressed files, aggregate evidence, maps and a run manifest.
Custom locations: `--raw-root <path>` and `--output-root <path>`. A run with a custom
output root keeps processed data under that root; notebooks use the default layout.

Five notebooks can be opened and run in order using the same Python environment.
They discover the repository root relative to the current directory and use
`PATHSNAP_RAW_ROOT` if set, otherwise `data/raw`. Run the full preparation command
before the notebooks. Their saved outputs contain actual development-data results.

## Outputs

| Location | Contents |
| --- | --- |
| `data/raw/` | Local development dataset, original bytes preserved |
| `data/processed/<partition>/<trip>/` | Clean GPS, rejected rows, candidates, reference proxy labels, metadata |
| `data/features/<partition>/<trip>.csv.gz` | IDs plus ordered inference features; no targets |
| `results/member1/dataset_inventory.csv` | Every development trip, raw data quality and topology |
| `results/member1/processing_summary.csv` | Per-trip processing, proxy-label and coverage counts |
| `results/member1/candidate_sensitivity.csv` | Radius/candidate-cap study on first training trip |
| `results/member1/feature_schema.json` | Ordered schema and exact processing settings |
| `results/member1/run_manifest.json` | Dataset fingerprint, dependency versions and run timing |
| `results/member1/*raw_map.html` | Interactive EDA maps with layer controls and legend |
| `results/member1/*.png` | Dataset, raw-network and feature figures |
| `docs/member1_report.md` | Completion report with actual results and limitations |

The raw/processed/features directories contain local `.gitignore` files to keep
large datasets out of ordinary Git adds. This does not untrack pre-existing files.
Source, docs, notebooks and compact EDA evidence may be committed when authorized.
Map tile backgrounds and Folium JavaScript/CSS depend on external web services;
static PNG figures are available offline. Road display thinning is visualization
only; candidate extraction always uses the complete road network.

## Important limitations

Reference alignment produces approximate labels, not independently timestamped
ground truth. See `feature_specification.md` before interpreting metrics. No
evaluator data or model performance claim is included. A model must not be selected
from EDA feature distributions alone. Member 2 must evaluate five models and tune
with trip-grouped splits; Member 3 must validate sequence reconstruction.

AI assistance: this implementation and documentation were prepared with Codex.
The team should review the code, reproduce the results and disclose assistance as
required by the event policy. The report records executed data processing rather
than invented model experiments.
