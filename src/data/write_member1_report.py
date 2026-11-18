"""Create Member 1's Markdown findings and a standalone illustrated HTML report."""
import base64
from datetime import datetime, timedelta, timezone
import html
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def table(frame):
    headers = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([headers, rule, *rows])


def markdown_html(text):
    """Render the Markdown subset used by these generated reports."""
    def inline(value):
        value = html.escape(value)
        value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
        value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
        return re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
    result, paragraph, table_rows, bullets = [], [], [], []
    def flush():
        if paragraph:
            result.append("<p>" + inline(" ".join(paragraph)) + "</p>")
            paragraph.clear()
        if table_rows:
            header = table_rows[0]
            body = [r for r in table_rows[1:] if not all(re.fullmatch(r"[: -]+", c) for c in r)]
            result.append('<div style="overflow-x:auto"><table><thead><tr>' + ''.join('<th>'+inline(c)+'</th>' for c in header)
                          + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>' for row in body) + '</tbody></table></div>')
            table_rows.clear()
        if bullets:
            result.append('<ul>'+''.join('<li>'+inline(b)+'</li>' for b in bullets)+'</ul>')
            bullets.clear()
    for line in text.splitlines():
        if line.startswith("|"):
            if paragraph or bullets:
                flush()
            table_rows.append([c.strip() for c in line.strip().strip('|').split('|')])
        elif line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip('#'))
            result.append(f'<h{level}>{inline(line[level:].strip())}</h{level}>')
        elif line.startswith("- "):
            if paragraph or table_rows:
                flush()
            bullets.append(line[2:])
        elif not line.strip():
            flush()
        elif bullets and line.startswith("  "):
            bullets[-1] += ' ' + line.strip()
        else:
            if table_rows or bullets:
                flush()
            paragraph.append(line.strip())
    flush()
    return '\n'.join(result)


def write_report(root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    evidence = root / "results/member1"
    inventory = pd.read_csv(evidence / "dataset_inventory.csv", dtype={"trip_id": str})
    summary = pd.read_csv(evidence / "processing_summary.csv", dtype={"trip_id": str})
    diagnostics = pd.read_csv(evidence / "reference_alignment_diagnostics.csv", dtype={"trip_id": str})
    manifest = json.loads((evidence / "run_manifest.json").read_text())
    audit = json.loads((evidence / "full_data_verification.json").read_text())
    notebooks = json.loads((evidence / "notebook_verification.json").read_text())
    suites = ET.parse(evidence / "unit_tests.xml").getroot().findall("testsuite")
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    assert manifest["full_dataset_run"] and audit["all_passed"] and failures == 0
    assert len(notebooks) == 5 and all(n["errors"] == 0 and n["code_cells"] == n["executed_cells"] for n in notebooks)
    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %B %Y")
    totals = summary.groupby("partition").sum(numeric_only=True)
    raw = inventory.groupby("partition").agg(trips=("trip_id", "size"), gps=("gps_observations", "sum"),
                                             nodes=("nodes", "sum"), arcs=("arcs", "sum"), reference_arcs=("route_arcs", "sum")).reset_index()
    coverage = totals[["clean_observations", "candidate_rows", "usable_proxy_labels", "candidate_misses", "training_eligible_observations"]].copy()
    coverage["proxy_miss_rate_percent"] = (100 * coverage.candidate_misses / coverage.usable_proxy_labels).round(3)
    coverage["mean_candidates"] = (coverage.candidate_rows / coverage.clean_observations).round(3)
    coverage = coverage.reset_index()
    quality_columns = ["missing_values", "invalid_coordinates", "duplicate_timestamps", "backward_timestamps",
                       "gaps_over_10s", "self_loops", "duplicate_arcs", "reference_route_breaks"]
    quality = inventory.groupby("partition")[quality_columns].sum().reset_index()
    label_counts = totals[["usable_proxy_labels", "far_reference_observations", "ambiguous_observations", "speed_outliers"]].reset_index()
    diag = diagnostics.groupby("partition")[["nearest_reference_over_50m", "aligned_reference_over_50m", "additional_far_after_alignment"]].sum().reset_index()
    step_fraction = 100 * inventory.one_second_steps.sum() / inventory.time_steps.sum()
    far_trips = diagnostics.nlargest(5, "nearest_reference_over_50m")
    sensitivity = pd.read_csv(evidence / "candidate_sensitivity.csv")
    sensitivity["candidate_miss_rate"] = (100 * sensitivity.candidate_miss_rate).round(3)
    sensitivity = sensitivity.rename(columns={"candidate_miss_rate": "proxy_miss_percent"})
    raw_paths = [p for p in (root / "data/raw").rglob("*") if p.suffix in {".track", ".nodes", ".arcs", ".route"}]
    findings = f'''# Dataset understanding and EDA findings

Generated {date} from a full development-data run. No evaluator data was used.

## Dataset size

{table(raw)}

All {len(raw_paths)} raw files were parsed and validated. Their combined size is
{sum(p.stat().st_size for p in raw_paths):,} bytes. Node/arc totals sum trip-local networks;
they must not be interpreted as globally unique roads. There are {int(inventory.gps_observations.sum()):,}
GPS observations. The largest trip has {int(inventory.gps_observations.max()):,} observations;
the largest network has {int(inventory.arcs.max()):,} directed arcs.

## Data quality and topology

{table(quality)}

{step_fraction:.3f}% of adjacent timestamp differences are exactly one second.
The largest observed gap is {inventory.max_gap_s.max():,.0f} seconds. Such gaps must
reset motion/decoder assumptions. No observations were silently reordered or
interpolated. {int(totals.rejected_observations.sum()):,} observations were rejected by preprocessing.

The trip coordinates span longitude {inventory.min_longitude.min():.6f} to
{inventory.max_longitude.max():.6f}, latitude {inventory.min_latitude.min():.6f} to
{inventory.max_latitude.max():.6f}. A single local CRS would be inappropriate;
per-trip UTM projections are used. Weak connectivity varies by trip, and directed
reachability cannot be inferred from weak-component membership alone.

Duplicate arcs and self-loops retain original IDs. Zero-length arcs are excluded
from candidates. {int(inventory.reference_route_breaks.sum())} consecutive reference-arc pairs
are disconnected by directed endpoint identity; no artificial connecting arcs are created.

## Candidate features and reference coverage

Default policy: exact segment distance, radius 50 m, maximum 8 candidates, nearest
fallback flagged when the radius contains none. All 23 model inputs are reference-free.

{table(coverage)}

Candidate miss rate is computed only among usable **approximate reference proxy
labels**, not independently timestamped ground truth. Reference information is
never injected into candidates or inference features. Unusable labels and misses
are excluded from the eligible candidate-classifier training join.

{table(label_counts)}

## Reference alignment diagnostic

Monotone route alignment can increase distance relative to independently choosing
the nearest reference arc. The following independent STRtree check quantifies both:

{table(diag)}

Trips with the largest number of observations over 50 m from **any** supplied reference arc:

{table(far_trips[["partition", "trip_id", "observations", "nearest_reference_over_50m", "additional_far_after_alignment"]])}

These observations cannot be made reliable by merely assigning a row-wise route
label. The discrepancy may reflect reference coverage, GPS quality or reference
limitations; these data alone do not establish its cause. Additional alignment
distance is an algorithmic limitation, reported separately rather than attributed
entirely to the raw data.

## Training-trip candidate sensitivity

The study below uses the first training trip only. It is a reproducible sensitivity
check, not a claim that the selected defaults are optimal for all trips.

{table(sensitivity[["radius_m", "max_candidates", "rows", "usable_proxy_observations", "proxy_miss_percent", "fallback_observations"]])}

## Visual evidence

![Dataset inventory](../results/member1/dataset_overview.png)

![Training-trip GPS and network](../results/member1/train_00000001_raw_network.png)

![Training-trip feature distributions](../results/member1/train_00000001_features.png)

Interactive maps: [training trip](../results/member1/train_00000001_raw_map.html)
and [student-test trip](../results/member1/student_test_00000000_raw_map.html).
Each has GPS, road-network and reference-route layers, a legend and layer controls.
These maps are development EDA, not the evaluator-facing UI.

## Implications for the next workstreams

Use trip-grouped cross-validation; retain the student-test partition for development
comparison. Fit scalers only on training folds. Evaluate observation-level ranking
and candidate coverage alongside candidate-level classification metrics. Retain
the candidate-to-arc mapping for the learned decoder, reset at gaps, and handle
disconnected road transitions explicitly. Report proxy-label exclusions and avoid
claiming pointwise ground-truth accuracy without explaining the alignment policy.
'''
    (root / "docs/eda_findings.md").write_text(findings, encoding="utf-8")
    completion = f'''# Member 1 completion report

Date: {date}. Scope: data, EDA, candidates, features and raw-network visualization.

## Result

Completed and locally verified the Member 1 pipeline for all 80 training trips and
10 student-test trips. Generated {int(totals.candidate_rows.sum()):,} candidate-feature rows
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

- {tests} focused tests passed; failures/errors: {failures}. Evidence: `results/member1/unit_tests.xml`.
- All {audit['trips_verified']} generated trip tables passed the full-data audit, covering
  {audit['candidate_rows_verified']:,} candidate rows. Evidence: `full_data_verification.json`.
- All five notebooks executed every code cell with zero errors. Evidence: `notebook_verification.json`.
- Checks include malformed files, metre-based projection, long-road geometry, zero-length arcs,
  timestamp order, GPS gaps, empty candidate sets, reference-free inference schema,
  uncertain labels, missing reference candidates, cross-trip join rejection and evaluator exclusion.
- Dataset processing took {manifest['elapsed_s']:.1f} seconds in this environment, excluding
  dependency installation, notebook execution and later audits. This is not model training time.
- Data fingerprint: `{manifest['dataset_sha256']}`. Dependency versions are in `run_manifest.json`.

## Actual data findings

{table(coverage)}

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
'''
    (root / "docs/member1_report.md").write_text(completion, encoding="utf-8")
    figures = []
    for filename, caption in [("dataset_overview.png", "Full dataset inventory"),
                              ("train_00000001_raw_network.png", "Training trip: GPS, roads and supplied reference"),
                              ("train_00000001_features.png", "Training-trip feature distributions")]:
        encoded = base64.b64encode((evidence / filename).read_bytes()).decode()
        figures.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    html_report = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PathSnap | Member 1 report</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#193247;background:#edf2f6;margin:0}main{max-width:1120px;margin:32px auto;padding:36px;background:white;border-radius:12px}h1,h2{color:#126680}h1{font-size:34px}table{border-collapse:collapse;width:100%;font-size:13px;margin:24px 0}th,td{border-bottom:1px solid #d7e1e8;text-align:left;padding:10px}th{background:#eaf4f7}img{max-width:100%;height:auto}figure{margin:32px 0}figcaption{color:#546b7b}.badge{background:#e0f2e8;padding:12px;border-radius:6px}.source{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 system-ui}a{color:#126680}@media print{body{background:white}main{margin:0;padding:0}figure{break-inside:avoid}}
</style></head><body><main><h1>PathSnap: Member 1 completion report</h1>'''
    html_report += f'<p>{html.escape(date)} · Data, EDA, candidates and features</p><p class="badge">90 trips processed · {audit["candidate_rows_verified"]:,} candidate rows · 23 features · {tests} tests passed · 5 notebooks executed</p>'
    html_report += '<h2>Data coverage</h2>' + coverage.to_html(index=False, border=0)
    html_report += markdown_html(completion)
    html_report += markdown_html(findings.split('## Visual evidence')[0])
    html_report += ''.join(figures)
    html_report += '<p>Interactive EDA maps: <a href="../results/member1/train_00000001_raw_map.html">Training trip</a> · <a href="../results/member1/student_test_00000000_raw_map.html">Student-test trip</a></p></main></body></html>'
    (root / "docs/member1_report.html").write_text(html_report, encoding="utf-8")
    print("Created docs/eda_findings.md, docs/member1_report.md and docs/member1_report.html", flush=True)


if __name__ == "__main__":
    write_report()
