"""Quick sanity check on one trip: candidate coverage, label distances, arc direction."""
import sys, time
import numpy as np

sys.path.insert(0, "/home/claude/work/pathsnap/src")
from pathsnap.data import load_trip
from pathsnap.candidates import build_candidate_table

DATA = "/home/claude/work/student_folder/Student folder"
trip_id = sys.argv[1] if len(sys.argv) > 1 else "00000010"
t0 = time.time()
trip = load_trip(DATA, "train", trip_id)
print("loaded", trip_id, "points", len(trip.xy), "arcs", len(trip.arc_from), "route", len(trip.route), "%.1fs" % (time.time() - t0))

t0 = time.time()
cand = build_candidate_table(trip)
print("candidate rows", len(cand), "per point", round(len(cand) / len(trip.xy), 2), "%.1fs" % (time.time() - t0))

pts = cand.groupby("point").agg(has_pos=("label", "max"))
print("points with reference arc in candidates: %.2f%%" % (100 * pts["has_pos"].mean()))
print("reference distance (m) quantiles:", np.percentile(cand.drop_duplicates("point")["ref_dist"], [50, 90, 99]).round(1))

# are arcs two-way? (reverse arc exists)
pairs = set(zip(trip.arc_from.tolist(), trip.arc_to.tolist()))
rev = sum((b, a) in pairs for a, b in pairs)
print("arcs with a reverse twin: %.1f%%" % (100 * rev / len(pairs)))
print("duplicate arcs:", len(pairs) != len(trip.arc_from))
