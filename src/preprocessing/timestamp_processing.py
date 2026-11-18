"""Movement context from current and preceding observations only."""
import numpy as np
from .coordinate_utils import bearing_degrees


def add_motion_context(track, gap_threshold_s=10.0):
    out = track.copy()
    dt = out.timestamp_s.diff().to_numpy()
    dx, dy = out.x_m.diff().to_numpy(), out.y_m.diff().to_numpy()
    distance = np.hypot(dx, dy)
    gap = (~np.isfinite(dt)) | (dt <= 0) | (dt > gap_threshold_s)
    speed = np.divide(distance, dt, out=np.zeros(len(out)), where=(~gap))
    heading_valid = (~gap) & (distance >= 2.0)
    out["dt_s"] = np.where(np.isfinite(dt), dt, 0.0)
    out["step_distance_m"] = np.nan_to_num(distance)
    out["speed_mps"] = speed
    out["heading_deg"] = np.where(heading_valid, bearing_degrees(dx, dy), 0.0)
    out["heading_valid"] = heading_valid
    out["gap_before"] = gap
    out["sequence_id"] = np.cumsum(gap) - 1
    return out
