"""Conservative cleaning: preserve raw observation IDs; never silently reorder."""
import numpy as np
import pandas as pd
from .coordinate_utils import LocalProjection
from .timestamp_processing import add_motion_context


def clean_track(track, projection=None, gap_threshold_s=10.0):
    out = track.copy()
    out.insert(0, "observation_id", np.arange(len(out), dtype=np.int64))
    valid = (np.isfinite(out[["longitude", "latitude", "timestamp_s"]]).all(axis=1)
             & out.longitude.between(-180, 180) & out.latitude.between(-90, 90))
    rejected = out.loc[~valid].copy()
    out = out.loc[valid].copy()
    # Keep the first observation at a timestamp; reject duplicates and backwards
    # timestamps relative to the maximum retained time. Do not sort a trajectory.
    if len(out):
        t = out.timestamp_s.to_numpy()
        keep = np.r_[True, t[1:] > np.maximum.accumulate(t)[:-1]]
        rejected = pd.concat([rejected, out.loc[~keep]])
        out = out.loc[keep].reset_index(drop=True)
    if not len(out):
        raise ValueError("No valid GPS observations remain")
    projection = projection or LocalProjection.from_coordinates(out.longitude, out.latitude)
    out[["x_m", "y_m"]] = projection.project(out.longitude.to_numpy(), out.latitude.to_numpy())
    out = add_motion_context(out, gap_threshold_s)
    # Retain and flag implausible motion rather than inventing replacement points.
    out["speed_outlier"] = out.speed_mps > 70.0
    return out, rejected, projection
