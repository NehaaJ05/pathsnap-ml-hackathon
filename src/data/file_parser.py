"""Strict parsers. Row positions are the authoritative node/arc identifiers."""
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMAS = {
    ".track": ["longitude", "latitude", "timestamp_s"],
    ".nodes": ["longitude", "latitude"],
    ".arcs": ["from_node_id", "to_node_id"],
    ".route": ["arc_id"],
}


def parse_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix not in SCHEMAS:
        raise ValueError(f"Unsupported raw file: {path}")
    try:
        values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric table: {path}") from exc
    columns = SCHEMAS[path.suffix]
    if values.size == 0 or values.shape[1] != len(columns):
        raise ValueError(f"{path}: expected nonempty {len(columns)}-column table")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: nonfinite value")
    if path.suffix in {".arcs", ".route"}:
        if (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"{path}: indices must be nonnegative integers")
        values = values.astype(np.int64)
    if path.suffix == ".nodes":
        if (np.abs(values[:, 0]) > 180).any() or (np.abs(values[:, 1]) > 90).any():
            raise ValueError(f"{path}: invalid node coordinates")
    return pd.DataFrame(values, columns=columns)
