"""Load a trip without opening its reference route unless explicitly requested."""
from dataclasses import dataclass
from pathlib import Path
import zipfile
import pandas as pd
from .file_parser import parse_file

DEVELOPMENT_PARTITIONS = ("train", "student_test")


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame | None = None


def list_trips(raw_root: str | Path, partition: str) -> list[str]:
    if partition not in DEVELOPMENT_PARTITIONS:
        raise ValueError("Member 1 tools only accept train and student_test")
    root = Path(raw_root) / partition
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_trip(raw_root: str | Path, partition: str, trip_id: str,
              include_reference: bool = False) -> Trip:
    if partition not in DEVELOPMENT_PARTITIONS:
        raise ValueError("Evaluator data is excluded from development")
    if not trip_id or Path(trip_id).name != trip_id or trip_id in {".", ".."}:
        raise ValueError("trip_id must be a single directory name")
    folder = Path(raw_root) / partition / trip_id
    tables = {ext: parse_file(folder / f"{trip_id}.{ext}")
              for ext in ("track", "nodes", "arcs")}
    if tables["arcs"].to_numpy().max() >= len(tables["nodes"]):
        raise ValueError(f"{trip_id}: arc references nonexistent node")
    route = None
    if include_reference:
        route = parse_file(folder / f"{trip_id}.route")
        if route.arc_id.max() >= len(tables["arcs"]):
            raise ValueError(f"{trip_id}: route references nonexistent arc")
    return Trip(trip_id, partition, tables["track"], tables["nodes"], tables["arcs"], route)


def extract_student_archive(archive: str | Path, raw_root: str | Path) -> int:
    """Copy only the four known file types under the two development partitions.

    No extractall: archive paths cannot escape the destination. Existing different
    files are rejected rather than silently replacing a local dataset.
    """
    root = Path(raw_root).resolve()
    copied = 0
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            parts = item.filename.split("/")
            if item.is_dir():
                continue
            if (len(parts) != 4 or parts[0] != "Student folder"
                    or parts[1] not in DEVELOPMENT_PARTITIONS
                    or not parts[2].isdigit()
                    or parts[3] not in {parts[2] + ext for ext in
                                       (".track", ".nodes", ".arcs", ".route")}):
                raise ValueError(f"Unexpected archive member: {item.filename}")
            target = root.joinpath(*parts[1:])
            content = z.read(item)
            if target.exists() and target.read_bytes() != content:
                raise FileExistsError(f"Existing dataset differs: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(content)
            copied += 1
    return copied
