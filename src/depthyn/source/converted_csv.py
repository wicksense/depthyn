from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from depthyn.models import Frame, Point3D


def discover_converted_csv_frames(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    frames = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if not frames:
        raise FileNotFoundError(f"No CSV frames found under: {input_dir}")
    return frames


def load_converted_csv_frame(
    path: Path,
    *,
    voxel_size_m: float,
    min_range_m: float,
    max_range_m: float,
    z_min_m: float,
    z_max_m: float,
) -> Frame:
    points_by_voxel: dict[tuple[int, int, int], Point3D] = {}
    min_ts: int | None = None
    max_ts: int | None = None

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        ts_idx, x_idx, y_idx, z_idx = _resolve_columns(header)

        for row in reader:
            if not row:
                continue

            timestamp_ns = int(float(row[ts_idx]))
            x = float(row[x_idx])
            y = float(row[y_idx])
            z = float(row[z_idx])

            radius_m = math.sqrt(x * x + y * y + z * z)
            if radius_m < min_range_m or radius_m > max_range_m:
                continue
            if z < z_min_m or z > z_max_m:
                continue

            if min_ts is None or timestamp_ns < min_ts:
                min_ts = timestamp_ns
            if max_ts is None or timestamp_ns > max_ts:
                max_ts = timestamp_ns

            if voxel_size_m > 0:
                key = (
                    math.floor(x / voxel_size_m),
                    math.floor(y / voxel_size_m),
                    math.floor(z / voxel_size_m),
                )
                points_by_voxel.setdefault(key, (x, y, z))
            else:
                points_by_voxel[(len(points_by_voxel), 0, 0)] = (x, y, z)

    points = list(points_by_voxel.values())
    timestamp_ns = ((min_ts or 0) + (max_ts or 0)) // 2
    return Frame(
        frame_id=path.stem,
        timestamp_ns=timestamp_ns,
        points=points,
        source_path=path,
    )


def _resolve_columns(header: list[str]) -> tuple[int, int, int, int]:
    normalized = [_normalize_column(name) for name in header]

    def find_index(candidates: tuple[str, ...], fallback: int) -> int:
        for index, name in enumerate(normalized):
            if any(name.startswith(candidate) for candidate in candidates):
                return index
        return fallback

    ts_idx = find_index(("timestamp", "time"), 0)
    x_idx = find_index(("x1", "x", "xglobal"), 1)
    y_idx = find_index(("y1", "y", "yglobal"), 2)
    z_idx = find_index(("z1", "z", "zglobal"), 3)
    return ts_idx, x_idx, y_idx, z_idx


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())

