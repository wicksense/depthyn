from __future__ import annotations

import json
import struct
from pathlib import Path

from depthyn.source.converted_csv import (
    discover_converted_csv_frames,
    load_converted_csv_frame,
)


def export_ml_replay_bundle(
    *,
    input_dir: Path,
    output_dir: Path,
    max_frames: int | None,
    voxel_size_m: float,
    min_range_m: float,
    max_range_m: float,
    z_min_m: float,
    z_max_m: float,
    default_intensity: float = 0.0,
) -> dict[str, object]:
    frame_paths = discover_converted_csv_frames(input_dir)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_manifest: list[dict[str, object]] = []

    for frame_path in frame_paths:
        frame = load_converted_csv_frame(
            frame_path,
            voxel_size_m=voxel_size_m,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
        )
        point_path = frames_dir / f"{frame.frame_id}.bin"
        _write_xyzi_bin(point_path, frame.points, default_intensity)
        frame_manifest.append(
            {
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "source_path": str(frame.source_path),
                "points_path": str(point_path.relative_to(output_dir)),
                "point_count": len(frame.points),
                "feature_count": 4,
            }
        )

    manifest = {
        "project": "Depthyn",
        "export_type": "ml_replay_bundle",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "frame_count": len(frame_manifest),
        "point_format": "xyzi_float32_le",
        "default_intensity": default_intensity,
        "frames": frame_manifest,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _write_xyzi_bin(
    path: Path,
    points: list[tuple[float, float, float]],
    default_intensity: float,
) -> None:
    with path.open("wb") as handle:
        for x, y, z in points:
            handle.write(
                struct.pack(
                    "<ffff",
                    float(x),
                    float(y),
                    float(z),
                    float(default_intensity),
                )
            )
