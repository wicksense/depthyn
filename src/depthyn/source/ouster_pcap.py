"""Ouster pcap source adapter.

Reads raw Ouster LiDAR pcap files using the Ouster SDK and yields
Frame objects compatible with the Depthyn replay pipeline.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from pathlib import Path

from depthyn.models import Frame, Point3D


def discover_ouster_pcap_files(input_dir: Path) -> list[Path]:
    """Find all .pcap files in *input_dir*, sorted by name."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    pcaps = sorted(p for p in input_dir.glob("*.pcap") if p.is_file())
    if not pcaps:
        raise FileNotFoundError(f"No .pcap files found under: {input_dir}")
    return pcaps


def find_metadata_json(pcap_path: Path) -> Path:
    """Locate the Ouster sensor metadata JSON for a pcap file.

    Tries several naming conventions and fallbacks including glob
    patterns and configuration directories.
    """
    parent = pcap_path.parent
    stem = pcap_path.stem

    # Strip chunk suffix for multi-chunk pcaps (e.g. _chunk0001 -> base stem)
    base_stem = stem
    if "_chunk" in stem:
        base_stem = stem[: stem.index("_chunk")]

    # Strip duplicate suffix like " (1)" from copied files
    clean_stem = re.sub(r"\s*\(\d+\)$", "", stem)
    if clean_stem != stem:
        base_stem = clean_stem

    candidates = [
        parent / f"{stem}_0.json",
        parent / f"{base_stem}_0.json",
        parent / f"{stem}.json",
        parent / f"{base_stem}.json",
        parent / "metadata.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Glob for <stem>_<serial>.json
    for prefix in (stem, base_stem):
        glob_matches = sorted(parent.glob(f"{prefix}_*.json"))
        if glob_matches:
            return glob_matches[0]

    # Check inside <stem>_configuration/ directory
    for prefix in (stem, base_stem):
        config_dir = parent / f"{prefix}_configuration"
        if config_dir.is_dir():
            json_files = sorted(config_dir.glob("*.json"))
            if json_files:
                return json_files[0]

    raise FileNotFoundError(
        f"Cannot find Ouster metadata JSON for {pcap_path}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def iter_ouster_pcap_frames(
    input_dir: Path,
    *,
    voxel_size_m: float = 0.3,
    min_range_m: float = 1.0,
    max_range_m: float = 60.0,
    z_min_m: float = -2.5,
    z_max_m: float = 4.5,
    max_frames: int | None = None,
) -> Iterator[Frame]:
    """Iterate over LiDAR frames from Ouster pcap files.

    Yields one :class:`Frame` per complete LiDAR revolution. Points are
    range/z-filtered and voxel-downsampled, matching the converted CSV
    adapter contract.
    """
    try:
        import numpy as np
        from ouster.sdk import open_source
        from ouster.sdk.core import XYZLut
    except ImportError as exc:
        raise ImportError(
            "Ouster pcap source requires 'ouster-sdk' and 'numpy'. "
            "Install with: pip install ouster-sdk numpy"
        ) from exc

    pcap_files = discover_ouster_pcap_files(input_dir)
    frame_count = 0

    for pcap_path in pcap_files:
        if max_frames is not None and frame_count >= max_frames:
            break

        try:
            metadata_path = find_metadata_json(pcap_path)
        except FileNotFoundError:
            continue
        source = open_source(str(pcap_path), meta=[str(metadata_path)])
        sensor_info = source.sensor_info[0]
        xyz_lut = XYZLut(sensor_info)

        for scan_set in source:
            if max_frames is not None and frame_count >= max_frames:
                break

            for scan in scan_set.valid_scans():
                if max_frames is not None and frame_count >= max_frames:
                    break

                xyz = xyz_lut(scan)  # (H, W, 3)
                rng = scan.field("RANGE")  # (H, W) uint32

                # Flatten
                xyz_flat = xyz.reshape(-1, 3)
                rng_flat = rng.ravel()
                valid_mask = rng_flat > 0

                xyz_valid = xyz_flat[valid_mask]
                if len(xyz_valid) == 0:
                    continue

                # Apply range and Z filtering
                x = xyz_valid[:, 0]
                y = xyz_valid[:, 1]
                z = xyz_valid[:, 2]
                radius = np.sqrt(x * x + y * y + z * z)

                keep = (
                    (radius >= min_range_m)
                    & (radius <= max_range_m)
                    & (z >= z_min_m)
                    & (z <= z_max_m)
                )
                filtered = xyz_valid[keep]

                if len(filtered) == 0:
                    continue

                # Voxel downsampling
                if voxel_size_m > 0:
                    voxel_keys = np.floor(filtered / voxel_size_m).astype(np.int32)
                    _, unique_idx = np.unique(
                        voxel_keys, axis=0, return_index=True
                    )
                    filtered = filtered[unique_idx]

                # Build point list
                points: list[Point3D] = [
                    (float(row[0]), float(row[1]), float(row[2]))
                    for row in filtered
                ]

                # Timestamp: median of per-column timestamps
                ts_array = scan.timestamp
                valid_ts = ts_array[ts_array > 0]
                if len(valid_ts) > 0:
                    timestamp_ns = int(np.median(valid_ts))
                else:
                    timestamp_ns = 0

                sensor_fid = scan.frame_id if hasattr(scan, "frame_id") else None
                frame_id = f"pcap_{pcap_path.stem}_f{frame_count:06d}"

                yield Frame(
                    frame_id=frame_id,
                    timestamp_ns=timestamp_ns,
                    points=points,
                    source_path=pcap_path,
                    sensor_frame_id=sensor_fid,
                )
                frame_count += 1

        source.close()
