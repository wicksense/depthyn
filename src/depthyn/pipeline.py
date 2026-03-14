from __future__ import annotations

import json
from pathlib import Path

from depthyn.config import ReplayConfig
from depthyn.perception.background import BackgroundModel
from depthyn.perception.clustering import cluster_points
from depthyn.source.converted_csv import (
    discover_converted_csv_frames,
    load_converted_csv_frame,
)
from depthyn.tracking.simple import SimpleTracker


def run_replay(config: ReplayConfig) -> dict[str, object]:
    frame_paths = discover_converted_csv_frames(config.input_dir)
    if config.max_frames is not None:
        frame_paths = frame_paths[: config.max_frames]

    background_model = (
        BackgroundModel(
            cell_size_m=config.cluster_cell_size_m,
            min_hits=config.background_min_hits,
        )
        if config.mode == "stationary"
        else None
    )
    tracker = SimpleTracker(
        max_distance_m=config.track_max_distance_m,
        max_missed_frames=config.track_max_missed_frames,
    )

    frame_summaries: list[dict[str, object]] = []
    total_points = 0
    total_foreground_points = 0
    total_detections = 0
    max_active_tracks = 0

    for frame_index, frame_path in enumerate(frame_paths):
        frame = load_converted_csv_frame(
            frame_path,
            voxel_size_m=config.voxel_size_m,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
            z_min_m=config.z_min_m,
            z_max_m=config.z_max_m,
        )
        total_points += len(frame.points)
        stage = "tracking"
        working_points = frame.points

        if background_model is not None and frame_index < config.background_warmup_frames:
            background_model.observe(frame.points)
            working_points = []
            detections = []
            active_tracks = tracker.update([], frame.timestamp_ns)
            stage = "background_warmup"
        else:
            if background_model is not None:
                working_points = background_model.filter_foreground(frame.points)
            detections = cluster_points(
                working_points,
                cell_size_m=config.cluster_cell_size_m,
                min_cluster_points=config.min_cluster_points,
                min_cluster_cells=config.min_cluster_cells,
                min_cluster_height_m=config.min_cluster_height_m,
                max_cluster_height_m=config.max_cluster_height_m,
                max_cluster_width_m=config.max_cluster_width_m,
            )
            active_tracks = tracker.update(detections, frame.timestamp_ns)
            total_detections += len(detections)

        total_foreground_points += len(working_points)
        max_active_tracks = max(max_active_tracks, len(active_tracks))

        frame_summaries.append(
            {
                "frame_index": frame_index,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "source_path": str(frame.source_path),
                "stage": stage,
                "points_after_filtering": len(frame.points),
                "foreground_points": len(working_points),
                "detection_count": len(detections),
                "detections": [detection.to_dict() for detection in detections],
                "active_track_ids": [track.track_id for track in active_tracks],
            }
        )

    tracks = tracker.all_tracks()
    summary: dict[str, object] = {
        "project": "Depthyn",
        "pipeline": "baseline_replay",
        "config": config.to_dict(),
        "frames_processed": len(frame_summaries),
        "metrics": {
            "avg_points_after_filtering": _safe_average(total_points, len(frame_summaries)),
            "avg_foreground_points": _safe_average(
                total_foreground_points, len(frame_summaries)
            ),
            "total_detections": total_detections,
            "total_tracks": len(tracks),
            "max_active_tracks": max_active_tracks,
        },
        "frame_summaries": frame_summaries,
        "track_summaries": [track.to_dict() for track in tracks],
    }
    return summary


def write_summary(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _safe_average(total: int, count: int) -> float:
    if count == 0:
        return 0.0
    return round(total / count, 2)

