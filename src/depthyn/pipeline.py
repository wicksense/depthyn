from __future__ import annotations

import json
import statistics
from pathlib import Path

from depthyn.config import ReplayConfig
from depthyn.detectors.factory import create_detector
from depthyn.pose import discover_gps_csv, load_gps_pose_provider, transform_detection, transform_points
from depthyn.perception.background import BackgroundModel
from depthyn.rules import ZoneMonitor, load_zone_definitions
from depthyn.scene import build_scene_state
from depthyn.source.converted_csv import (
    discover_converted_csv_frames,
    load_converted_csv_frame,
)
from depthyn.tracking.simple import SimpleTracker


def _resolve_source_type(config: ReplayConfig) -> str:
    """Determine source type from config or auto-detect from input directory."""
    if config.source_type != "auto":
        return config.source_type
    input_dir = config.input_dir
    if any(input_dir.glob("*.pcap")):
        return "pcap"
    return "csv"


def _iter_frames(config: ReplayConfig) -> list:
    """Return a list of frames from the configured source."""
    source_type = _resolve_source_type(config)

    if source_type == "pcap":
        from depthyn.source.ouster_pcap import iter_ouster_pcap_frames

        return list(
            iter_ouster_pcap_frames(
                config.input_dir,
                voxel_size_m=config.voxel_size_m,
                min_range_m=config.min_range_m,
                max_range_m=config.max_range_m,
                z_min_m=config.z_min_m,
                z_max_m=config.z_max_m,
                max_frames=config.max_frames,
            )
        )

    # Default: CSV
    frame_paths = discover_converted_csv_frames(config.input_dir)
    if config.max_frames is not None:
        frame_paths = frame_paths[: config.max_frames]
    return [
        load_converted_csv_frame(
            path,
            voxel_size_m=config.voxel_size_m,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
            z_min_m=config.z_min_m,
            z_max_m=config.z_max_m,
        )
        for path in frame_paths
    ]


def _stream_frames(config: ReplayConfig):
    """Yield frames lazily from the configured source."""
    source_type = _resolve_source_type(config)

    if source_type == "pcap":
        from depthyn.source.ouster_pcap import iter_ouster_pcap_frames

        yield from iter_ouster_pcap_frames(
            config.input_dir,
            voxel_size_m=config.voxel_size_m,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
            z_min_m=config.z_min_m,
            z_max_m=config.z_max_m,
            max_frames=config.max_frames,
        )
        return

    frame_paths = discover_converted_csv_frames(config.input_dir)
    if config.max_frames is not None:
        frame_paths = frame_paths[: config.max_frames]
    for path in frame_paths:
        yield load_converted_csv_frame(
            path,
            voxel_size_m=config.voxel_size_m,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
            z_min_m=config.z_min_m,
            z_max_m=config.z_max_m,
        )


def _detector_uses_foreground(config: ReplayConfig, detector) -> bool:
    return detector.input_mode == "foreground" or config.detector_on_foreground


def run_replay(config: ReplayConfig) -> dict[str, object]:
    detector = create_detector(config)
    detector_uses_foreground = _detector_uses_foreground(config, detector)
    pose_provider = _load_pose_provider(config)
    zones = (
        load_zone_definitions(config.zone_config)
        if config.zone_config is not None
        else []
    )
    zone_monitor = ZoneMonitor(zones) if zones else None
    background_model = (
        BackgroundModel(
            cell_size_m=config.cluster_cell_size_m,
            min_hits=config.background_min_hits,
            fade_time_s=config.background_fade_time_s,
        )
        if config.mode == "stationary"
        and detector_uses_foreground
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
    label_counts: dict[str, int] = {}
    total_detection_score = 0.0
    scored_detection_count = 0
    timestamp_ns_values: list[int] = []
    all_zone_events: list[dict[str, object]] = []
    scene_min = [float("inf"), float("inf"), float("inf")]
    scene_max = [float("-inf"), float("-inf"), float("-inf")]

    frame_count = 0
    for frame_index, frame in enumerate(_stream_frames(config)):
        frame_count += 1
        timestamp_ns_values.append(frame.timestamp_ns)
        total_points += len(frame.points)
        stage = "tracking"
        working_points = frame.points
        frame_pose = pose_provider.pose_at(frame.timestamp_ns) if pose_provider else None

        if background_model is not None and frame_index < config.background_warmup_frames:
            background_model.observe(frame.points, timestamp_ns=frame.timestamp_ns)
            working_points = []
            detections = []
            detector_metadata = {"mode": "background_warmup"}
            active_tracks = tracker.update([], frame.timestamp_ns)
            stage = "background_warmup"
        else:
            if background_model is not None:
                working_points = background_model.filter_foreground(
                    frame.points, timestamp_ns=frame.timestamp_ns
                )
            detector_points = (
                working_points if detector_uses_foreground else frame.points
            )
            detector_result = detector.detect(frame, detector_points)
            detections = detector_result.detections
            if frame_pose is not None:
                detections = [
                    transform_detection(detection, frame_pose)
                    for detection in detections
                ]
            detector_metadata = detector_result.metadata
            active_tracks = tracker.update(detections, frame.timestamp_ns)

            # Continuous background observation with object-aware freezing
            if background_model is not None:
                protected = background_model.protected_cells_from_detections(
                    detections, margin_cells=2
                )
                background_model.observe(
                    frame.points,
                    timestamp_ns=frame.timestamp_ns,
                    protected_cells=protected,
                )
            total_detections += len(detections)
            for detection in detections:
                label_counts[detection.label] = label_counts.get(detection.label, 0) + 1
                if detection.score is not None:
                    total_detection_score += detection.score
                    scored_detection_count += 1

        zone_occupancy = []
        zone_events = []
        if zone_monitor is not None:
            raw_zone_occupancy, raw_zone_events = zone_monitor.evaluate(
                active_tracks, frame.timestamp_ns
            )
            zone_occupancy = [entry.to_dict() for entry in raw_zone_occupancy]
            zone_events = [event.to_dict() for event in raw_zone_events]
            all_zone_events.extend(zone_events)

        total_foreground_points += len(working_points)
        max_active_tracks = max(max_active_tracks, len(active_tracks))
        preview_source = working_points if working_points else frame.points
        if frame_pose is not None:
            preview_source = transform_points(preview_source, frame_pose)
        preview_points = _sample_preview_points(
            preview_source,
            config.preview_point_limit,
        )
        detail_points = _sample_preview_points(
            preview_source,
            config.detail_point_limit,
        )
        scanline_points = _sample_scanline_points(
            frame.scanline_points or [],
            config.detail_point_limit,
        )
        active_track_payload = [track.to_dict() for track in active_tracks]
        scene_state = build_scene_state(
            frame_index=frame_index,
            frame_id=frame.frame_id,
            timestamp_ns=frame.timestamp_ns,
            mode=config.mode,
            stage=stage,
            detector_name=config.detector.kind,
            tracks=active_tracks,
            zones=zone_occupancy,
            events=zone_events,
        )
        _expand_bounds(scene_min, scene_max, preview_points)
        for detection in detections:
            _expand_bounds(
                scene_min,
                scene_max,
                [detection.bbox_min, detection.bbox_max, detection.centroid],
            )
        for track in active_tracks:
            _expand_bounds(
                scene_min,
                scene_max,
                [track.bbox_min, track.bbox_max, track.centroid],
            )

        frame_summaries.append(
            {
                "frame_index": frame_index,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "source_path": str(frame.source_path),
                "frame_pose": None if frame_pose is None else frame_pose.to_dict(),
                "stage": stage,
                "points_after_filtering": len(frame.points),
                "foreground_points": len(working_points),
                "detection_count": len(detections),
                "preview_points": [list(point) for point in preview_points],
                "detail_points": [list(point) for point in detail_points],
                "scanline_shape": (
                    None if frame.scanline_shape is None else list(frame.scanline_shape)
                ),
                "scanline_points": [
                    [sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]]
                    for sample in scanline_points
                ],
                "detections": [detection.to_dict() for detection in detections],
                "active_tracks": active_track_payload,
                "scene_state": scene_state.to_dict(),
                "detector_input_points": len(
                    working_points if detector_uses_foreground else frame.points
                ),
                "detector_metadata": detector_metadata,
            }
        )

    tracks = tracker.all_tracks()
    summary: dict[str, object] = {
        "project": "Depthyn",
        "pipeline": "scene_replay",
        "reference_frame": "world" if pose_provider is not None else "sensor",
        "world_alignment": (
            None if pose_provider is None else pose_provider.metadata()
        ),
        "scene_contract": {
            "version": "v1",
            "object_source": "tracked_objects",
            "zone_shape": "axis_aligned_xy_rectangles",
        },
        "detector": config.detector.to_dict(),
        "config": config.to_dict(),
        "frames_processed": frame_count,
        "scene_bounds": _finalize_bounds(scene_min, scene_max),
        "zone_definitions": [zone.to_dict() for zone in zones],
        "playback": {
            "median_frame_interval_ms": _median_frame_interval_ms(timestamp_ns_values),
        },
        "metrics": {
            "avg_points_after_filtering": _safe_average(total_points, frame_count),
            "avg_foreground_points": _safe_average(
                total_foreground_points, frame_count
            ),
            "total_detections": total_detections,
            "total_tracks": len(tracks),
            "max_active_tracks": max_active_tracks,
            "total_zone_events": len(all_zone_events),
            "label_counts": dict(sorted(label_counts.items())),
            "avg_detection_score": (
                round(total_detection_score / scored_detection_count, 4)
                if scored_detection_count
                else None
            ),
            "detector_name": config.detector.kind,
        },
        "frame_summaries": frame_summaries,
        "event_summaries": all_zone_events,
        "track_summaries": [track.to_dict() for track in tracks],
    }
    return summary


def _load_pose_provider(config: ReplayConfig):
    if not config.world_align:
        return None
    gps_path = config.gps_path or discover_gps_csv(config.input_dir)
    return load_gps_pose_provider(gps_path)


def write_summary(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _safe_average(total: int, count: int) -> float:
    if count == 0:
        return 0.0
    return round(total / count, 2)


def _sample_preview_points(
    points: list[tuple[float, float, float]], limit: int
) -> list[tuple[float, float, float]]:
    if limit <= 0 or len(points) <= limit:
        return points
    step = max(1, len(points) // limit)
    sampled = points[::step]
    return sampled[:limit]


def _sample_scanline_points(
    points: list[tuple[int, int, float, float, float, float]], limit: int
) -> list[tuple[int, int, float, float, float, float]]:
    if limit <= 0 or len(points) <= limit:
        return points
    step = max(1, len(points) // limit)
    sampled = points[::step]
    return sampled[:limit]


def _expand_bounds(
    scene_min: list[float], scene_max: list[float], points: list[tuple[float, float, float]]
) -> None:
    for point in points:
        for index, value in enumerate(point):
            if value < scene_min[index]:
                scene_min[index] = value
            if value > scene_max[index]:
                scene_max[index] = value


def _finalize_bounds(scene_min: list[float], scene_max: list[float]) -> dict[str, list[float]]:
    if scene_min[0] == float("inf"):
        return {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    return {
        "min": [round(value, 3) for value in scene_min],
        "max": [round(value, 3) for value in scene_max],
    }


def _median_frame_interval_ms(timestamp_ns_values: list[int]) -> float:
    if len(timestamp_ns_values) < 2:
        return 100.0
    intervals_ms = [
        (later - earlier) / 1_000_000.0
        for earlier, later in zip(timestamp_ns_values, timestamp_ns_values[1:])
        if later >= earlier
    ]
    if not intervals_ms:
        return 100.0
    return round(statistics.median(intervals_ms), 2)
