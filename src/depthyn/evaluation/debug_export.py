"""Debug export helpers for visually inspecting GT vs detections."""

from __future__ import annotations

from pathlib import Path

from depthyn.config import DetectorConfig, ReplayConfig
from depthyn.detectors.factory import create_detector
from depthyn.evaluation.ground_truth import parse_ground_truth_log
from depthyn.models import Detection
from depthyn.pipeline import _iter_frames


def export_debug_frame(
    input_dir: Path,
    gt_log_path: Path,
    output_path: Path,
    *,
    frame_count: int,
    detector: DetectorConfig,
    source_type: str = "pcap",
    voxel_size_m: float = 0.3,
    detector_on_foreground: bool = False,
) -> dict[str, object]:
    """Export one overlapping frame as a viewer-friendly debug bundle."""
    config = ReplayConfig(
        input_dir=input_dir,
        output_json=output_path,
        source_type=source_type,
        detector=detector,
        voxel_size_m=voxel_size_m,
        detector_on_foreground=detector_on_foreground,
        preview_point_limit=20000,
    )

    gt_frames = parse_ground_truth_log(gt_log_path, min_distance_m=0.0)
    gt_index = {frame.frame_count: frame for frame in gt_frames}
    gt_frame = gt_index.get(frame_count)
    if gt_frame is None:
        raise ValueError(f"Frame count {frame_count} not found in GT log: {gt_log_path}")

    frames = _iter_frames(config)
    frame = next((item for item in frames if item.sensor_frame_id == frame_count), None)
    if frame is None:
        raise ValueError(f"Frame count {frame_count} not found in replay source: {input_dir}")

    detector_impl = create_detector(config)
    result = detector_impl.detect(frame, frame.points)

    gt_detections = [_gt_object_to_detection(index + 1, obj) for index, obj in enumerate(gt_frame.objects)]
    bundle = {
        "project": "Depthyn",
        "pipeline": "debug_frame_overlay",
        "config": {
            "frame_count": frame_count,
            "detector": detector.to_dict(),
            "source_type": source_type,
            "detector_on_foreground": detector_on_foreground,
        },
        "frames_processed": 1,
        "scene_bounds": _compute_bounds(frame.points, result.detections, gt_detections),
        "playback": {
            "median_frame_interval_ms": 0.0,
        },
        "metrics": {
            "detector_name": detector.kind,
            "gt_object_count": len(gt_detections),
            "detector_detection_count": len(result.detections),
        },
        "frame_summaries": [
            {
                "frame_index": 0,
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "source_path": str(frame.source_path),
                "stage": "debug_overlay",
                "points_after_filtering": len(frame.points),
                "foreground_points": len(frame.points),
                "detection_count": len(result.detections),
                "preview_points": [list(point) for point in frame.points[: config.preview_point_limit]],
                "detections": [detection.to_dict() for detection in result.detections],
                "active_tracks": [detection.to_dict() for detection in gt_detections],
                "scene_state": {
                    "frame_index": 0,
                    "frame_id": frame.frame_id,
                    "timestamp_ns": frame.timestamp_ns,
                    "mode": "debug",
                    "stage": "debug_overlay",
                    "detector_name": detector.kind,
                    "tracked_objects": [],
                    "zones": [],
                    "events": [],
                },
                "detector_input_points": len(frame.points),
                "detector_metadata": {
                    **result.metadata,
                    "gt_frame_count": frame_count,
                    "gt_timestamp_us": gt_frame.timestamp_us,
                    "gt_reference_in_active_tracks": True,
                },
            }
        ],
        "event_summaries": [],
        "track_summaries": [detection.to_dict() for detection in gt_detections],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(__import__("json").dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def _gt_object_to_detection(index: int, obj) -> Detection:
    length, width, height = obj.dimensions
    cx, cy, cz = obj.position
    return Detection(
        detection_id=f"gt-{obj.gt_id}-{index:03d}",
        centroid=(cx, cy, cz),
        bbox_min=(cx - length / 2, cy - width / 2, cz - height / 2),
        bbox_max=(cx + length / 2, cy + width / 2, cz + height / 2),
        point_count=0,
        cell_count=0,
        label=f"gt-{obj.mapped_label}",
        score=obj.confidence,
        source="gemini-gt",
        heading_rad=obj.heading,
    )


def _compute_bounds(points, detections, gt_detections):
    coords = [*points]
    for detection in [*detections, *gt_detections]:
        coords.extend([detection.centroid, detection.bbox_min, detection.bbox_max])
    if not coords:
        return {"min_xyz": [0.0, 0.0, 0.0], "max_xyz": [0.0, 0.0, 0.0]}
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    zs = [point[2] for point in coords]
    return {
        "min_xyz": [min(xs), min(ys), min(zs)],
        "max_xyz": [max(xs), max(ys), max(zs)],
    }
