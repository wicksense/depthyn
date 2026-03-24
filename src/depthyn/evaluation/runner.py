"""Run detector evaluation against Ouster ground truth logs."""

from __future__ import annotations

import json
from pathlib import Path

from depthyn.config import DetectorConfig, ReplayConfig
from depthyn.detectors.factory import create_detector
from depthyn.evaluation.ground_truth import parse_ground_truth_log
from depthyn.evaluation.matching import FrameMatch, build_gt_index, match_frame
from depthyn.evaluation.metrics import EvaluationResult, compute_metrics
from depthyn.perception.background import BackgroundModel
from depthyn.pipeline import _iter_frames


def _detector_uses_foreground(config: ReplayConfig, detector) -> bool:
    return detector.input_mode == "foreground" or config.detector_on_foreground


def run_evaluation(
    config: ReplayConfig,
    gt_log_path: Path,
    *,
    max_distance_m: float = 3.0,
    min_gt_distance_m: float = 2.0,
    class_match: bool = True,
) -> dict[str, object]:
    """Run detector on pcap frames and evaluate against ground truth.

    For foreground-mode detectors (e.g. baseline clustering), runs a
    background model with time-based fade and object-aware freezing.

    Matches pcap frames to GT frames by sensor frame_id (the Ouster
    internal frame counter). Only frames with matching GT are evaluated.

    Returns a JSON-serializable evaluation report.
    """
    # Load ground truth
    gt_frames = parse_ground_truth_log(gt_log_path, min_distance_m=min_gt_distance_m)
    gt_index = build_gt_index(gt_frames)

    if not gt_frames:
        raise ValueError(f"No ground truth frames found in {gt_log_path}")

    gt_frame_counts = set(gt_index.keys())
    print(f"Loaded {len(gt_frames)} GT frames (frame_count {min(gt_frame_counts)}-{max(gt_frame_counts)})")

    # Load frames
    frames = _iter_frames(config)
    print(f"Loaded {len(frames)} replay frames")

    # Check for sensor_frame_id overlap
    pcap_frame_ids = {f.sensor_frame_id for f in frames if f.sensor_frame_id is not None}
    overlap = pcap_frame_ids & gt_frame_counts
    print(f"Pcap sensor frame_ids: {min(pcap_frame_ids) if pcap_frame_ids else 'N/A'}-{max(pcap_frame_ids) if pcap_frame_ids else 'N/A'}")
    print(f"Overlapping frames: {len(overlap)}")

    if not overlap:
        raise ValueError(
            "No overlapping frames between pcap and GT. "
            f"Pcap frame_ids: {min(pcap_frame_ids)}-{max(pcap_frame_ids)}, "
            f"GT frame_counts: {min(gt_frame_counts)}-{max(gt_frame_counts)}"
        )

    # Create detector
    detector = create_detector(config)
    print(f"Detector: {config.detector.resolved_label()} (input_mode={detector.input_mode})")

    # Set up background model for foreground-mode detectors
    detector_uses_foreground = _detector_uses_foreground(config, detector)

    bg_model: BackgroundModel | None = None
    if config.mode == "stationary" and detector_uses_foreground:
        bg_model = BackgroundModel(
            cell_size_m=config.cluster_cell_size_m,
            min_hits=config.background_min_hits,
            fade_time_s=config.background_fade_time_s,
        )
        print(f"Background model: fade_time={config.background_fade_time_s}s, min_hits={config.background_min_hits}")

    # Process all frames in order
    matches: list[FrameMatch] = []
    evaluated = 0
    warmup = config.background_warmup_frames
    for frame_index, frame in enumerate(frames):
        ts = frame.timestamp_ns

        # Determine working points
        if bg_model is not None:
            if frame_index < warmup:
                bg_model.observe(frame.points, timestamp_ns=ts)
                continue

            working_points = bg_model.filter_foreground(frame.points, timestamp_ns=ts)
        else:
            working_points = frame.points

        # Determine detector input
        detector_points = working_points if detector_uses_foreground else frame.points

        # Run detector
        result = detector.detect(frame, detector_points)
        detections = result.detections

        # Update background with object-aware freezing
        if bg_model is not None:
            protected = bg_model.protected_cells_from_detections(detections, margin_cells=2)
            bg_model.observe(frame.points, timestamp_ns=ts, protected_cells=protected)

        # Only evaluate frames that overlap with GT
        if frame.sensor_frame_id is None or frame.sensor_frame_id not in gt_index:
            continue

        gt_frame = gt_index[frame.sensor_frame_id]

        # Match detections to GT
        fm = match_frame(
            detections,
            gt_frame,
            max_distance_m=max_distance_m,
            class_match=class_match,
        )
        matches.append(fm)
        evaluated += 1

        tp = len(fm.true_positives)
        fp = len(fm.false_positives)
        fn = len(fm.false_negatives)
        fg_count = len(working_points) if bg_model else len(frame.points)
        if evaluated <= 5 or evaluated % 10 == 0:
            print(
                f"  Frame {frame.sensor_frame_id}: TP={tp} FP={fp} FN={fn} "
                f"(dets={tp+fp}, gt={tp+fn}, fg={fg_count})"
            )

    print(f"\nEvaluated {evaluated} frames")

    # Compute metrics
    metrics_result = compute_metrics(matches)

    # Build report
    report: dict[str, object] = {
        "project": "Depthyn",
        "pipeline": "evaluation",
        "config": {
            "detector": config.detector.resolved_label(),
            "gt_log": str(gt_log_path),
            "max_distance_m": max_distance_m,
            "min_gt_distance_m": min_gt_distance_m,
            "class_match": class_match,
            "mode": config.mode,
            "detector_on_foreground": config.detector_on_foreground,
            "voxel_size_m": config.voxel_size_m,
            "score_threshold": config.detector.score_threshold,
            "background_fade_time_s": config.background_fade_time_s,
            "cluster_cell_size_m": config.cluster_cell_size_m,
            "min_cluster_points": config.min_cluster_points,
        },
        "metrics": metrics_result.to_dict(),
    }

    return report
