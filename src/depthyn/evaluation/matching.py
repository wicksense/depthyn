"""Match detections to ground truth objects spatially."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from depthyn.evaluation.ground_truth import (
    CENTERPOINT_CLASS_MAP,
    GroundTruthFrame,
    GroundTruthObject,
)
from depthyn.models import Detection


@dataclass(slots=True)
class FrameMatch:
    """Result of matching one frame's detections against ground truth."""

    frame_count: int
    true_positives: list[tuple[Detection, GroundTruthObject]]
    false_positives: list[Detection]
    false_negatives: list[GroundTruthObject]


def _distance_xy(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def match_frame(
    detections: list[Detection],
    gt_frame: GroundTruthFrame,
    *,
    max_distance_m: float = 3.0,
    class_match: bool = True,
) -> FrameMatch:
    """Match detections to GT objects using greedy nearest-centroid.

    Args:
        detections: CenterPoint detections for this frame.
        gt_frame: Ground truth frame to match against.
        max_distance_m: Maximum XY distance for a valid match.
        class_match: If True, TP requires matching mapped labels.
    """
    # Build candidate pairs (distance, det_idx, gt_idx)
    candidates: list[tuple[float, int, int]] = []
    for di, det in enumerate(detections):
        det_label = CENTERPOINT_CLASS_MAP.get(det.label, det.label)
        for gi, gt_obj in enumerate(gt_frame.objects):
            dist = _distance_xy(det.centroid, gt_obj.position)
            if dist <= max_distance_m:
                if class_match and det_label != gt_obj.mapped_label:
                    continue
                candidates.append((dist, di, gi))

    # Greedy assignment by ascending distance
    candidates.sort(key=lambda x: x[0])
    used_dets: set[int] = set()
    used_gts: set[int] = set()
    tp_pairs: list[tuple[Detection, GroundTruthObject]] = []

    for _, di, gi in candidates:
        if di in used_dets or gi in used_gts:
            continue
        tp_pairs.append((detections[di], gt_frame.objects[gi]))
        used_dets.add(di)
        used_gts.add(gi)

    fp = [det for i, det in enumerate(detections) if i not in used_dets]
    fn = [gt for i, gt in enumerate(gt_frame.objects) if i not in used_gts]

    return FrameMatch(
        frame_count=gt_frame.frame_count,
        true_positives=tp_pairs,
        false_positives=fp,
        false_negatives=fn,
    )


def build_gt_index(
    gt_frames: list[GroundTruthFrame],
) -> dict[int, GroundTruthFrame]:
    """Build a frame_count → GroundTruthFrame lookup."""
    return {f.frame_count: f for f in gt_frames}
