"""Compute evaluation metrics from frame matches."""

from __future__ import annotations

from dataclasses import dataclass, field

from depthyn.evaluation.ground_truth import CENTERPOINT_CLASS_MAP
from depthyn.evaluation.matching import FrameMatch


@dataclass
class ClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    scores: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def ap(self) -> float:
        """Average precision (all-point interpolation)."""
        if not self.scores:
            return 0.0
        # Sort by score descending
        sorted_scores = sorted(self.scores, key=lambda x: -x[0])
        tp_cum = 0
        fp_cum = 0
        precisions = []
        recalls = []
        total_gt = self.tp + self.fn
        if total_gt == 0:
            return 0.0
        for score, is_tp in sorted_scores:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            p = tp_cum / (tp_cum + fp_cum)
            r = tp_cum / total_gt
            precisions.append(p)
            recalls.append(r)
        # All-point interpolation
        max_p = 0.0
        ap = 0.0
        prev_r = 0.0
        for i in range(len(precisions) - 1, -1, -1):
            max_p = max(max_p, precisions[i])
            precisions[i] = max_p
        for i in range(len(recalls)):
            r = recalls[i]
            if i == 0:
                ap += precisions[i] * r
            else:
                ap += precisions[i] * (r - recalls[i - 1])
        return ap

    def to_dict(self) -> dict[str, object]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "ap": round(self.ap, 4),
        }


@dataclass
class EvaluationResult:
    matched_frames: int = 0
    total_gt_objects: int = 0
    total_detections: int = 0
    overall: ClassMetrics = field(default_factory=ClassMetrics)
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "matched_frames": self.matched_frames,
            "total_gt_objects": self.total_gt_objects,
            "total_detections": self.total_detections,
            "overall": self.overall.to_dict(),
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
            "confusion_matrix": self.confusion,
        }


def _ensure_class(result: EvaluationResult, label: str) -> ClassMetrics:
    if label not in result.per_class:
        result.per_class[label] = ClassMetrics()
    return result.per_class[label]


def _add_confusion(result: EvaluationResult, pred: str, gt: str) -> None:
    if pred not in result.confusion:
        result.confusion[pred] = {}
    result.confusion[pred][gt] = result.confusion[pred].get(gt, 0) + 1


def compute_metrics(matches: list[FrameMatch]) -> EvaluationResult:
    """Aggregate per-frame matches into evaluation metrics."""
    result = EvaluationResult()

    for fm in matches:
        result.matched_frames += 1
        result.total_gt_objects += len(fm.true_positives) + len(fm.false_negatives)
        result.total_detections += len(fm.true_positives) + len(fm.false_positives)

        # True positives
        for det, gt_obj in fm.true_positives:
            det_label = CENTERPOINT_CLASS_MAP.get(det.label, det.label)
            gt_label = gt_obj.mapped_label

            result.overall.tp += 1
            result.overall.scores.append((det.score or 0.0, True))
            cls = _ensure_class(result, gt_label)
            cls.tp += 1
            cls.scores.append((det.score or 0.0, True))
            _add_confusion(result, det_label, gt_label)

        # False positives
        for det in fm.false_positives:
            det_label = CENTERPOINT_CLASS_MAP.get(det.label, det.label)
            result.overall.fp += 1
            result.overall.scores.append((det.score or 0.0, False))
            cls = _ensure_class(result, det_label)
            cls.fp += 1
            cls.scores.append((det.score or 0.0, False))

        # False negatives
        for gt_obj in fm.false_negatives:
            gt_label = gt_obj.mapped_label
            result.overall.fn += 1
            cls = _ensure_class(result, gt_label)
            cls.fn += 1

    return result
