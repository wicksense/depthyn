from __future__ import annotations

from depthyn.config import ReplayConfig
from depthyn.detectors.base import DetectorResult
from depthyn.models import Detection, Frame
from depthyn.perception.clustering import cluster_points
from depthyn.perception.ground import remove_ground


def classify_cluster(det: Detection) -> tuple[str, float]:
    """Classify a cluster by XY footprint size.

    For stationary elevated LiDAR, vertical extent is unreliable —
    people at range appear as thin flat clusters because the sensor
    only sees their top/shoulders. Classification relies on the XY
    footprint dimensions instead.

    Returns (label, confidence).
    """
    dx = det.bbox_max[0] - det.bbox_min[0]
    dy = det.bbox_max[1] - det.bbox_min[1]

    width = max(dx, dy)   # horizontal extent (longer axis)
    depth = min(dx, dy)   # horizontal extent (shorter axis)

    # Person: small compact footprint (< ~1.5m each side)
    if width <= 1.5 and depth <= 1.2:
        confidence = 0.5
        # Higher confidence if footprint is compact and round-ish
        aspect = depth / width if width > 0.01 else 1.0
        if aspect > 0.4 and width <= 1.0:
            confidence = 0.7
        return "person", round(confidence, 2)

    # Bicycle: elongated, moderate footprint
    if width <= 2.5 and depth <= 1.0 and width > 1.2:
        return "bicycle", 0.4

    # Large vehicle: very wide/long
    if width > 5.0 or (width > 2.5 and depth > 2.0):
        return "vehicle", 0.6

    # Vehicle: wider footprint
    if width > 1.5:
        confidence = 0.5 if width > 2.5 else 0.35
        return "vehicle", confidence

    # Default
    return "object", 0.2


class BaselineClusterDetector:
    name = "baseline"
    input_mode = "foreground"

    def __init__(self, config: ReplayConfig) -> None:
        self._config = config

    def detect(self, frame: Frame, points: list[tuple[float, float, float]]) -> DetectorResult:
        # Remove ground plane before clustering
        elevated = remove_ground(
            points,
            cell_size_m=2.0,
            ground_tolerance_m=0.25,
            min_cell_points=3,
        )
        detections = cluster_points(
            elevated,
            cell_size_m=self._config.cluster_cell_size_m,
            min_cluster_points=self._config.min_cluster_points,
            min_cluster_cells=self._config.min_cluster_cells,
            min_cluster_height_m=self._config.min_cluster_height_m,
            max_cluster_height_m=self._config.max_cluster_height_m,
            max_cluster_width_m=self._config.max_cluster_width_m,
        )
        for detection in detections:
            detection.source = self.name
            label, score = classify_cluster(detection)
            detection.label = label
            detection.score = score
        return DetectorResult(
            detections=detections,
            input_point_count=len(points),
            metadata={
                "cluster_cell_size_m": self._config.cluster_cell_size_m,
                "elevated_points": len(elevated),
                "ground_removed": len(points) - len(elevated),
            },
        )

