from __future__ import annotations

from depthyn.config import ReplayConfig
from depthyn.detectors.base import DetectorResult
from depthyn.models import Frame
from depthyn.perception.clustering import cluster_points


class BaselineClusterDetector:
    name = "baseline"
    input_mode = "foreground"

    def __init__(self, config: ReplayConfig) -> None:
        self._config = config

    def detect(self, frame: Frame, points: list[tuple[float, float, float]]) -> DetectorResult:
        detections = cluster_points(
            points,
            cell_size_m=self._config.cluster_cell_size_m,
            min_cluster_points=self._config.min_cluster_points,
            min_cluster_cells=self._config.min_cluster_cells,
            min_cluster_height_m=self._config.min_cluster_height_m,
            max_cluster_height_m=self._config.max_cluster_height_m,
            max_cluster_width_m=self._config.max_cluster_width_m,
        )
        for detection in detections:
            detection.source = self.name
            detection.label = "object"
        return DetectorResult(
            detections=detections,
            input_point_count=len(points),
            metadata={"cluster_cell_size_m": self._config.cluster_cell_size_m},
        )

