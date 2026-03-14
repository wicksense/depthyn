from __future__ import annotations

from depthyn.config import ReplayConfig
from depthyn.detectors.baseline import BaselineClusterDetector
from depthyn.detectors.base import Detector, DetectorUnavailableError
from depthyn.detectors.mmdet3d import MMDet3DDetector


def create_detector(config: ReplayConfig) -> Detector:
    kind = config.detector.kind.lower()
    if kind == "baseline":
        return BaselineClusterDetector(config)
    if kind in {"pointpillars", "centerpoint", "dsvt"}:
        return MMDet3DDetector(config.detector)
    raise DetectorUnavailableError(f"Unsupported detector kind: {config.detector.kind}")
