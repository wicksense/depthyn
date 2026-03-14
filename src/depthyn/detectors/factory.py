from __future__ import annotations

from depthyn.config import ReplayConfig
from depthyn.detectors.baseline import BaselineClusterDetector
from depthyn.detectors.base import Detector, DetectorUnavailableError
from depthyn.detectors.mmdet3d import MMDet3DDetector
from depthyn.detectors.onnx_centerpoint import OnnxCenterPointDetector
from depthyn.detectors.precomputed import PrecomputedDetector


def create_detector(config: ReplayConfig) -> Detector:
    kind = config.detector.kind.lower()
    if kind == "baseline":
        return BaselineClusterDetector(config)
    if kind == "precomputed":
        return PrecomputedDetector(config.detector)
    if kind == "centerpoint-onnx":
        return OnnxCenterPointDetector(config.detector)
    if kind in {"pointpillars", "centerpoint", "dsvt"}:
        return MMDet3DDetector(config.detector)
    raise DetectorUnavailableError(f"Unsupported detector kind: {config.detector.kind}")
