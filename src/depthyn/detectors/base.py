from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from depthyn.models import Detection, Frame, Point3D


@dataclass(slots=True)
class DetectorResult:
    detections: list[Detection]
    input_point_count: int
    metadata: dict[str, object]


class Detector(Protocol):
    name: str
    input_mode: str

    def detect(self, frame: Frame, points: list[Point3D]) -> DetectorResult:
        """Return detections for a frame."""


class DetectorError(RuntimeError):
    """Base error for detector execution issues."""


class DetectorUnavailableError(DetectorError):
    """Raised when an optional detector backend is not configured."""

