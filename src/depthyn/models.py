from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

Point3D = tuple[float, float, float]


@dataclass(slots=True)
class Frame:
    frame_id: str
    timestamp_ns: int
    points: list[Point3D]
    source_path: Path
    sensor_frame_id: int | None = None


@dataclass(slots=True)
class Detection:
    detection_id: str
    centroid: Point3D
    bbox_min: Point3D
    bbox_max: Point3D
    point_count: int
    cell_count: int
    label: str = "object"
    score: float | None = None
    source: str = "baseline"
    heading_rad: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "centroid": list(self.centroid),
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "point_count": self.point_count,
            "cell_count": self.cell_count,
            "label": self.label,
            "score": None if self.score is None else round(self.score, 4),
            "source": self.source,
            "heading_rad": self.heading_rad,
        }


@dataclass(slots=True)
class Track:
    track_id: int
    centroid: Point3D
    velocity: Point3D
    bbox_min: Point3D
    bbox_max: Point3D
    point_count: int
    first_seen_ns: int
    last_seen_ns: int
    label: str = "object"
    score: float | None = None
    hits: int = 1
    misses: int = 0
    age_frames: int = 1
    total_distance_m: float = 0.0

    def predicted_centroid(self, timestamp_ns: int) -> Point3D:
        dt_s = max(0.0, (timestamp_ns - self.last_seen_ns) / 1_000_000_000.0)
        return (
            self.centroid[0] + self.velocity[0] * dt_s,
            self.centroid[1] + self.velocity[1] * dt_s,
            self.centroid[2] + self.velocity[2] * dt_s,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "score": None if self.score is None else round(self.score, 4),
            "centroid": list(self.centroid),
            "velocity_mps": list(self.velocity),
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "point_count": self.point_count,
            "first_seen_ns": self.first_seen_ns,
            "last_seen_ns": self.last_seen_ns,
            "hits": self.hits,
            "misses": self.misses,
            "age_frames": self.age_frames,
            "total_distance_m": round(self.total_distance_m, 3),
        }
