from __future__ import annotations

import csv
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

from depthyn.models import Detection, FramePose, Point3D


@dataclass(slots=True)
class GpsSample:
    timestamp_ns: int
    easting_m: float
    northing_m: float
    altitude_m: float


class GpsPoseProvider:
    def __init__(self, samples: list[GpsSample], *, source_path: Path) -> None:
        if len(samples) < 2:
            raise ValueError("GPS pose provider requires at least two samples.")
        self._samples = sorted(samples, key=lambda item: item.timestamp_ns)
        self._timestamps = [sample.timestamp_ns for sample in self._samples]
        self._origin = self._samples[0]
        self._headings = _compute_headings(self._samples)
        self.source_path = source_path

    def pose_at(self, timestamp_ns: int) -> FramePose:
        index = bisect_left(self._timestamps, timestamp_ns)
        if index <= 0:
            sample = self._samples[0]
            heading = self._headings[0]
            return self._pose_from_sample(sample, heading, timestamp_ns)
        if index >= len(self._samples):
            sample = self._samples[-1]
            heading = self._headings[-1]
            return self._pose_from_sample(sample, heading, timestamp_ns)

        earlier = self._samples[index - 1]
        later = self._samples[index]
        span_ns = max(1, later.timestamp_ns - earlier.timestamp_ns)
        alpha = (timestamp_ns - earlier.timestamp_ns) / span_ns

        east = _lerp(earlier.easting_m, later.easting_m, alpha)
        north = _lerp(earlier.northing_m, later.northing_m, alpha)
        # Keep Z in sensor-local replay space for now; raw GPS altitude is too noisy.
        up = 0.0
        heading = _lerp_angle(self._headings[index - 1], self._headings[index], alpha)
        return FramePose(
            timestamp_ns=timestamp_ns,
            position_m=(
                east - self._origin.easting_m,
                north - self._origin.northing_m,
                up,
            ),
            heading_rad=heading,
            source="gps",
        )

    def metadata(self) -> dict[str, object]:
        return {
            "source": "gps",
            "gps_path": str(self.source_path),
            "sample_count": len(self._samples),
            "origin_easting_m": round(self._origin.easting_m, 4),
            "origin_northing_m": round(self._origin.northing_m, 4),
            "origin_altitude_m": round(self._origin.altitude_m, 4),
            "axes": {
                "x": "east",
                "y": "north",
                "z": "up_local_sensor",
            },
            "altitude_mode": "ignored_for_world_translation",
        }

    def _pose_from_sample(
        self, sample: GpsSample, heading: float, timestamp_ns: int
    ) -> FramePose:
        return FramePose(
            timestamp_ns=timestamp_ns,
            position_m=(
                sample.easting_m - self._origin.easting_m,
                sample.northing_m - self._origin.northing_m,
                0.0,
            ),
            heading_rad=heading,
            source="gps",
        )


def discover_gps_csv(input_dir: Path) -> Path:
    candidates = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No GPS CSV found under: {input_dir}")
    raw_candidates = [path for path in candidates if path.name.startswith("raw_gps_")]
    if len(raw_candidates) == 1:
        return raw_candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        "Expected exactly one GPS CSV for world alignment under "
        f"{input_dir}, found {len(candidates)}"
    )


def load_gps_pose_provider(gps_path: Path) -> GpsPoseProvider:
    samples: list[GpsSample] = []
    with gps_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pi_time_raw = row.get("pi_time_ns")
            east_raw = row.get("easting")
            north_raw = row.get("northing")
            altitude_raw = row.get("altitude_m") or "0.0"
            if not pi_time_raw or not east_raw or not north_raw:
                continue
            samples.append(
                GpsSample(
                    timestamp_ns=int(float(pi_time_raw)),
                    easting_m=float(east_raw),
                    northing_m=float(north_raw),
                    altitude_m=float(altitude_raw),
                )
            )
    return GpsPoseProvider(samples, source_path=gps_path)


def rotate_xy(point: Point3D, heading_rad: float) -> Point3D:
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    x, y, z = point
    return (
        x * cos_h - y * sin_h,
        x * sin_h + y * cos_h,
        z,
    )


def inverse_rotate_xy(point: Point3D, heading_rad: float) -> Point3D:
    return rotate_xy(point, -heading_rad)


def transform_points(points: list[Point3D], pose: FramePose) -> list[Point3D]:
    tx, ty, tz = pose.position_m
    transformed: list[Point3D] = []
    for point in points:
        rx, ry, rz = rotate_xy(point, pose.heading_rad)
        transformed.append((rx + tx, ry + ty, rz + tz))
    return transformed


def inverse_transform_points(points: list[Point3D], pose: FramePose) -> list[Point3D]:
    tx, ty, tz = pose.position_m
    transformed: list[Point3D] = []
    for point in points:
        shifted = (point[0] - tx, point[1] - ty, point[2] - tz)
        transformed.append(inverse_rotate_xy(shifted, pose.heading_rad))
    return transformed


def transform_detection(detection: Detection, pose: FramePose) -> Detection:
    cx, cy, cz = transform_points([detection.centroid], pose)[0]
    sx = detection.bbox_max[0] - detection.bbox_min[0]
    sy = detection.bbox_max[1] - detection.bbox_min[1]
    sz = detection.bbox_max[2] - detection.bbox_min[2]
    heading = detection.heading_rad
    if heading is not None:
        heading += pose.heading_rad
    return Detection(
        detection_id=detection.detection_id,
        centroid=(cx, cy, cz),
        bbox_min=(cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0),
        bbox_max=(cx + sx / 2.0, cy + sy / 2.0, cz + sz / 2.0),
        point_count=detection.point_count,
        cell_count=detection.cell_count,
        label=detection.label,
        score=detection.score,
        source=detection.source,
        heading_rad=heading,
    )


def inverse_transform_detection(detection: Detection, pose: FramePose) -> Detection:
    cx, cy, cz = inverse_transform_points([detection.centroid], pose)[0]
    sx = detection.bbox_max[0] - detection.bbox_min[0]
    sy = detection.bbox_max[1] - detection.bbox_min[1]
    sz = detection.bbox_max[2] - detection.bbox_min[2]
    heading = detection.heading_rad
    if heading is not None:
        heading -= pose.heading_rad
    return Detection(
        detection_id=detection.detection_id,
        centroid=(cx, cy, cz),
        bbox_min=(cx - sx / 2.0, cy - sy / 2.0, cz - sz / 2.0),
        bbox_max=(cx + sx / 2.0, cy + sy / 2.0, cz + sz / 2.0),
        point_count=detection.point_count,
        cell_count=detection.cell_count,
        label=detection.label,
        score=detection.score,
        source=detection.source,
        heading_rad=heading,
    )


def _compute_headings(samples: list[GpsSample]) -> list[float]:
    headings: list[float] = []
    last_heading = 0.0
    for index, sample in enumerate(samples):
        prev_index = max(0, index - 1)
        next_index = min(len(samples) - 1, index + 1)
        prev_sample = samples[prev_index]
        next_sample = samples[next_index]
        delta_e = next_sample.easting_m - prev_sample.easting_m
        delta_n = next_sample.northing_m - prev_sample.northing_m
        if math.hypot(delta_e, delta_n) >= 0.25:
            last_heading = math.atan2(delta_n, delta_e)
        headings.append(last_heading)
    return headings


def _lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * alpha


def _lerp_angle(start: float, end: float, alpha: float) -> float:
    delta = math.atan2(math.sin(end - start), math.cos(end - start))
    return start + delta * alpha
