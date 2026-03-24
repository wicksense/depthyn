"""Parse Ouster Gemini classification logs as ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Map Ouster classifications to normalized labels
OUSTER_CLASS_MAP: dict[str, str] = {
    "PERSON": "person",
    "VEHICLE": "vehicle",
    "BICYCLE": "bicycle",
}

# Map CenterPoint (nuScenes) classes to the same normalized labels
CENTERPOINT_CLASS_MAP: dict[str, str] = {
    "pedestrian": "person",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "bicycle": "bicycle",
}


@dataclass(slots=True)
class GroundTruthObject:
    gt_id: int
    classification: str
    mapped_label: str
    position: tuple[float, float, float]
    dimensions: tuple[float, float, float]  # (length, width, height)
    confidence: float
    heading: float


@dataclass(slots=True)
class GroundTruthFrame:
    frame_count: int
    timestamp_us: int
    objects: list[GroundTruthObject]


def parse_ground_truth_log(
    log_path: Path,
    *,
    min_distance_m: float = 2.0,
) -> list[GroundTruthFrame]:
    """Parse an Ouster classification log file.

    Args:
        log_path: Path to the JSON-per-line log file.
        min_distance_m: Exclude objects closer than this (likely self-returns).

    Returns:
        List of GroundTruthFrame sorted by frame_count.
    """
    frames: list[GroundTruthFrame] = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Skip occupation/zone records
            if "frame_count" not in record:
                continue

            objects: list[GroundTruthObject] = []
            for obj in record.get("objects", []):
                classification = obj.get("classification", "UNKNOWN")

                # Skip UNKNOWN objects
                mapped = OUSTER_CLASS_MAP.get(classification)
                if mapped is None:
                    continue

                # Skip self-returns (too close to sensor)
                distance = obj.get("distance_to_primary_sensor", 0)
                if distance < min_distance_m:
                    continue

                pos = obj["position"]
                dims = obj["dimensions"]
                objects.append(GroundTruthObject(
                    gt_id=obj["id"],
                    classification=classification,
                    mapped_label=mapped,
                    position=(pos["x"], pos["y"], pos["z"]),
                    dimensions=(dims["length"], dims["width"], dims["height"]),
                    confidence=obj.get("classification_confidence", 0.0),
                    heading=obj.get("heading", 0.0),
                ))

            frames.append(GroundTruthFrame(
                frame_count=record["frame_count"],
                timestamp_us=record["timestamp"],
                objects=objects,
            ))

    frames.sort(key=lambda f: f.frame_count)
    return frames
