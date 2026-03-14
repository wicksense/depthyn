from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from depthyn.models import Track


@dataclass(slots=True)
class SceneObject:
    object_id: str
    source_type: str
    label: str
    centroid: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    velocity_mps: tuple[float, float, float] | None = None
    speed_mps: float | None = None
    point_count: int | None = None

    @classmethod
    def from_track(cls, track: Track) -> "SceneObject":
        speed_mps = sqrt(
            (track.velocity[0] ** 2)
            + (track.velocity[1] ** 2)
            + (track.velocity[2] ** 2)
        )
        return cls(
            object_id=f"track-{track.track_id}",
            source_type="track",
            label="object",
            centroid=track.centroid,
            bbox_min=track.bbox_min,
            bbox_max=track.bbox_max,
            velocity_mps=track.velocity,
            speed_mps=speed_mps,
            point_count=track.point_count,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "object_id": self.object_id,
            "source_type": self.source_type,
            "label": self.label,
            "centroid": [round(value, 3) for value in self.centroid],
            "bbox_min": [round(value, 3) for value in self.bbox_min],
            "bbox_max": [round(value, 3) for value in self.bbox_max],
            "point_count": self.point_count,
        }
        if self.velocity_mps is not None:
            payload["velocity_mps"] = [round(value, 3) for value in self.velocity_mps]
        if self.speed_mps is not None:
            payload["speed_mps"] = round(self.speed_mps, 3)
        return payload


@dataclass(slots=True)
class SceneState:
    frame_index: int
    frame_id: str
    timestamp_ns: int
    mode: str
    stage: str
    detector_name: str
    objects: list[SceneObject] = field(default_factory=list)
    zones: list[dict[str, object]] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "mode": self.mode,
            "stage": self.stage,
            "detector_name": self.detector_name,
            "object_count": len(self.objects),
            "zone_count": len(self.zones),
            "event_count": len(self.events),
            "objects": [scene_object.to_dict() for scene_object in self.objects],
            "zones": self.zones,
            "events": self.events,
        }


def build_scene_state(
    *,
    frame_index: int,
    frame_id: str,
    timestamp_ns: int,
    mode: str,
    stage: str,
    detector_name: str,
    tracks: list[Track],
    zones: list[dict[str, object]] | None = None,
    events: list[dict[str, object]] | None = None,
) -> SceneState:
    return SceneState(
        frame_index=frame_index,
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        mode=mode,
        stage=stage,
        detector_name=detector_name,
        objects=[SceneObject.from_track(track) for track in tracks],
        zones=list(zones or []),
        events=list(events or []),
    )
