from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from depthyn.models import Point3D, Track


class ZoneConfigError(ValueError):
    """Raised when a zone definition file is invalid."""


@dataclass(slots=True)
class ZoneDefinition:
    zone_id: str
    name: str
    min_xy: tuple[float, float]
    max_xy: tuple[float, float]
    kind: str = "inclusion"
    color: str = "#6f8f77"
    dwell_alert_seconds: float = 0.0
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ZoneDefinition":
        try:
            raw_zone_id = payload.get("zone_id") or payload["name"]
            raw_name = payload.get("name") or payload["zone_id"]
            raw_min_xy = payload["min_xy"]
            raw_max_xy = payload["max_xy"]
        except KeyError as exc:
            raise ZoneConfigError(f"Missing zone field: {exc.args[0]}") from exc

        if not isinstance(raw_zone_id, str) or not raw_zone_id.strip():
            raise ZoneConfigError("zone_id must be a non-empty string")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ZoneConfigError("name must be a non-empty string")
        if not isinstance(raw_min_xy, list | tuple) or len(raw_min_xy) != 2:
            raise ZoneConfigError("min_xy must contain exactly two numeric values")
        if not isinstance(raw_max_xy, list | tuple) or len(raw_max_xy) != 2:
            raise ZoneConfigError("max_xy must contain exactly two numeric values")

        min_x = float(raw_min_xy[0])
        min_y = float(raw_min_xy[1])
        max_x = float(raw_max_xy[0])
        max_y = float(raw_max_xy[1])
        normalized_min = (min(min_x, max_x), min(min_y, max_y))
        normalized_max = (max(min_x, max_x), max(min_y, max_y))
        raw_tags = payload.get("tags", ())
        if not isinstance(raw_tags, list | tuple):
            raise ZoneConfigError("tags must be a list of strings when provided")

        return cls(
            zone_id=raw_zone_id.strip(),
            name=raw_name.strip(),
            min_xy=normalized_min,
            max_xy=normalized_max,
            kind=str(payload.get("kind", "inclusion")),
            color=str(payload.get("color", "#6f8f77")),
            dwell_alert_seconds=max(0.0, float(payload.get("dwell_alert_seconds", 0.0))),
            tags=tuple(str(tag) for tag in raw_tags),
        )

    def contains(self, point: Point3D) -> bool:
        return (
            self.min_xy[0] <= point[0] <= self.max_xy[0]
            and self.min_xy[1] <= point[1] <= self.max_xy[1]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "kind": self.kind,
            "min_xy": [round(self.min_xy[0], 3), round(self.min_xy[1], 3)],
            "max_xy": [round(self.max_xy[0], 3), round(self.max_xy[1], 3)],
            "color": self.color,
            "dwell_alert_seconds": round(self.dwell_alert_seconds, 3),
            "tags": list(self.tags),
        }


@dataclass(slots=True)
class ZoneOccupancy:
    zone_id: str
    zone_name: str
    kind: str
    color: str
    track_ids: list[int]
    object_count: int
    dwell_alert_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "kind": self.kind,
            "color": self.color,
            "track_ids": self.track_ids,
            "object_count": self.object_count,
            "dwell_alert_seconds": round(self.dwell_alert_seconds, 3),
        }


@dataclass(slots=True)
class ZoneEvent:
    event_type: str
    zone_id: str
    zone_name: str
    track_id: int
    timestamp_ns: int
    dwell_seconds: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "event_type": self.event_type,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "track_id": self.track_id,
            "timestamp_ns": self.timestamp_ns,
        }
        if self.dwell_seconds is not None:
            payload["dwell_seconds"] = round(self.dwell_seconds, 3)
        return payload


class ZoneMonitor:
    def __init__(self, zones: list[ZoneDefinition]) -> None:
        self.zones = zones
        self._zone_by_id = {zone.zone_id: zone for zone in zones}
        self._active_memberships: dict[tuple[int, str], int] = {}
        self._dwell_emitted: set[tuple[int, str]] = set()

    def evaluate(
        self, tracks: list[Track], timestamp_ns: int
    ) -> tuple[list[ZoneOccupancy], list[ZoneEvent]]:
        events: list[ZoneEvent] = []
        occupancy: list[ZoneOccupancy] = []
        current_memberships: set[tuple[int, str]] = set()

        sorted_tracks = sorted(tracks, key=lambda track: track.track_id)

        for zone in self.zones:
            inside_track_ids: list[int] = []
            for track in sorted_tracks:
                if not zone.contains(track.centroid):
                    continue
                inside_track_ids.append(track.track_id)
                membership = (track.track_id, zone.zone_id)
                current_memberships.add(membership)
                if membership not in self._active_memberships:
                    self._active_memberships[membership] = timestamp_ns
                    events.append(
                        ZoneEvent(
                            event_type="entered",
                            zone_id=zone.zone_id,
                            zone_name=zone.name,
                            track_id=track.track_id,
                            timestamp_ns=timestamp_ns,
                        )
                    )
                    continue

                dwell_seconds = (
                    timestamp_ns - self._active_memberships[membership]
                ) / 1_000_000_000.0
                if (
                    zone.dwell_alert_seconds > 0.0
                    and dwell_seconds >= zone.dwell_alert_seconds
                    and membership not in self._dwell_emitted
                ):
                    self._dwell_emitted.add(membership)
                    events.append(
                        ZoneEvent(
                            event_type="dwell",
                            zone_id=zone.zone_id,
                            zone_name=zone.name,
                            track_id=track.track_id,
                            timestamp_ns=timestamp_ns,
                            dwell_seconds=dwell_seconds,
                        )
                    )

            occupancy.append(
                ZoneOccupancy(
                    zone_id=zone.zone_id,
                    zone_name=zone.name,
                    kind=zone.kind,
                    color=zone.color,
                    track_ids=inside_track_ids,
                    object_count=len(inside_track_ids),
                    dwell_alert_seconds=zone.dwell_alert_seconds,
                )
            )

        exited_memberships = sorted(set(self._active_memberships) - current_memberships)
        for membership in exited_memberships:
            entry_ts = self._active_memberships.pop(membership)
            self._dwell_emitted.discard(membership)
            track_id, zone_id = membership
            zone = self._zone_by_id[zone_id]
            dwell_seconds = (timestamp_ns - entry_ts) / 1_000_000_000.0
            events.append(
                ZoneEvent(
                    event_type="exited",
                    zone_id=zone.zone_id,
                    zone_name=zone.name,
                    track_id=track_id,
                    timestamp_ns=timestamp_ns,
                    dwell_seconds=dwell_seconds,
                )
            )

        return occupancy, events


def load_zone_definitions(zone_config_path: Path) -> list[ZoneDefinition]:
    try:
        raw_payload = json.loads(zone_config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ZoneConfigError(f"Zone config not found: {zone_config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ZoneConfigError(
            f"Zone config is not valid JSON: {zone_config_path}"
        ) from exc

    if isinstance(raw_payload, dict):
        raw_zones = raw_payload.get("zones", [])
    elif isinstance(raw_payload, list):
        raw_zones = raw_payload
    else:
        raise ZoneConfigError("Zone config must be a JSON object or list")

    if not isinstance(raw_zones, list):
        raise ZoneConfigError("zones must be a list")

    zones = [ZoneDefinition.from_dict(item) for item in raw_zones]
    zone_ids = [zone.zone_id for zone in zones]
    if len(zone_ids) != len(set(zone_ids)):
        raise ZoneConfigError("zone_id values must be unique")
    return zones
