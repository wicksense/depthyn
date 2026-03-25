from __future__ import annotations

import json
from dataclasses import dataclass
from math import isclose
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
class TripwireDefinition:
    tripwire_id: str
    name: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    color: str = "#59b8ff"
    positive_direction_label: str = "negative_to_positive"
    negative_direction_label: str = "positive_to_negative"
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TripwireDefinition":
        try:
            raw_tripwire_id = payload.get("tripwire_id") or payload["name"]
            raw_name = payload.get("name") or payload["tripwire_id"]
            raw_start_xy = payload.get("start_xy") or payload["point_a"]
            raw_end_xy = payload.get("end_xy") or payload["point_b"]
        except KeyError as exc:
            raise ZoneConfigError(f"Missing tripwire field: {exc.args[0]}") from exc

        if not isinstance(raw_tripwire_id, str) or not raw_tripwire_id.strip():
            raise ZoneConfigError("tripwire_id must be a non-empty string")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ZoneConfigError("name must be a non-empty string")
        if not isinstance(raw_start_xy, list | tuple) or len(raw_start_xy) != 2:
            raise ZoneConfigError("start_xy must contain exactly two numeric values")
        if not isinstance(raw_end_xy, list | tuple) or len(raw_end_xy) != 2:
            raise ZoneConfigError("end_xy must contain exactly two numeric values")

        start_xy = (float(raw_start_xy[0]), float(raw_start_xy[1]))
        end_xy = (float(raw_end_xy[0]), float(raw_end_xy[1]))
        if isclose(start_xy[0], end_xy[0]) and isclose(start_xy[1], end_xy[1]):
            raise ZoneConfigError("tripwire start_xy and end_xy must not be identical")

        raw_tags = payload.get("tags", ())
        if not isinstance(raw_tags, list | tuple):
            raise ZoneConfigError("tags must be a list of strings when provided")

        return cls(
            tripwire_id=raw_tripwire_id.strip(),
            name=raw_name.strip(),
            start_xy=start_xy,
            end_xy=end_xy,
            color=str(payload.get("color", "#59b8ff")),
            positive_direction_label=str(
                payload.get("positive_direction_label", "negative_to_positive")
            ),
            negative_direction_label=str(
                payload.get("negative_direction_label", "positive_to_negative")
            ),
            tags=tuple(str(tag) for tag in raw_tags),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tripwire_id": self.tripwire_id,
            "name": self.name,
            "start_xy": [round(self.start_xy[0], 3), round(self.start_xy[1], 3)],
            "end_xy": [round(self.end_xy[0], 3), round(self.end_xy[1], 3)],
            "color": self.color,
            "positive_direction_label": self.positive_direction_label,
            "negative_direction_label": self.negative_direction_label,
            "tags": list(self.tags),
        }

    def _signed_side(self, point: Point3D) -> float:
        edge_x = self.end_xy[0] - self.start_xy[0]
        edge_y = self.end_xy[1] - self.start_xy[1]
        point_x = point[0] - self.start_xy[0]
        point_y = point[1] - self.start_xy[1]
        return edge_x * point_y - edge_y * point_x

    def crossing_direction(
        self, previous_point: Point3D, current_point: Point3D, epsilon: float = 1e-6
    ) -> str | None:
        if not _segments_intersect(
            (previous_point[0], previous_point[1]),
            (current_point[0], current_point[1]),
            self.start_xy,
            self.end_xy,
            epsilon=epsilon,
        ):
            return None

        edge_x = self.end_xy[0] - self.start_xy[0]
        edge_y = self.end_xy[1] - self.start_xy[1]
        motion_x = current_point[0] - previous_point[0]
        motion_y = current_point[1] - previous_point[1]
        positive_normal_x = edge_y
        positive_normal_y = -edge_x
        normal_projection = (
            motion_x * positive_normal_x + motion_y * positive_normal_y
        )

        if normal_projection > epsilon:
            return self.positive_direction_label
        if normal_projection < -epsilon:
            return self.negative_direction_label

        return None


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
    rule_kind: str = "zone"
    dwell_seconds: float | None = None
    direction: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "event_type": self.event_type,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "track_id": self.track_id,
            "timestamp_ns": self.timestamp_ns,
            "rule_kind": self.rule_kind,
        }
        if self.dwell_seconds is not None:
            payload["dwell_seconds"] = round(self.dwell_seconds, 3)
        if self.direction is not None:
            payload["direction"] = self.direction
        return payload


class ZoneMonitor:
    def __init__(
        self,
        zones: list[ZoneDefinition],
        tripwires: list[TripwireDefinition] | None = None,
    ) -> None:
        self.zones = zones
        self.tripwires = list(tripwires or [])
        self._zone_by_id = {zone.zone_id: zone for zone in zones}
        self._active_memberships: dict[tuple[int, str], int] = {}
        self._dwell_emitted: set[tuple[int, str]] = set()
        self._last_positions: dict[int, Point3D] = {}

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
                            rule_kind="zone",
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
                            rule_kind="zone",
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
                    rule_kind="zone",
                    dwell_seconds=dwell_seconds,
                )
            )

        for track in sorted_tracks:
            previous_point = self._last_positions.get(track.track_id)
            if previous_point is None:
                continue
            for tripwire in self.tripwires:
                direction = tripwire.crossing_direction(previous_point, track.centroid)
                if direction is None:
                    continue
                events.append(
                    ZoneEvent(
                        event_type="crossed",
                        zone_id=tripwire.tripwire_id,
                        zone_name=tripwire.name,
                        track_id=track.track_id,
                        timestamp_ns=timestamp_ns,
                        rule_kind="tripwire",
                        direction=direction,
                    )
                )

        self._last_positions = {
            track.track_id: track.centroid for track in sorted_tracks
        }
        return occupancy, events


def load_zone_definitions(zone_config_path: Path) -> list[ZoneDefinition]:
    zones, _ = load_rule_definitions(zone_config_path)
    return zones


def load_rule_definitions(
    zone_config_path: Path,
) -> tuple[list[ZoneDefinition], list[TripwireDefinition]]:
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
        raw_tripwires = raw_payload.get("tripwires", [])
    elif isinstance(raw_payload, list):
        raw_zones = raw_payload
        raw_tripwires = []
    else:
        raise ZoneConfigError("Zone config must be a JSON object or list")

    if not isinstance(raw_zones, list):
        raise ZoneConfigError("zones must be a list")
    if not isinstance(raw_tripwires, list):
        raise ZoneConfigError("tripwires must be a list")

    zones = [ZoneDefinition.from_dict(item) for item in raw_zones]
    zone_ids = [zone.zone_id for zone in zones]
    if len(zone_ids) != len(set(zone_ids)):
        raise ZoneConfigError("zone_id values must be unique")
    tripwires = [TripwireDefinition.from_dict(item) for item in raw_tripwires]
    tripwire_ids = [tripwire.tripwire_id for tripwire in tripwires]
    if len(tripwire_ids) != len(set(tripwire_ids)):
        raise ZoneConfigError("tripwire_id values must be unique")
    return zones, tripwires


def _segments_intersect(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    point_c: tuple[float, float],
    point_d: tuple[float, float],
    *,
    epsilon: float = 1e-6,
) -> bool:
    orientation_abc = _orientation(point_a, point_b, point_c)
    orientation_abd = _orientation(point_a, point_b, point_d)
    orientation_cda = _orientation(point_c, point_d, point_a)
    orientation_cdb = _orientation(point_c, point_d, point_b)

    if orientation_abc * orientation_abd < -epsilon and orientation_cda * orientation_cdb < -epsilon:
        return True

    return (
        _point_on_segment(point_a, point_b, point_c, epsilon)
        or _point_on_segment(point_a, point_b, point_d, epsilon)
        or _point_on_segment(point_c, point_d, point_a, epsilon)
        or _point_on_segment(point_c, point_d, point_b, epsilon)
    )


def _orientation(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    point_c: tuple[float, float],
) -> float:
    return (
        (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
        - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
    )


def _point_on_segment(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    point_c: tuple[float, float],
    epsilon: float,
) -> bool:
    if abs(_orientation(point_a, point_b, point_c)) > epsilon:
        return False
    return (
        min(point_a[0], point_b[0]) - epsilon
        <= point_c[0]
        <= max(point_a[0], point_b[0]) + epsilon
        and min(point_a[1], point_b[1]) - epsilon
        <= point_c[1]
        <= max(point_a[1], point_b[1]) + epsilon
    )
