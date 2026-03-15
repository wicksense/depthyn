from __future__ import annotations

from depthyn.models import Detection, Track
from depthyn.perception.preprocess import distance_xy


class SimpleTracker:
    def __init__(self, max_distance_m: float, max_missed_frames: int) -> None:
        self.max_distance_m = max_distance_m
        self.max_missed_frames = max_missed_frames
        self._next_track_id = 1
        self._active_tracks: dict[int, Track] = {}
        self._finished_tracks: list[Track] = []

    def update(self, detections: list[Detection], timestamp_ns: int) -> list[Track]:
        assignments, unassigned_tracks, unassigned_detections = self._assign(
            detections, timestamp_ns
        )

        for track_id, detection_index in assignments.items():
            track = self._active_tracks[track_id]
            detection = detections[detection_index]
            self._apply_detection(track, detection, timestamp_ns)

        for track_id in unassigned_tracks:
            track = self._active_tracks[track_id]
            track.misses += 1
            track.age_frames += 1

        stale_tracks = [
            track_id
            for track_id, track in self._active_tracks.items()
            if track.misses > self.max_missed_frames
        ]
        for track_id in stale_tracks:
            self._finished_tracks.append(self._active_tracks.pop(track_id))

        for detection_index in unassigned_detections:
            detection = detections[detection_index]
            track = Track(
                track_id=self._next_track_id,
                centroid=detection.centroid,
                velocity=(0.0, 0.0, 0.0),
                bbox_min=detection.bbox_min,
                bbox_max=detection.bbox_max,
                point_count=detection.point_count,
                first_seen_ns=timestamp_ns,
                last_seen_ns=timestamp_ns,
                label=detection.label,
                score=detection.score,
            )
            self._active_tracks[track.track_id] = track
            self._next_track_id += 1

        return sorted(self._active_tracks.values(), key=lambda track: track.track_id)

    def all_tracks(self) -> list[Track]:
        return sorted(
            [*self._finished_tracks, *self._active_tracks.values()],
            key=lambda track: track.track_id,
        )

    def _assign(
        self, detections: list[Detection], timestamp_ns: int
    ) -> tuple[dict[int, int], set[int], set[int]]:
        candidate_pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._active_tracks.items():
            predicted = track.predicted_centroid(timestamp_ns)
            for detection_index, detection in enumerate(detections):
                distance_m = distance_xy(predicted, detection.centroid)
                if distance_m <= self.max_distance_m:
                    candidate_pairs.append((distance_m, track_id, detection_index))

        candidate_pairs.sort(key=lambda item: item[0])
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        for _, track_id, detection_index in candidate_pairs:
            if track_id in used_tracks or detection_index in used_detections:
                continue
            assignments[track_id] = detection_index
            used_tracks.add(track_id)
            used_detections.add(detection_index)

        unassigned_tracks = set(self._active_tracks) - used_tracks
        unassigned_detections = set(range(len(detections))) - used_detections
        return assignments, unassigned_tracks, unassigned_detections

    def _apply_detection(
        self, track: Track, detection: Detection, timestamp_ns: int
    ) -> None:
        dt_s = max(1e-6, (timestamp_ns - track.last_seen_ns) / 1_000_000_000.0)
        dx = detection.centroid[0] - track.centroid[0]
        dy = detection.centroid[1] - track.centroid[1]
        dz = detection.centroid[2] - track.centroid[2]

        track.total_distance_m += distance_xy(track.centroid, detection.centroid)
        track.velocity = (dx / dt_s, dy / dt_s, dz / dt_s)
        track.centroid = detection.centroid
        track.bbox_min = detection.bbox_min
        track.bbox_max = detection.bbox_max
        track.point_count = detection.point_count
        track.label = detection.label
        track.score = detection.score
        track.last_seen_ns = timestamp_ns
        track.hits += 1
        track.misses = 0
        track.age_frames += 1

