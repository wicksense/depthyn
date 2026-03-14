from __future__ import annotations

import unittest

from depthyn.models import Detection
from depthyn.tracking.simple import SimpleTracker


class TrackerTests(unittest.TestCase):
    def test_track_persists_across_frames(self) -> None:
        tracker = SimpleTracker(max_distance_m=2.0, max_missed_frames=1)
        detection_a = Detection(
            detection_id="det-1",
            centroid=(0.0, 0.0, 1.0),
            bbox_min=(-0.5, -0.5, 0.0),
            bbox_max=(0.5, 0.5, 2.0),
            point_count=20,
            cell_count=5,
        )
        tracks = tracker.update([detection_a], timestamp_ns=1_000_000_000)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)

        detection_b = Detection(
            detection_id="det-1",
            centroid=(0.8, 0.1, 1.0),
            bbox_min=(0.3, -0.4, 0.0),
            bbox_max=(1.3, 0.6, 2.0),
            point_count=18,
            cell_count=4,
        )
        tracks = tracker.update([detection_b], timestamp_ns=2_000_000_000)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertGreater(tracks[0].total_distance_m, 0.0)

    def test_track_expires_after_misses(self) -> None:
        tracker = SimpleTracker(max_distance_m=2.0, max_missed_frames=0)
        detection = Detection(
            detection_id="det-1",
            centroid=(0.0, 0.0, 1.0),
            bbox_min=(-0.5, -0.5, 0.0),
            bbox_max=(0.5, 0.5, 2.0),
            point_count=20,
            cell_count=5,
        )
        tracker.update([detection], timestamp_ns=1_000_000_000)
        tracks = tracker.update([], timestamp_ns=2_000_000_000)
        self.assertEqual(tracks, [])
        self.assertEqual(len(tracker.all_tracks()), 1)


if __name__ == "__main__":
    unittest.main()

