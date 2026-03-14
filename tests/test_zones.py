from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from depthyn.models import Detection
from depthyn.rules.zones import ZoneMonitor, load_zone_definitions
from depthyn.tracking.simple import SimpleTracker


class ZoneRuleTests(unittest.TestCase):
    def test_zone_monitor_emits_enter_dwell_and_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "zones.json"
            config_path.write_text(
                json.dumps(
                    {
                        "zones": [
                            {
                                "zone_id": "yard",
                                "name": "Yard",
                                "min_xy": [0.0, 0.0],
                                "max_xy": [5.0, 5.0],
                                "dwell_alert_seconds": 1.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            zones = load_zone_definitions(config_path)
            self.assertEqual(len(zones), 1)

            tracker = SimpleTracker(max_distance_m=10.0, max_missed_frames=0)
            monitor = ZoneMonitor(zones)

            inside = Detection(
                detection_id="det-1",
                centroid=(2.0, 2.0, 1.0),
                bbox_min=(1.5, 1.5, 0.0),
                bbox_max=(2.5, 2.5, 2.0),
                point_count=10,
                cell_count=4,
            )
            tracks = tracker.update([inside], timestamp_ns=1_000_000_000)
            occupancy, events = monitor.evaluate(tracks, timestamp_ns=1_000_000_000)
            self.assertEqual(occupancy[0].object_count, 1)
            self.assertEqual([event.event_type for event in events], ["entered"])

            tracks = tracker.update([inside], timestamp_ns=2_500_000_000)
            _, events = monitor.evaluate(tracks, timestamp_ns=2_500_000_000)
            self.assertEqual([event.event_type for event in events], ["dwell"])

            tracks = tracker.update([], timestamp_ns=3_000_000_000)
            _, events = monitor.evaluate(tracks, timestamp_ns=3_000_000_000)
            self.assertEqual([event.event_type for event in events], ["exited"])


if __name__ == "__main__":
    unittest.main()
