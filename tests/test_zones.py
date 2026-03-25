from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from depthyn.models import Detection
from depthyn.rules.zones import ZoneMonitor, load_rule_definitions, load_zone_definitions
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

    def test_tripwire_monitor_emits_crossed_event_with_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rules.json"
            config_path.write_text(
                json.dumps(
                    {
                        "tripwires": [
                            {
                                "tripwire_id": "gate",
                                "name": "Gate",
                                "start_xy": [0.0, -2.0],
                                "end_xy": [0.0, 2.0],
                                "positive_direction_label": "eastbound",
                                "negative_direction_label": "westbound",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            zones, tripwires = load_rule_definitions(config_path)
            self.assertEqual(zones, [])
            self.assertEqual(len(tripwires), 1)

            tracker = SimpleTracker(max_distance_m=10.0, max_missed_frames=0)
            monitor = ZoneMonitor(zones, tripwires)

            west = Detection(
                detection_id="det-west",
                centroid=(-2.0, 0.0, 1.0),
                bbox_min=(-2.5, -0.5, 0.0),
                bbox_max=(-1.5, 0.5, 2.0),
                point_count=10,
                cell_count=4,
            )
            east = Detection(
                detection_id="det-east",
                centroid=(2.0, 0.0, 1.0),
                bbox_min=(1.5, -0.5, 0.0),
                bbox_max=(2.5, 0.5, 2.0),
                point_count=10,
                cell_count=4,
            )

            tracks = tracker.update([west], timestamp_ns=1_000_000_000)
            _, events = monitor.evaluate(tracks, timestamp_ns=1_000_000_000)
            self.assertEqual(events, [])

            tracks = tracker.update([east], timestamp_ns=2_000_000_000)
            _, events = monitor.evaluate(tracks, timestamp_ns=2_000_000_000)
            self.assertEqual([event.event_type for event in events], ["crossed"])
            self.assertEqual(events[0].rule_kind, "tripwire")
            self.assertEqual(events[0].direction, "eastbound")

    def test_rule_loader_supports_mixed_zone_and_tripwire_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rules.json"
            config_path.write_text(
                json.dumps(
                    {
                        "zones": [
                            {
                                "zone_id": "yard",
                                "name": "Yard",
                                "min_xy": [0.0, 0.0],
                                "max_xy": [5.0, 5.0],
                            }
                        ],
                        "tripwires": [
                            {
                                "tripwire_id": "gate",
                                "name": "Gate",
                                "start_xy": [0.0, -1.0],
                                "end_xy": [0.0, 1.0],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            zones, tripwires = load_rule_definitions(config_path)
            self.assertEqual([zone.zone_id for zone in zones], ["yard"])
            self.assertEqual([tripwire.tripwire_id for tripwire in tripwires], ["gate"])


if __name__ == "__main__":
    unittest.main()
