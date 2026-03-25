from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from depthyn.viewer import (
    load_saved_rules,
    rule_storage_path,
    save_rules,
    validate_rules_payload,
)


class ViewerServiceTests(unittest.TestCase):
    def test_rule_storage_path_uses_summary_stem_and_frame(self) -> None:
        summary_path = Path("/tmp/example-summary.json")
        self.assertEqual(
            rule_storage_path(summary_path, "world"),
            Path("/tmp/example-summary.rules.world.json"),
        )

    def test_validate_rules_payload_requires_lists(self) -> None:
        with self.assertRaises(ValueError):
            validate_rules_payload({"zones": {}}, frame="sensor")

        payload = validate_rules_payload(
            {
                "zones": [{"zone_id": "yard"}],
                "tripwires": [{"tripwire_id": "gate"}],
            },
            frame="world",
        )
        self.assertEqual(payload["reference_frame"], "world")
        self.assertEqual(len(payload["zones"]), 1)
        self.assertEqual(len(payload["tripwires"]), 1)

    def test_save_and_load_rules_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "replay.json"
            summary_path.write_text(json.dumps({"project": "Depthyn"}), encoding="utf-8")
            payload = {
                "zones": [{"zone_id": "yard", "name": "Yard", "min_xy": [0, 0], "max_xy": [1, 1]}],
                "tripwires": [{"tripwire_id": "gate", "name": "Gate", "start_xy": [0, 0], "end_xy": [1, 0]}],
            }
            saved_path = save_rules(summary_path, "sensor", payload)
            self.assertTrue(saved_path.exists())

            loaded = load_saved_rules(summary_path, "sensor")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["reference_frame"], "sensor")
            self.assertEqual(loaded["zones"][0]["zone_id"], "yard")
            self.assertEqual(loaded["tripwires"][0]["tripwire_id"], "gate")


if __name__ == "__main__":
    unittest.main()
