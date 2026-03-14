from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from depthyn.config import ReplayConfig
from depthyn.pipeline import run_replay


class PipelineTests(unittest.TestCase):
    def test_replay_processes_synthetic_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame1 = root / "frame_0001.csv"
            frame2 = root / "frame_0002.csv"
            zones = root / "zones.json"
            self._write_frame(
                frame1,
                [
                    (1_000_000_000, 0.0, 0.0, 0.4),
                    (1_000_000_010, 0.2, 0.0, 0.5),
                    (1_000_000_020, 0.1, 0.2, 0.6),
                    (1_000_000_030, 4.0, 4.0, 0.4),
                    (1_000_000_040, 4.2, 4.1, 0.5),
                    (1_000_000_050, 4.1, 4.2, 0.7),
                ],
            )
            self._write_frame(
                frame2,
                [
                    (2_000_000_000, 0.5, 0.2, 0.5),
                    (2_000_000_010, 0.6, 0.1, 0.6),
                    (2_000_000_020, 0.7, 0.3, 0.7),
                ],
            )
            zones.write_text(
                json.dumps(
                    {
                        "zones": [
                            {
                                "zone_id": "near-origin",
                                "name": "Near Origin",
                                "min_xy": [-1.0, -1.0],
                                "max_xy": [1.5, 1.5],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary = run_replay(
                ReplayConfig(
                    input_dir=root,
                    output_json=root / "summary.json",
                    mode="mobile",
                    zone_config=zones,
                    voxel_size_m=0.0,
                    min_range_m=0.0,
                    max_range_m=100.0,
                    z_min_m=-10.0,
                    z_max_m=10.0,
                    cluster_cell_size_m=1.0,
                    min_cluster_points=2,
                    min_cluster_cells=1,
                    min_cluster_height_m=0.0,
                    max_cluster_height_m=10.0,
                    max_cluster_width_m=10.0,
                    track_max_distance_m=2.0,
                )
            )

            self.assertEqual(summary["frames_processed"], 2)
            self.assertEqual(summary["metrics"]["total_tracks"], 2)
            self.assertEqual(summary["pipeline"], "scene_replay")
            self.assertEqual(summary["metrics"]["detector_name"], "baseline")
            self.assertIn("scene_bounds", summary)
            self.assertIn("playback", summary)
            self.assertEqual(len(summary["zone_definitions"]), 1)
            self.assertEqual(len(summary["frame_summaries"][0]["preview_points"]), 6)
            self.assertTrue(summary["frame_summaries"][0]["active_tracks"])
            self.assertIn("scene_state", summary["frame_summaries"][0])
            self.assertEqual(
                summary["frame_summaries"][0]["scene_state"]["zones"][0]["object_count"], 1
            )
            self.assertEqual(
                summary["frame_summaries"][0]["detections"][0]["source"], "baseline"
            )

    def _write_frame(
        self, path: Path, rows: list[tuple[int, float, float, float]]
    ) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["TIMESTAMP (ns)", "X1 (m)", "Y1 (m)", "Z1 (m)"])
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
