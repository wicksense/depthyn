from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from depthyn.ml_prep import export_ml_replay_bundle


class MLReplayPrepTests(unittest.TestCase):
    def test_export_ml_replay_bundle_writes_manifest_and_bins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_path = root / "frame_0001.csv"
            output_dir = root / "ml-export"
            self._write_frame(
                frame_path,
                [
                    (1_000_000_000, 1.0, 0.0, 0.5),
                    (1_000_000_010, 1.2, 0.2, 0.6),
                ],
            )

            manifest = export_ml_replay_bundle(
                input_dir=root,
                output_dir=output_dir,
                max_frames=None,
                voxel_size_m=0.0,
                min_range_m=0.0,
                max_range_m=100.0,
                z_min_m=-10.0,
                z_max_m=10.0,
                default_intensity=0.25,
            )

            self.assertEqual(manifest["frame_count"], 1)
            self.assertEqual(manifest["point_format"], "xyzi_float32_le")
            frame_entry = manifest["frames"][0]
            point_path = output_dir / frame_entry["points_path"]
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue(point_path.exists())
            self.assertEqual(point_path.stat().st_size, 2 * 4 * 4)

    def _write_frame(
        self, path: Path, rows: list[tuple[int, float, float, float]]
    ) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["TIMESTAMP (ns)", "X1 (m)", "Y1 (m)", "Z1 (m)"])
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
