from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from depthyn.comparison import run_detector_comparison
from depthyn.config import DetectorConfig, ReplayConfig
from depthyn.detectors.base import DetectorUnavailableError
from depthyn.detectors.factory import create_detector


class DetectorTests(unittest.TestCase):
    def test_factory_builds_baseline_detector(self) -> None:
        detector = create_detector(
            ReplayConfig(
                input_dir=Path("."),
                output_json=Path("summary.json"),
            )
        )
        self.assertEqual(detector.name, "baseline")
        self.assertEqual(detector.input_mode, "foreground")

    def test_factory_rejects_unknown_detector(self) -> None:
        with self.assertRaises(DetectorUnavailableError):
            create_detector(
                ReplayConfig(
                    input_dir=Path("."),
                    output_json=Path("summary.json"),
                    detector=DetectorConfig(kind="mystery"),
                )
            )

    def test_comparison_reports_missing_openpcdet_config_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame = root / "frame_0001.csv"
            frame.write_text(
                "TIMESTAMP (ns),X1 (m),Y1 (m),Z1 (m)\n"
                "1,0.0,0.0,0.5\n"
                "2,0.2,0.1,0.6\n"
                "3,0.3,0.2,0.7\n",
                encoding="utf-8",
            )

            comparison = run_detector_comparison(
                ReplayConfig(
                    input_dir=root,
                    output_json=root / "placeholder.json",
                    mode="mobile",
                    voxel_size_m=0.0,
                    min_range_m=0.0,
                    max_range_m=100.0,
                    z_min_m=-10.0,
                    z_max_m=10.0,
                    min_cluster_points=1,
                    min_cluster_cells=1,
                    min_cluster_height_m=0.0,
                    max_cluster_height_m=10.0,
                    max_cluster_width_m=10.0,
                ),
                [
                    DetectorConfig(kind="baseline"),
                    DetectorConfig(kind="pointpillars"),
                ],
                root / "comparison",
            )

            self.assertEqual(len(comparison["detector_runs"]), 2)
            self.assertEqual(comparison["detector_runs"][0]["status"], "ok")
            self.assertEqual(comparison["detector_runs"][1]["status"], "error")
            self.assertIn("requires --openpcdet-repo", comparison["detector_runs"][1]["error"])


if __name__ == "__main__":
    unittest.main()
