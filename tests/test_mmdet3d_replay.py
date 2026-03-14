from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from depthyn.mmdet3d_replay import (
    run_mmdet3d_manifest_inference,
    run_stage1_mmdet3d_compare,
)


class MMDet3DReplayTests(unittest.TestCase):
    def test_run_mmdet3d_manifest_inference_reads_normalized_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            output_path = root / "predictions.json"
            config_path = root / "centerpoint.py"
            checkpoint_path = root / "centerpoint.pth"

            manifest_path.write_text(json.dumps({"frames": []}), encoding="utf-8")
            config_path.write_text("# config placeholder\n", encoding="utf-8")
            checkpoint_path.write_text("checkpoint placeholder\n", encoding="utf-8")

            def fake_run(command, capture_output, text, check):  # noqa: ANN001
                self.assertIn("--manifest-json", command)
                self.assertIn(str(manifest_path), command)
                self.assertIn("--model-name", command)
                output_path.write_text(
                    json.dumps(
                        {
                            "frame_predictions": [
                                {"frame_id": "frame_0001", "detections": []}
                            ],
                            "frames_processed": 1,
                            "total_detections": 0,
                        }
                    ),
                    encoding="utf-8",
                )

                class Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Result()

            with patch("depthyn.mmdet3d_replay.subprocess.run", side_effect=fake_run):
                payload = run_mmdet3d_manifest_inference(
                    manifest_path=manifest_path,
                    output_path=output_path,
                    backend_python="python3",
                    backend_repo=None,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    score_threshold=0.35,
                    model_name="centerpoint",
                    device="cuda:0",
                )

            self.assertEqual(payload["frames_processed"], 1)
            self.assertEqual(payload["total_detections"], 0)

    def test_run_stage1_mmdet3d_compare_uses_precomputed_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_path = root / "frame_0001.csv"
            output_dir = root / "stage1b"
            config_path = root / "centerpoint.py"
            checkpoint_path = root / "centerpoint.pth"

            self._write_frame(
                frame_path,
                [
                    (1_000_000_000, 1.0, 0.0, 0.5),
                    (1_000_000_010, 1.2, 0.2, 0.6),
                ],
            )
            config_path.write_text("# config placeholder\n", encoding="utf-8")
            checkpoint_path.write_text("checkpoint placeholder\n", encoding="utf-8")

            def fake_manifest_inference(**kwargs):  # noqa: ANN003
                output_path = kwargs["output_path"]
                output_path.write_text(
                    json.dumps(
                        {
                            "frame_predictions": [
                                {
                                    "frame_id": "frame_0001",
                                    "detections": [
                                        {
                                            "detection_id": "centerpoint-0001",
                                            "centroid": [6.0, 4.0, 1.0],
                                            "bbox_min": [5.5, 3.5, 0.4],
                                            "bbox_max": [6.5, 4.5, 1.8],
                                            "label": "vehicle",
                                            "score": 0.93,
                                        }
                                    ],
                                }
                            ],
                            "frames_processed": 1,
                            "total_detections": 1,
                        }
                    ),
                    encoding="utf-8",
                )
                return {"frames_processed": 1, "total_detections": 1}

            with patch(
                "depthyn.mmdet3d_replay.run_mmdet3d_manifest_inference",
                side_effect=fake_manifest_inference,
            ):
                result = run_stage1_mmdet3d_compare(
                    input_dir=root,
                    output_dir=output_dir,
                    mode="mobile",
                    zone_config=None,
                    max_frames=1,
                    preview_point_limit=200,
                    voxel_size_m=0.0,
                    cluster_cell_size_m=1.0,
                    track_max_distance_m=2.0,
                    min_range_m=0.0,
                    max_range_m=100.0,
                    z_min_m=-10.0,
                    z_max_m=10.0,
                    default_intensity=0.0,
                    backend_python="python3",
                    backend_repo=None,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    score_threshold=0.25,
                    model_name="centerpoint",
                    device="cuda:0",
                )

            comparison = result["comparison"]
            self.assertEqual(comparison["detector_runs"][0]["status"], "ok")
            self.assertEqual(comparison["detector_runs"][1]["status"], "ok")
            self.assertEqual(
                comparison["detector_runs"][1]["detector"]["label"], "centerpoint"
            )
            self.assertEqual(
                comparison["detector_runs"][1]["metrics"]["label_counts"],
                {"vehicle": 1},
            )
            self.assertTrue((output_dir / "ml-replay" / "manifest.json").exists())
            self.assertTrue((output_dir / "centerpoint-predictions.json").exists())

    def _write_frame(
        self, path: Path, rows: list[tuple[int, float, float, float]]
    ) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["TIMESTAMP (ns)", "X1 (m)", "Y1 (m)", "Z1 (m)"])
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
