from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from depthyn.detectors.base import DetectorResult
from depthyn.config import ReplayConfig
from depthyn.models import Detection, Frame
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

    def test_stationary_mode_can_run_full_detector_on_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame1 = root / "frame_0001.csv"
            frame2 = root / "frame_0002.csv"
            self._write_frame(
                frame1,
                [
                    (1_000_000_000, 0.0, 0.0, 0.5),
                ],
            )
            self._write_frame(
                frame2,
                [
                    (2_000_000_000, 0.0, 0.0, 0.5),
                    (2_000_000_010, 5.0, 5.0, 0.6),
                ],
            )

            fake_detector = _RecordingDetector(input_mode="full")

            with patch("depthyn.pipeline.create_detector", return_value=fake_detector):
                summary = run_replay(
                    ReplayConfig(
                        input_dir=root,
                        output_json=root / "summary.json",
                        mode="stationary",
                        detector_on_foreground=True,
                        voxel_size_m=0.0,
                        min_range_m=0.0,
                        max_range_m=100.0,
                        z_min_m=-10.0,
                        z_max_m=10.0,
                        cluster_cell_size_m=1.0,
                        background_warmup_frames=1,
                        background_min_hits=1,
                        background_fade_time_s=0.0,
                    )
                )

            self.assertEqual(fake_detector.input_sizes, [1])
            self.assertEqual(
                summary["frame_summaries"][1]["detector_input_points"],
                1,
            )

    def test_stationary_mode_keeps_full_scene_for_full_detector_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame1 = root / "frame_0001.csv"
            frame2 = root / "frame_0002.csv"
            self._write_frame(
                frame1,
                [
                    (1_000_000_000, 0.0, 0.0, 0.5),
                ],
            )
            self._write_frame(
                frame2,
                [
                    (2_000_000_000, 0.0, 0.0, 0.5),
                    (2_000_000_010, 5.0, 5.0, 0.6),
                ],
            )

            fake_detector = _RecordingDetector(input_mode="full")

            with patch("depthyn.pipeline.create_detector", return_value=fake_detector):
                summary = run_replay(
                    ReplayConfig(
                        input_dir=root,
                        output_json=root / "summary.json",
                        mode="stationary",
                        voxel_size_m=0.0,
                        min_range_m=0.0,
                        max_range_m=100.0,
                        z_min_m=-10.0,
                        z_max_m=10.0,
                        cluster_cell_size_m=1.0,
                        background_warmup_frames=1,
                        background_min_hits=1,
                        background_fade_time_s=0.0,
                    )
                )

            self.assertEqual(fake_detector.input_sizes, [1, 2])
            self.assertEqual(
                summary["frame_summaries"][1]["detector_input_points"],
                2,
            )

    def test_world_aligned_replay_transforms_points_and_detections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame1 = root / "frame_0001.csv"
            gps_dir = root / "gps"
            gps_dir.mkdir()
            gps_path = gps_dir / "raw_gps_test.csv"

            self._write_frame(
                frame1,
                [
                    (1_000_000_000, 1.0, 0.0, 0.5),
                    (1_000_000_010, 2.0, 0.0, 0.5),
                ],
            )
            gps_path.write_text(
                "\n".join(
                    [
                        "pi_time_ns,pi_time_iso,gps_epoch_ns,gps_utc_time,gps_quality,num_sats,hdop,latitude,longitude,altitude_m,easting,northing",
                        "0,1970-01-01T00:00:00+00:00,0,00:00:00+00:00,1,10,1.0,0,0,0,0,0",
                        "2000000000,1970-01-01T00:00:02+00:00,2000000000,00:00:02+00:00,1,10,1.0,0,0,0,20,0",
                    ]
                ),
                encoding="utf-8",
            )

            fake_detector = _RecordingDetector(
                input_mode="full",
                detections=[
                    Detection(
                        detection_id="det-1",
                        centroid=(2.0, 0.0, 0.5),
                        bbox_min=(1.0, -0.5, 0.0),
                        bbox_max=(3.0, 0.5, 1.0),
                        point_count=2,
                        cell_count=1,
                        label="car",
                        score=0.9,
                        heading_rad=0.0,
                    )
                ],
            )

            with patch("depthyn.pipeline.create_detector", return_value=fake_detector):
                summary = run_replay(
                    ReplayConfig(
                        input_dir=root,
                        output_json=root / "summary.json",
                        mode="mobile",
                        source_type="csv",
                        world_align=True,
                        gps_path=gps_path,
                        voxel_size_m=0.0,
                        min_range_m=0.0,
                        max_range_m=100.0,
                        z_min_m=-10.0,
                        z_max_m=10.0,
                    )
                )

            self.assertEqual(summary["reference_frame"], "world")
            self.assertEqual(summary["available_reference_frames"], ["sensor", "world"])
            frame_summary = summary["frame_summaries"][0]
            self.assertEqual(frame_summary["frame_pose"]["position_m"], [10.0, 0.0, 0.0])
            self.assertAlmostEqual(frame_summary["sensor_preview_points"][0][0], 1.0, places=5)
            self.assertAlmostEqual(frame_summary["sensor_detections"][0]["centroid"][0], 2.0, places=5)
            self.assertAlmostEqual(frame_summary["sensor_active_tracks"][0]["centroid"][0], 2.0, places=5)
            self.assertAlmostEqual(frame_summary["preview_points"][0][0], 11.0, places=5)
            self.assertAlmostEqual(frame_summary["preview_points"][0][1], 0.0, places=5)
            self.assertAlmostEqual(frame_summary["detections"][0]["centroid"][0], 12.0, places=5)
            self.assertAlmostEqual(frame_summary["active_tracks"][0]["centroid"][0], 12.0, places=5)

    def test_replay_carries_scanline_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_frame = Frame(
                frame_id="pcap_fake_000001",
                timestamp_ns=1_000_000_000,
                points=[(1.0, 0.0, 0.5)],
                source_path=root / "capture.pcap",
                scanline_shape=(4, 8),
                scanline_points=[(0, 1, 1.0, 0.0, 0.5, 5.0)],
                scanline_pixel_shift_by_row=[0, 1, 2, 3],
            )
            fake_detector = _RecordingDetector(input_mode="full")

            with patch("depthyn.pipeline._stream_frames", return_value=iter([fake_frame])):
                with patch("depthyn.pipeline.create_detector", return_value=fake_detector):
                    summary = run_replay(
                        ReplayConfig(
                            input_dir=root,
                            output_json=root / "summary.json",
                            source_type="pcap",
                            preview_point_limit=10,
                            detail_point_limit=10,
                        )
                    )

            self.assertEqual(
                summary["scanline_metadata"],
                {
                    "shape": [4, 8],
                    "pixel_shift_by_row": [0, 1, 2, 3],
                },
            )
            self.assertEqual(summary["frame_summaries"][0]["scanline_shape"], [4, 8])
            self.assertEqual(
                summary["frame_summaries"][0]["scanline_points"],
                [[0, 1, 1.0, 0.0, 0.5, 5.0]],
            )

    def _write_frame(
        self, path: Path, rows: list[tuple[int, float, float, float]]
    ) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["TIMESTAMP (ns)", "X1 (m)", "Y1 (m)", "Z1 (m)"])
            writer.writerows(rows)

class _RecordingDetector:
    name = "recording"

    def __init__(self, input_mode: str, detections=None) -> None:
        self.input_mode = input_mode
        self.input_sizes: list[int] = []
        self._detections = detections or []

    def detect(self, frame, points):
        self.input_sizes.append(len(points))
        return DetectorResult(
            detections=list(self._detections),
            input_point_count=len(points),
            metadata={"input_size": len(points)},
        )


if __name__ == "__main__":
    unittest.main()
