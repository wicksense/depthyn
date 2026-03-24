from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from depthyn.models import Detection, FramePose
from depthyn.pose import discover_gps_csv, load_gps_pose_provider, transform_detection


class GpsPoseTests(unittest.TestCase):
    def test_discover_single_raw_gps_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "raw_gps_20260317_094022.csv").write_text("pi_time_ns,easting,northing\n")
            self.assertEqual(
                discover_gps_csv(root),
                root / "raw_gps_20260317_094022.csv",
            )

    def test_pose_interpolates_between_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw_gps.csv"
            path.write_text(
                "\n".join(
                    [
                        "pi_time_ns,pi_time_iso,gps_epoch_ns,gps_utc_time,gps_quality,num_sats,hdop,latitude,longitude,altitude_m,easting,northing",
                        "0,1970-01-01T00:00:00+00:00,0,00:00:00+00:00,1,10,1.0,0,0,0,0,0",
                        "1000000000,1970-01-01T00:00:01+00:00,1000000000,00:00:01+00:00,1,10,1.0,0,0,0,0,10",
                    ]
                ),
                encoding="utf-8",
            )
            provider = load_gps_pose_provider(path)
            pose = provider.pose_at(500_000_000)
            self.assertEqual(pose.position_m, (0.0, 5.0, 0.0))
            self.assertAlmostEqual(pose.heading_rad, math.pi / 2, places=6)

    def test_transform_detection_rotates_heading_and_centroid(self) -> None:
        detection = Detection(
            detection_id="d1",
            centroid=(2.0, 0.0, 0.5),
            bbox_min=(1.0, -1.0, 0.0),
            bbox_max=(3.0, 1.0, 1.0),
            point_count=12,
            cell_count=4,
            heading_rad=0.1,
        )
        pose = FramePose(
            timestamp_ns=1,
            position_m=(10.0, 5.0, 0.0),
            heading_rad=math.pi / 2,
        )
        transformed = transform_detection(detection, pose)
        self.assertAlmostEqual(transformed.centroid[0], 10.0, places=6)
        self.assertAlmostEqual(transformed.centroid[1], 7.0, places=6)
        self.assertAlmostEqual(transformed.heading_rad, math.pi / 2 + 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
