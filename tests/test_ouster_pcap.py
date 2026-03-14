"""Tests for the Ouster pcap source adapter."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from depthyn.source.ouster_pcap import (
    discover_ouster_pcap_files,
    find_metadata_json,
)


class DiscoverTests(unittest.TestCase):
    def test_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            discover_ouster_pcap_files(Path("/nonexistent"))

    def test_empty_directory(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                discover_ouster_pcap_files(Path(tmp))


class MetadataDiscoveryTests(unittest.TestCase):
    def test_finds_chunk_metadata(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pcap = tmp_path / "raw_lidar_20260226_101550_chunk0001.pcap"
            pcap.touch()
            meta = tmp_path / "raw_lidar_20260226_101550_0.json"
            meta.write_text("{}")
            result = find_metadata_json(pcap)
            self.assertEqual(result, meta)

    def test_finds_same_name_metadata(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pcap = tmp_path / "capture.pcap"
            pcap.touch()
            meta = tmp_path / "capture.json"
            meta.write_text("{}")
            result = find_metadata_json(pcap)
            self.assertEqual(result, meta)

    def test_finds_metadata_json_fallback(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pcap = tmp_path / "data.pcap"
            pcap.touch()
            meta = tmp_path / "metadata.json"
            meta.write_text("{}")
            result = find_metadata_json(pcap)
            self.assertEqual(result, meta)

    def test_missing_metadata_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pcap = tmp_path / "data.pcap"
            pcap.touch()
            with self.assertRaises(FileNotFoundError):
                find_metadata_json(pcap)


class SourceTypeAutoDetectTests(unittest.TestCase):
    def test_csv_auto_detect(self):
        from depthyn.pipeline import _resolve_source_type
        from depthyn.config import ReplayConfig

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "frame_000.csv").touch()
            config = ReplayConfig(
                input_dir=tmp_path,
                output_json=tmp_path / "out.json",
                source_type="auto",
            )
            self.assertEqual(_resolve_source_type(config), "csv")

    def test_pcap_auto_detect(self):
        from depthyn.pipeline import _resolve_source_type
        from depthyn.config import ReplayConfig

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "capture.pcap").touch()
            config = ReplayConfig(
                input_dir=tmp_path,
                output_json=tmp_path / "out.json",
                source_type="auto",
            )
            self.assertEqual(_resolve_source_type(config), "pcap")

    def test_explicit_source_type(self):
        from depthyn.pipeline import _resolve_source_type
        from depthyn.config import ReplayConfig

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = ReplayConfig(
                input_dir=tmp_path,
                output_json=tmp_path / "out.json",
                source_type="csv",
            )
            self.assertEqual(_resolve_source_type(config), "csv")


SAMPLE_PCAP_DIR = Path("SampleData/26")


@unittest.skipUnless(
    SAMPLE_PCAP_DIR.exists() and list(SAMPLE_PCAP_DIR.glob("*.pcap")),
    "Sample pcap data not available",
)
class IntegrationTests(unittest.TestCase):
    def test_discover_sample_pcaps(self):
        pcaps = discover_ouster_pcap_files(SAMPLE_PCAP_DIR)
        self.assertGreater(len(pcaps), 0)
        for p in pcaps:
            self.assertTrue(p.suffix == ".pcap")

    def test_find_sample_metadata(self):
        pcaps = discover_ouster_pcap_files(SAMPLE_PCAP_DIR)
        meta = find_metadata_json(pcaps[0])
        self.assertTrue(meta.exists())

    def test_load_frames_from_pcap(self):
        from depthyn.source.ouster_pcap import iter_ouster_pcap_frames

        frames = list(
            iter_ouster_pcap_frames(
                SAMPLE_PCAP_DIR,
                max_frames=3,
                voxel_size_m=0.3,
                min_range_m=1.0,
                max_range_m=60.0,
                z_min_m=-2.5,
                z_max_m=4.5,
            )
        )
        self.assertEqual(len(frames), 3)
        for frame in frames:
            self.assertGreater(len(frame.points), 1000)
            self.assertGreater(frame.timestamp_ns, 0)
            self.assertTrue(frame.frame_id.startswith("pcap_"))
            # Verify points are within bounds
            for x, y, z in frame.points[:10]:
                self.assertGreaterEqual(z, -2.5)
                self.assertLessEqual(z, 4.5)

    def test_pcap_raw_resolution(self):
        """Pcap without voxel downsampling should produce high-density frames."""
        from depthyn.source.ouster_pcap import iter_ouster_pcap_frames

        frames = list(
            iter_ouster_pcap_frames(
                SAMPLE_PCAP_DIR, max_frames=1, voxel_size_m=0,
            )
        )
        # OS-1-128 at 1024x10 mode should yield tens of thousands of valid
        # points per frame even after range/Z filtering.
        self.assertGreater(
            len(frames[0].points), 30000,
            "raw pcap frame should have >30k points without downsampling",
        )


if __name__ == "__main__":
    unittest.main()
