"""Tests for the ONNX CenterPoint detector backend.

Unit tests cover voxelization, feature generation, box decoding, and NMS
without requiring ONNX models.  The integration test (marked with
unittest.skipUnless) runs real inference when models are available.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from depthyn.config import DetectorConfig
from depthyn.detectors.base import DetectorUnavailableError

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "centerpoint-onnx"
MODELS_AVAILABLE = (
    MODEL_DIR.is_dir()
    and (MODEL_DIR / "pts_voxel_encoder_centerpoint.onnx").is_file()
    and (MODEL_DIR / "pts_backbone_neck_head_centerpoint.onnx").is_file()
)

try:
    import onnxruntime  # noqa: F401

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class VoxelizationTests(unittest.TestCase):
    def test_empty_points_return_zero_voxels(self) -> None:
        from depthyn.detectors.onnx_centerpoint import _voxelize

        points = np.zeros((0, 4), dtype=np.float32)
        features, coords, num_voxels = _voxelize(points, np)
        self.assertEqual(num_voxels, 0)
        self.assertEqual(features.shape[0], 0)

    def test_single_point_produces_one_voxel(self) -> None:
        from depthyn.detectors.onnx_centerpoint import _voxelize

        points = np.array([[5.0, 5.0, 1.0, 0.0]], dtype=np.float32)
        features, coords, num_voxels = _voxelize(points, np)
        self.assertEqual(num_voxels, 1)
        # First 3 features should be the point coords.
        self.assertAlmostEqual(features[0, 0, 0], 5.0, places=3)
        self.assertAlmostEqual(features[0, 0, 1], 5.0, places=3)
        self.assertAlmostEqual(features[0, 0, 2], 1.0, places=3)

    def test_out_of_range_points_filtered(self) -> None:
        from depthyn.detectors.onnx_centerpoint import _voxelize

        points = np.array(
            [[200.0, 200.0, 0.0, 0.0], [-200.0, -200.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        features, coords, num_voxels = _voxelize(points, np)
        self.assertEqual(num_voxels, 0)

    def test_nine_features_generated(self) -> None:
        from depthyn.detectors.onnx_centerpoint import _voxelize, ENCODER_IN_FEATURES

        points = np.array(
            [[10.0, 10.0, 1.0, 0.0], [10.1, 10.1, 1.1, 0.0]],
            dtype=np.float32,
        )
        features, coords, num_voxels = _voxelize(points, np)
        self.assertGreater(num_voxels, 0)
        self.assertEqual(features.shape[2], ENCODER_IN_FEATURES)


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class DecodeTests(unittest.TestCase):
    def test_decode_empty_heatmap(self) -> None:
        from depthyn.detectors.onnx_centerpoint import _decode_boxes

        outputs = {
            "heatmap": np.full((1, 5, 4, 4), -10.0, dtype=np.float32),
            "reg": np.zeros((1, 2, 4, 4), dtype=np.float32),
            "height": np.zeros((1, 1, 4, 4), dtype=np.float32),
            "dim": np.zeros((1, 3, 4, 4), dtype=np.float32),
            "rot": np.zeros((1, 2, 4, 4), dtype=np.float32),
            "vel": np.zeros((1, 2, 4, 4), dtype=np.float32),
        }
        boxes = _decode_boxes(outputs, np)
        self.assertEqual(len(boxes), 0)

    def test_decode_single_detection(self) -> None:
        from depthyn.detectors.onnx_centerpoint import _decode_boxes

        h, w = 4, 4
        outputs = {
            "heatmap": np.full((1, 5, h, w), -10.0, dtype=np.float32),
            "reg": np.zeros((1, 2, h, w), dtype=np.float32),
            "height": np.zeros((1, 1, h, w), dtype=np.float32),
            "dim": np.zeros((1, 3, h, w), dtype=np.float32),
            "rot": np.zeros((1, 2, h, w), dtype=np.float32),
            "vel": np.zeros((1, 2, h, w), dtype=np.float32),
        }
        # Place a strong car detection at grid cell (2, 2).
        outputs["heatmap"][0, 0, 2, 2] = 5.0  # sigmoid(5) ≈ 0.993
        # Set rotation to valid norm.
        outputs["rot"][0, 0, 2, 2] = 0.0  # sin
        outputs["rot"][0, 1, 2, 2] = 1.0  # cos
        # Set dim to something reasonable.
        outputs["dim"][0, 0, 2, 2] = math.log(4.0)  # width
        outputs["dim"][0, 1, 2, 2] = math.log(1.8)  # length
        outputs["dim"][0, 2, 2, 2] = math.log(1.5)  # height

        boxes = _decode_boxes(outputs, np)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["cls_id"], 0)  # car
        self.assertGreater(boxes[0]["score"], 0.9)


@unittest.skipUnless(HAS_NUMPY, "numpy not available")
class CircleNMSTests(unittest.TestCase):
    def test_suppresses_nearby_box(self) -> None:
        from depthyn.detectors.onnx_centerpoint import _circle_nms

        boxes = [
            {"cx": 0.0, "cy": 0.0, "score": 0.9},
            {"cx": 0.1, "cy": 0.1, "score": 0.8},
            {"cx": 10.0, "cy": 10.0, "score": 0.7},
        ]
        kept = _circle_nms(boxes)
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(kept[0]["score"], 0.9)
        self.assertAlmostEqual(kept[1]["score"], 0.7)


class FactoryTests(unittest.TestCase):
    @unittest.skipUnless(HAS_NUMPY and HAS_ORT, "numpy or onnxruntime not available")
    def test_factory_creates_onnx_detector(self) -> None:
        from depthyn.config import ReplayConfig
        from depthyn.detectors.factory import create_detector

        if not MODELS_AVAILABLE:
            self.skipTest("ONNX models not downloaded")

        detector = create_detector(
            ReplayConfig(
                input_dir=Path("."),
                output_json=Path("summary.json"),
                detector=DetectorConfig(kind="centerpoint-onnx"),
            )
        )
        self.assertEqual(detector.name, "centerpoint-onnx")
        self.assertEqual(detector.input_mode, "full")

    def test_factory_errors_without_deps(self) -> None:
        """If models are missing, factory should raise DetectorUnavailableError."""
        from depthyn.config import ReplayConfig
        from depthyn.detectors.factory import create_detector

        if MODELS_AVAILABLE and HAS_NUMPY and HAS_ORT:
            self.skipTest("Models and deps are available, nothing to test")

        with self.assertRaises(DetectorUnavailableError):
            create_detector(
                ReplayConfig(
                    input_dir=Path("."),
                    output_json=Path("summary.json"),
                    detector=DetectorConfig(kind="centerpoint-onnx"),
                )
            )


@unittest.skipUnless(
    HAS_NUMPY and HAS_ORT and MODELS_AVAILABLE,
    "requires numpy, onnxruntime, and downloaded ONNX models",
)
class IntegrationTests(unittest.TestCase):
    def test_detect_with_real_model(self) -> None:
        from depthyn.detectors.onnx_centerpoint import OnnxCenterPointDetector
        from depthyn.models import Frame

        detector = OnnxCenterPointDetector(DetectorConfig(kind="centerpoint-onnx"))
        frame = Frame(
            frame_id="test_001",
            timestamp_ns=1,
            points=[(5.0, 5.0, 1.0)] * 100,
            source_path=Path("."),
        )
        result = detector.detect(frame, frame.points)
        self.assertIsInstance(result.detections, list)
        self.assertEqual(result.input_point_count, 100)
        self.assertIn("voxels", result.metadata)


if __name__ == "__main__":
    unittest.main()
