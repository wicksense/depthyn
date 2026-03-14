from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_runner_module():
    runner_path = Path(__file__).resolve().parents[1] / "tools" / "mmdet3d_runner.py"
    spec = importlib.util.spec_from_file_location("depthyn_mmdet3d_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load mmdet3d_runner module for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MMDet3DRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner_module()

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") is not None,
        "numpy is only required in the external MMDetection3D environment",
    )
    def test_ensure_model_feature_count_adds_zero_sweep_column(self) -> None:
        import numpy as np

        point_cloud = np.asarray(
            [
                [1.0, 2.0, 3.0, 0.5],
                [4.0, 5.0, 6.0, 0.7],
            ],
            dtype=np.float32,
        )

        normalized = self.runner._ensure_model_feature_count(point_cloud, np)

        self.assertEqual(normalized.shape, (2, 5))
        self.assertEqual(normalized[0, 4], 0.0)
        self.assertEqual(normalized[1, 4], 0.0)

    def test_patch_config_for_depthyn_removes_multi_sweeps_and_sets_dims(self) -> None:
        config = {
            "test_dataloader": {
                "dataset": {
                    "pipeline": [
                        {"type": "LoadPointsFromFile", "load_dim": 5, "use_dim": 5},
                        {"type": "LoadPointsFromMultiSweeps", "sweeps_num": 9},
                        {"type": "Pack3DDetInputs", "keys": ["points"]},
                    ]
                }
            }
        }

        self.runner._patch_config_for_depthyn(config, load_dim=5, use_dim=5)

        pipeline = config["test_dataloader"]["dataset"]["pipeline"]
        self.assertEqual(len(pipeline), 2)
        self.assertEqual(pipeline[0]["type"], "LoadPointsFromFile")
        self.assertEqual(pipeline[0]["load_dim"], 5)
        self.assertEqual(pipeline[0]["use_dim"], [0, 1, 2, 3, 4])
        self.assertEqual(pipeline[1]["type"], "Pack3DDetInputs")


if __name__ == "__main__":
    unittest.main()
