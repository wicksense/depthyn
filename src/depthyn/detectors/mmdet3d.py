from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from depthyn.config import DetectorConfig
from depthyn.detectors.base import DetectorResult, DetectorUnavailableError
from depthyn.models import Detection, Frame


class MMDet3DDetector:
    input_mode = "full"

    def __init__(self, detector_config: DetectorConfig) -> None:
        self.name = detector_config.kind
        self._config = detector_config

    def detect(self, frame: Frame, points: list[tuple[float, float, float]]) -> DetectorResult:
        self._validate()

        runner_path = Path(__file__).resolve().parents[3] / "tools" / "mmdet3d_runner.py"
        with tempfile.TemporaryDirectory(prefix="depthyn-mmdet3d-") as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "frame.json"
            output_path = temp_root / "predictions.json"
            input_payload = {
                "frame_id": frame.frame_id,
                "timestamp_ns": frame.timestamp_ns,
                "points_xyz": [list(point) for point in points],
                "default_intensity": 0.0,
            }
            input_path.write_text(json.dumps(input_payload), encoding="utf-8")

            command = [
                self._config.backend_python or sys.executable,
                str(runner_path),
                "--config",
                str(self._config.config_path),
                "--checkpoint",
                str(self._config.checkpoint_path),
                "--input-json",
                str(input_path),
                "--output-json",
                str(output_path),
                "--score-threshold",
                str(self._config.score_threshold),
                "--model-name",
                self.name,
                "--device",
                self._config.device,
            ]
            if self._config.backend_repo is not None:
                command.extend(["--mmdet3d-repo", str(self._config.backend_repo)])

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                details = completed.stderr.strip() or completed.stdout.strip()
                raise DetectorUnavailableError(
                    f"{self.name} detector execution failed: {details}"
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            detections = [
                Detection(
                    detection_id=item["detection_id"],
                    centroid=tuple(item["centroid"]),
                    bbox_min=tuple(item["bbox_min"]),
                    bbox_max=tuple(item["bbox_max"]),
                    point_count=int(item.get("point_count", 0)),
                    cell_count=int(item.get("cell_count", 0)),
                    label=item.get("label", "object"),
                    score=item.get("score"),
                    source=self.name,
                    heading_rad=item.get("heading_rad"),
                )
                for item in payload.get("detections", [])
            ]
            return DetectorResult(
                detections=detections,
                input_point_count=len(points),
                metadata={
                    "backend": "mmdet3d",
                    "config": str(self._config.config_path),
                    "checkpoint": str(self._config.checkpoint_path),
                    "stdout": payload.get("stdout", ""),
                },
            )

    def _validate(self) -> None:
        if self._config.config_path is None:
            raise DetectorUnavailableError(
                f"{self.name} requires a model config path."
            )
        if self._config.checkpoint_path is None:
            raise DetectorUnavailableError(
                f"{self.name} requires a model checkpoint path."
            )
