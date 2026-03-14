from __future__ import annotations

import json
from pathlib import Path

from depthyn.config import DetectorConfig
from depthyn.detectors.base import DetectorResult, DetectorUnavailableError
from depthyn.models import Detection, Frame


class PrecomputedDetector:
    input_mode = "full"

    def __init__(self, detector_config: DetectorConfig) -> None:
        self.name = detector_config.resolved_label()
        self._config = detector_config
        self._predictions_by_frame = self._load_predictions()

    def detect(self, frame: Frame, points: list[tuple[float, float, float]]) -> DetectorResult:
        raw_detections = self._predictions_by_frame.get(frame.frame_id, [])
        detections = [
            _normalize_detection(item, detection_index=index, source_name=self.name)
            for index, item in enumerate(raw_detections, start=1)
        ]
        return DetectorResult(
            detections=detections,
            input_point_count=len(points),
            metadata={
                "backend": "precomputed",
                "prediction_path": str(self._config.prediction_path),
            },
        )

    def _load_predictions(self) -> dict[str, list[dict[str, object]]]:
        prediction_path = self._config.prediction_path
        if prediction_path is None:
            raise DetectorUnavailableError(
                f"{self.name} requires a prediction path."
            )
        if not prediction_path.exists():
            raise DetectorUnavailableError(
                f"{self.name} prediction path does not exist: {prediction_path}"
            )

        if prediction_path.is_dir():
            predictions: dict[str, list[dict[str, object]]] = {}
            for child in sorted(prediction_path.glob("*.json")):
                payload = json.loads(child.read_text(encoding="utf-8"))
                detections = payload.get("detections", payload)
                if not isinstance(detections, list):
                    raise DetectorUnavailableError(
                        f"{self.name} prediction file must contain a detection list: {child}"
                    )
                predictions[child.stem] = detections
            return predictions

        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "frames" in payload:
                raw_frames = payload["frames"]
                if not isinstance(raw_frames, list):
                    raise DetectorUnavailableError(
                        f"{self.name} prediction file 'frames' value must be a list."
                    )
                return _frames_to_mapping(raw_frames)
            if "frame_predictions" in payload:
                raw_frames = payload["frame_predictions"]
                if not isinstance(raw_frames, list):
                    raise DetectorUnavailableError(
                        f"{self.name} prediction file 'frame_predictions' value must be a list."
                    )
                return _frames_to_mapping(raw_frames)
            if all(isinstance(value, list) for value in payload.values()):
                return {str(key): value for key, value in payload.items()}

        raise DetectorUnavailableError(
            f"{self.name} prediction file format is not supported: {prediction_path}"
        )


def _frames_to_mapping(raw_frames: list[object]) -> dict[str, list[dict[str, object]]]:
    mapping: dict[str, list[dict[str, object]]] = {}
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            raise DetectorUnavailableError("Frame prediction entries must be objects.")
        frame_id = raw_frame.get("frame_id")
        detections = raw_frame.get("detections", [])
        if not isinstance(frame_id, str) or not frame_id:
            raise DetectorUnavailableError("Frame prediction entries must include frame_id.")
        if not isinstance(detections, list):
            raise DetectorUnavailableError(
                f"Frame prediction detections must be a list for frame_id={frame_id}."
            )
        mapping[frame_id] = detections
    return mapping


def _normalize_detection(
    item: object, *, detection_index: int, source_name: str
) -> Detection:
    if not isinstance(item, dict):
        raise DetectorUnavailableError("Detection entries must be JSON objects.")

    try:
        centroid = tuple(float(value) for value in item["centroid"])
        bbox_min = tuple(float(value) for value in item["bbox_min"])
        bbox_max = tuple(float(value) for value in item["bbox_max"])
    except KeyError as exc:
        raise DetectorUnavailableError(
            f"Detection entry is missing required field: {exc.args[0]}"
        ) from exc

    if len(centroid) != 3 or len(bbox_min) != 3 or len(bbox_max) != 3:
        raise DetectorUnavailableError(
            "Detection centroid and bounding boxes must each contain exactly 3 values."
        )

    return Detection(
        detection_id=str(item.get("detection_id", f"{source_name}-{detection_index:04d}")),
        centroid=centroid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        point_count=int(item.get("point_count", 0)),
        cell_count=int(item.get("cell_count", 0)),
        label=str(item.get("label", "object")),
        score=(
            None
            if item.get("score") is None
            else float(item["score"])
        ),
        source=str(item.get("source", source_name)),
        heading_rad=(
            None
            if item.get("heading_rad") is None
            else float(item["heading_rad"])
        ),
    )
