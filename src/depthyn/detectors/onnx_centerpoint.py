"""ONNX CenterPoint detector backend for Depthyn.

Uses Autoware-compatible CenterPoint ONNX models (PointPillars encoder +
backbone/neck/head) for in-process GPU inference.  No MMDetection3D or
OpenPCDet dependency -- just numpy and onnxruntime.
"""

from __future__ import annotations

import math
from pathlib import Path

from depthyn.config import DetectorConfig
from depthyn.detectors.base import DetectorResult, DetectorUnavailableError
from depthyn.models import Detection, Frame, Point3D

# ---------------------------------------------------------------------------
# Autoware CenterPoint v2 model configuration
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = ["car", "truck", "bus", "bicycle", "pedestrian"]

# Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max] in metres.
PC_RANGE = [-76.8, -76.8, -4.0, 76.8, 76.8, 6.0]

# Voxel (pillar) size in metres.
VOXEL_SIZE = [0.32, 0.32, 10.0]

# BEV grid dimensions (derived from range / voxel size).
GRID_X = int((PC_RANGE[3] - PC_RANGE[0]) / VOXEL_SIZE[0])  # 480
GRID_Y = int((PC_RANGE[4] - PC_RANGE[1]) / VOXEL_SIZE[1])  # 480

MAX_VOXELS = 40_000
MAX_POINTS_PER_VOXEL = 32
ENCODER_IN_FEATURES = 9
ENCODER_OUT_FEATURES = 32

SCORE_THRESHOLD = 0.35
CIRCLE_NMS_DIST = 0.5
YAW_NORM_THRESHOLDS = [0.3, 0.3, 0.3, 0.3, 0.0]


# ---------------------------------------------------------------------------
# Pure-numpy preprocessing
# ---------------------------------------------------------------------------

def _voxelize(points, np):
    """Assign points to pillars and build the raw feature tensor.

    Parameters
    ----------
    points : ndarray [N, 4]  (x, y, z, time_lag)

    Returns
    -------
    features : ndarray [num_voxels, MAX_POINTS_PER_VOXEL, ENCODER_IN_FEATURES]
    coords   : ndarray [num_voxels, 2]   pillar grid indices (xi, yi)
    num_voxels : int
    """
    x_min, y_min, z_min = PC_RANGE[0], PC_RANGE[1], PC_RANGE[2]
    x_max, y_max, z_max = PC_RANGE[3], PC_RANGE[4], PC_RANGE[5]

    # Filter to range.
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min) & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    points = points[mask]
    if points.shape[0] == 0:
        return (
            np.zeros((0, MAX_POINTS_PER_VOXEL, ENCODER_IN_FEATURES), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
            0,
        )

    # Shuffle so the voxel cap is unbiased.
    rng = np.random.default_rng(42)
    rng.shuffle(points)

    # Compute grid indices.
    xi = ((points[:, 0] - x_min) / VOXEL_SIZE[0]).astype(np.int32)
    yi = ((points[:, 1] - y_min) / VOXEL_SIZE[1]).astype(np.int32)
    xi = np.clip(xi, 0, GRID_X - 1)
    yi = np.clip(yi, 0, GRID_Y - 1)

    # Map each (xi, yi) to a voxel index using a flat hash.
    grid_key = yi.astype(np.int64) * GRID_X + xi.astype(np.int64)

    # Discover unique pillars and assign voxel indices.
    unique_keys, inverse = np.unique(grid_key, return_inverse=True)
    num_voxels = min(len(unique_keys), MAX_VOXELS)
    unique_keys = unique_keys[:num_voxels]

    # Build per-voxel buffers.
    raw_features = np.zeros(
        (num_voxels, MAX_POINTS_PER_VOXEL, 4), dtype=np.float32
    )
    point_counts = np.zeros(num_voxels, dtype=np.int32)
    coords = np.zeros((num_voxels, 2), dtype=np.int32)

    # Map unique_key -> voxel slot.
    key_to_slot = np.full(unique_keys.max() + 1, -1, dtype=np.int32)
    key_to_slot[unique_keys] = np.arange(num_voxels, dtype=np.int32)

    for pt_idx in range(points.shape[0]):
        slot = key_to_slot[grid_key[pt_idx]]
        if slot < 0 or slot >= num_voxels:
            continue
        cnt = point_counts[slot]
        if cnt >= MAX_POINTS_PER_VOXEL:
            continue
        raw_features[slot, cnt] = points[pt_idx]
        point_counts[slot] = cnt + 1

    # Fill coords (xi, yi per voxel).
    for v_idx in range(num_voxels):
        key = unique_keys[v_idx]
        coords[v_idx, 0] = int(key % GRID_X)   # xi
        coords[v_idx, 1] = int(key // GRID_X)  # yi

    # Build 9-feature tensor:
    # [x, y, z, t, x-mean, y-mean, z-mean, x-center, y-center]
    features = np.zeros(
        (num_voxels, MAX_POINTS_PER_VOXEL, ENCODER_IN_FEATURES), dtype=np.float32
    )
    for v_idx in range(num_voxels):
        cnt = point_counts[v_idx]
        if cnt == 0:
            continue
        pts = raw_features[v_idx, :cnt]  # [cnt, 4]
        mean_xyz = pts[:, :3].mean(axis=0)  # [3]

        pillar_center_x = coords[v_idx, 0] * VOXEL_SIZE[0] + VOXEL_SIZE[0] / 2.0 + PC_RANGE[0]
        pillar_center_y = coords[v_idx, 1] * VOXEL_SIZE[1] + VOXEL_SIZE[1] / 2.0 + PC_RANGE[1]

        for p_idx in range(cnt):
            x, y, z, t = raw_features[v_idx, p_idx]
            features[v_idx, p_idx, 0] = x
            features[v_idx, p_idx, 1] = y
            features[v_idx, p_idx, 2] = z
            features[v_idx, p_idx, 3] = t
            features[v_idx, p_idx, 4] = x - mean_xyz[0]
            features[v_idx, p_idx, 5] = y - mean_xyz[1]
            features[v_idx, p_idx, 6] = z - mean_xyz[2]
            features[v_idx, p_idx, 7] = x - pillar_center_x
            features[v_idx, p_idx, 8] = y - pillar_center_y

    return features, coords, num_voxels


def _scatter_to_bev(pillar_features, coords, num_voxels, np):
    """Scatter pillar features onto a dense BEV grid.

    Parameters
    ----------
    pillar_features : ndarray [num_voxels, 1, 32]
    coords          : ndarray [num_voxels, 2]  (xi, yi)

    Returns
    -------
    spatial_features : ndarray [1, 32, GRID_Y, GRID_X]
    """
    spatial = np.zeros((1, ENCODER_OUT_FEATURES, GRID_Y, GRID_X), dtype=np.float32)
    for v_idx in range(num_voxels):
        xi = coords[v_idx, 0]
        yi = coords[v_idx, 1]
        spatial[0, :, yi, xi] = pillar_features[v_idx, 0, :]
    return spatial


# ---------------------------------------------------------------------------
# Post-processing: decode + NMS
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + _exp_safe(-x))


def _exp_safe(x):
    import numpy as _np
    return _np.exp(_np.clip(x, -50.0, 50.0))


def _decode_boxes(outputs, np):
    """Decode CenterPoint head outputs into a list of raw box dicts."""
    heatmap = outputs["heatmap"][0]   # [5, H, W]
    reg = outputs["reg"][0]           # [2, H, W]
    height = outputs["height"][0]     # [1, H, W]
    dim = outputs["dim"][0]           # [3, H, W]
    rot = outputs["rot"][0]           # [2, H, W]
    vel = outputs["vel"][0]           # [2, H, W]

    num_classes, h, w = heatmap.shape
    scores = _sigmoid(heatmap)  # [5, H, W]

    # For each cell, pick the best class.
    best_class = np.argmax(scores, axis=0)  # [H, W]
    best_score = np.max(scores, axis=0)     # [H, W]

    # Threshold.
    mask = best_score >= SCORE_THRESHOLD
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    boxes = []
    for idx in range(len(xs)):
        xi = int(xs[idx])
        yi = int(ys[idx])
        cls_id = int(best_class[yi, xi])
        score = float(best_score[yi, xi])

        # Position.
        cx = VOXEL_SIZE[0] * (xi + float(reg[0, yi, xi])) + PC_RANGE[0]
        cy = VOXEL_SIZE[1] * (yi + float(reg[1, yi, xi])) + PC_RANGE[1]
        cz = float(height[0, yi, xi])

        # Dimensions (exp transform).
        dx = float(_exp_safe(np.array(dim[0, yi, xi])))
        dy = float(_exp_safe(np.array(dim[1, yi, xi])))
        dz = float(_exp_safe(np.array(dim[2, yi, xi])))

        # Yaw.
        yaw_sin = float(rot[0, yi, xi])
        yaw_cos = float(rot[1, yi, xi])
        yaw_norm = math.sqrt(yaw_sin ** 2 + yaw_cos ** 2)
        if yaw_norm < YAW_NORM_THRESHOLDS[cls_id]:
            continue
        yaw = math.atan2(yaw_sin, yaw_cos)

        boxes.append({
            "cx": cx, "cy": cy, "cz": cz,
            "dx": dx, "dy": dy, "dz": dz,
            "yaw": yaw, "score": score,
            "cls_id": cls_id,
        })

    return boxes


def _circle_nms(boxes):
    """Simple circle NMS: suppress nearby boxes by 2D center distance."""
    if not boxes:
        return boxes
    boxes.sort(key=lambda b: b["score"], reverse=True)
    keep = []
    for box in boxes:
        suppressed = False
        for kept in keep:
            dist = math.sqrt(
                (box["cx"] - kept["cx"]) ** 2 + (box["cy"] - kept["cy"]) ** 2
            )
            if dist < CIRCLE_NMS_DIST:
                suppressed = True
                break
        if not suppressed:
            keep.append(box)
    return keep


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class OnnxCenterPointDetector:
    """In-process CenterPoint detector using ONNX Runtime."""

    name = "centerpoint-onnx"
    input_mode = "full"

    def __init__(self, detector_config: DetectorConfig) -> None:
        self._config = detector_config
        self._score_threshold = detector_config.score_threshold or SCORE_THRESHOLD

        try:
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            raise DetectorUnavailableError(
                "centerpoint-onnx requires numpy and onnxruntime-gpu.  "
                "Install them with: pip install numpy onnxruntime-gpu"
            ) from exc

        model_dir = self._resolve_model_dir()

        encoder_path = model_dir / "pts_voxel_encoder_centerpoint.onnx"
        backbone_path = model_dir / "pts_backbone_neck_head_centerpoint.onnx"

        for path in (encoder_path, backbone_path):
            if not path.is_file():
                raise DetectorUnavailableError(
                    f"ONNX model not found: {path}\n"
                    "Run: python tools/download_models.py --model centerpoint"
                )

        providers = self._select_providers(onnxruntime)
        self._encoder = onnxruntime.InferenceSession(
            str(encoder_path), providers=providers
        )
        self._backbone = onnxruntime.InferenceSession(
            str(backbone_path), providers=providers
        )
        self._np = numpy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Frame, points: list[Point3D]) -> DetectorResult:
        np = self._np

        if not points:
            return DetectorResult(
                detections=[], input_point_count=0,
                metadata={"backend": "onnx-centerpoint"},
            )

        # Convert to [N, 4]  (x, y, z, time_lag=0).
        pts = np.zeros((len(points), 4), dtype=np.float32)
        for i, (x, y, z) in enumerate(points):
            pts[i, 0] = x
            pts[i, 1] = y
            pts[i, 2] = z
            # pts[i, 3] remains 0.0  (time_lag for single-frame)

        # 1. Voxelize.
        features, coords, num_voxels = _voxelize(pts, np)
        if num_voxels == 0:
            return DetectorResult(
                detections=[], input_point_count=len(points),
                metadata={"backend": "onnx-centerpoint", "voxels": 0},
            )

        # 2. Encoder.
        encoder_out = self._encoder.run(
            None, {"input_features": features}
        )[0]  # [num_voxels, 1, 32]

        # 3. Scatter to BEV.
        spatial = _scatter_to_bev(encoder_out, coords, num_voxels, np)

        # 4. Backbone + head.
        backbone_out = self._backbone.run(None, {"spatial_features": spatial})
        output_names = [o.name for o in self._backbone.get_outputs()]
        outputs = dict(zip(output_names, backbone_out))

        # 5. Decode + NMS.
        raw_boxes = _decode_boxes(outputs, np)
        boxes = _circle_nms(raw_boxes)

        # 6. Convert to Depthyn Detection objects.
        detections: list[Detection] = []
        for det_idx, box in enumerate(boxes, start=1):
            cx, cy, cz = box["cx"], box["cy"], box["cz"]
            dx, dy, dz = box["dx"], box["dy"], box["dz"]
            label = CLASS_NAMES[box["cls_id"]]
            detections.append(Detection(
                detection_id=f"cp-{det_idx:04d}",
                centroid=(cx, cy, cz),
                bbox_min=(cx - dx / 2, cy - dy / 2, cz - dz / 2),
                bbox_max=(cx + dx / 2, cy + dy / 2, cz + dz / 2),
                point_count=0,
                cell_count=0,
                label=label,
                score=round(box["score"], 4),
                source=self.name,
                heading_rad=box["yaw"],
            ))

        return DetectorResult(
            detections=detections,
            input_point_count=len(points),
            metadata={
                "backend": "onnx-centerpoint",
                "voxels": num_voxels,
                "raw_boxes": len(raw_boxes),
                "after_nms": len(detections),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_dir(self) -> Path:
        if self._config.config_path is not None:
            candidate = Path(self._config.config_path)
            if candidate.is_dir():
                return candidate
            if candidate.is_file():
                return candidate.parent

        # Default: <project_root>/models/centerpoint-onnx/
        project_root = Path(__file__).resolve().parents[3]
        default = project_root / "models" / "centerpoint-onnx"
        if default.is_dir():
            return default

        raise DetectorUnavailableError(
            "Could not locate ONNX model directory. Either place models in "
            f"{default} or pass --ml-config <model_dir>."
        )

    @staticmethod
    def _select_providers(ort):
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
