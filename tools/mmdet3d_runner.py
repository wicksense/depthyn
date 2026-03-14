from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _resolve_class_names(model) -> list[str]:
    dataset_meta = getattr(model, "dataset_meta", None)
    if isinstance(dataset_meta, dict):
        for key in ("classes", "CLASSES", "class_names"):
            value = dataset_meta.get(key)
            if value:
                return list(value)
    return []


def _normalize_result(result, model_name: str, score_threshold: float, class_names: list[str]) -> list[dict[str, object]]:
    if isinstance(result, (list, tuple)) and result:
        result = result[0]

    pred_instances = getattr(result, "pred_instances_3d", None)
    if pred_instances is None and isinstance(result, dict):
        pred_instances = result.get("pred_instances_3d")
    if pred_instances is None:
        raise RuntimeError("MMDetection3D result did not include pred_instances_3d.")

    boxes = _to_numpy(getattr(getattr(pred_instances, "bboxes_3d", None), "tensor", None))
    scores = _to_numpy(getattr(pred_instances, "scores_3d", None))
    labels = _to_numpy(getattr(pred_instances, "labels_3d", None))

    if boxes is None or scores is None or labels is None:
        raise RuntimeError("MMDetection3D result is missing boxes, scores, or labels.")

    detections: list[dict[str, object]] = []
    for det_index, (box, score, label_index) in enumerate(zip(boxes, scores, labels), start=1):
        score_value = float(score)
        if score_value < score_threshold:
            continue

        cx, cy, cz, dx, dy, dz = [float(value) for value in box[:6]]
        heading = float(box[6]) if len(box) > 6 else None
        label_value = int(label_index)
        label = (
            class_names[label_value]
            if 0 <= label_value < len(class_names)
            else f"class_{label_value}"
        )

        detections.append(
            {
                "detection_id": f"{model_name}-{det_index:04d}",
                "centroid": [cx, cy, cz],
                "bbox_min": [cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0],
                "bbox_max": [cx + dx / 2.0, cy + dy / 2.0, cz + dz / 2.0],
                "point_count": 0,
                "cell_count": 0,
                "label": label,
                "score": round(score_value, 6),
                "heading_rad": heading,
            }
        )
    return detections


def _load_single_frame_cloud(payload, np):
    points_xyz = payload["points_xyz"]
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected points_xyz with shape [N, 3].")

    if points.shape[0] == 0:
        return None

    intensities = np.full(
        (points.shape[0], 1),
        payload.get("default_intensity", 0.0),
        dtype=np.float32,
    )
    return np.concatenate([points, intensities], axis=1)


def _load_xyzi_bin(path, np):
    point_cloud = np.fromfile(path, dtype=np.float32)
    if point_cloud.size == 0:
        return None
    if point_cloud.size % 4 != 0:
        raise ValueError(f"Expected XYZI float32 data in {path}")
    return point_cloud.reshape(-1, 4)


def _ensure_model_feature_count(point_cloud, np):
    if point_cloud is None:
        return None
    if point_cloud.ndim != 2:
        raise ValueError("Expected point cloud array with shape [N, C].")
    if point_cloud.shape[1] == 5:
        return point_cloud
    if point_cloud.shape[1] == 4:
        zeros = np.zeros((point_cloud.shape[0], 1), dtype=np.float32)
        return np.concatenate([point_cloud, zeros], axis=1)
    raise ValueError(
        f"Expected point cloud with 4 or 5 features, received shape {point_cloud.shape}"
    )


def _patch_test_pipeline_node(node, load_dim: int, use_dim: int):
    if isinstance(node, dict):
        node_type = node.get("type")
        if node_type == "LoadPointsFromFile":
            node["load_dim"] = load_dim
            node["use_dim"] = list(range(use_dim))
        transforms = node.get("transforms")
        if isinstance(transforms, list):
            _patch_test_pipeline_list(transforms, load_dim, use_dim)
    elif isinstance(node, list):
        _patch_test_pipeline_list(node, load_dim, use_dim)


def _patch_test_pipeline_list(transforms, load_dim: int, use_dim: int):
    filtered = []
    for transform in transforms:
        if isinstance(transform, dict) and transform.get("type") == "LoadPointsFromMultiSweeps":
            continue
        _patch_test_pipeline_node(transform, load_dim, use_dim)
        filtered.append(transform)
    transforms[:] = filtered


def _patch_config_for_depthyn(config, load_dim: int, use_dim: int):
    # Depthyn replay exports a single frame, so nuScenes-style multi-sweep
    # loaders need to be removed and the point feature count patched.
    for attr in ("test_pipeline", "eval_pipeline"):
        pipeline = config.get(attr)
        if isinstance(pipeline, list):
            _patch_test_pipeline_list(pipeline, load_dim, use_dim)

    test_dataloader = config.get("test_dataloader")
    if isinstance(test_dataloader, dict):
        dataset = test_dataloader.get("dataset")
        if isinstance(dataset, dict):
            pipeline = dataset.get("pipeline")
            if isinstance(pipeline, list):
                _patch_test_pipeline_list(pipeline, load_dim, use_dim)


def _init_depthyn_model(config_path: Path, checkpoint_path: Path, device: str, load_dim: int, use_dim: int):
    from mmengine.config import Config
    from mmdet3d.apis import init_model

    config = Config.fromfile(str(config_path))
    _patch_config_for_depthyn(config, load_dim=load_dim, use_dim=use_dim)
    return init_model(config, str(checkpoint_path), device=device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdet3d-repo", type=Path, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--model-name", default="centerpoint")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if (args.input_json is None) == (args.manifest_json is None):
        raise ValueError("Provide exactly one of --input-json or --manifest-json.")

    if args.mmdet3d_repo is not None:
        sys.path.insert(0, str(args.mmdet3d_repo))

    import numpy as np

    from mmdet3d.apis import inference_detector

    if args.input_json is not None:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        point_cloud = _load_single_frame_cloud(payload, np)
        point_cloud = _ensure_model_feature_count(point_cloud, np)

        if point_cloud is None:
            args.output_json.write_text(
                json.dumps(
                    {
                        "detections": [],
                        "stdout": f"{args.model_name} detections: 0",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return 0

        with tempfile.TemporaryDirectory(prefix="depthyn-mmdet3d-") as temp_dir:
            point_path = Path(temp_dir) / "frame.bin"
            point_cloud.tofile(point_path)
            model = _init_depthyn_model(
                args.config,
                args.checkpoint,
                device=args.device,
                load_dim=point_cloud.shape[1],
                use_dim=point_cloud.shape[1],
            )
            class_names = _resolve_class_names(model)
            result = inference_detector(model, str(point_path))

        detections = _normalize_result(
            result,
            args.model_name,
            args.score_threshold,
            class_names,
        )
        args.output_json.write_text(
            json.dumps(
                {
                    "detections": detections,
                    "stdout": f"{args.model_name} detections: {len(detections)}",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0

    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    raw_frames = manifest.get("frames", [])
    if not isinstance(raw_frames, list):
        raise ValueError("Manifest 'frames' value must be a list.")

    frame_predictions: list[dict[str, object]] = []
    total_detections = 0
    manifest_root = args.manifest_json.parent

    for raw_frame in raw_frames:
        frame_id = raw_frame["frame_id"]
        point_path = manifest_root / raw_frame["points_path"]
        point_cloud = _load_xyzi_bin(point_path, np)
        if point_cloud is None:
            detections = []
        else:
            point_cloud = _ensure_model_feature_count(point_cloud, np)
            with tempfile.TemporaryDirectory(prefix="depthyn-mmdet3d-frame-") as temp_dir:
                temp_point_path = Path(temp_dir) / f"{frame_id}.bin"
                point_cloud.tofile(temp_point_path)
                if "model" not in locals():
                    model = _init_depthyn_model(
                        args.config,
                        args.checkpoint,
                        device=args.device,
                        load_dim=point_cloud.shape[1],
                        use_dim=point_cloud.shape[1],
                    )
                    class_names = _resolve_class_names(model)
                result = inference_detector(model, str(temp_point_path))
            detections = _normalize_result(
                result,
                args.model_name,
                args.score_threshold,
                class_names,
            )
        total_detections += len(detections)
        frame_predictions.append(
            {
                "frame_id": frame_id,
                "timestamp_ns": raw_frame.get("timestamp_ns"),
                "detections": detections,
            }
        )

    args.output_json.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "frame_predictions": frame_predictions,
                "frames_processed": len(frame_predictions),
                "total_detections": total_detections,
                "stdout": (
                    f"{args.model_name} frames: {len(frame_predictions)}, "
                    f"detections: {total_detections}"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
