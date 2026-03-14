from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpcdet-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--model-name", default="pointpillars")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.openpcdet_repo))

    import numpy as np
    import torch

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    class DepthynDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names, points, logger):
            super().__init__(
                dataset_cfg=dataset_cfg,
                class_names=class_names,
                training=False,
                root_path=args.input_json,
                logger=logger,
            )
            self._points = points

        def __len__(self):
            return 1

        def __getitem__(self, index):
            input_dict = {
                "points": self._points,
                "frame_id": index,
            }
            return self.prepare_data(data_dict=input_dict)

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    points_xyz = payload["points_xyz"]
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected points_xyz with shape [N, 3].")
    intensities = np.full((points.shape[0], 1), payload.get("default_intensity", 0.0), dtype=np.float32)
    point_cloud = np.concatenate([points, intensities], axis=1)

    cfg_from_yaml_file(str(args.config), cfg)
    logger = common_utils.create_logger()
    dataset = DepthynDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        points=point_cloud,
        logger=logger,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.checkpoint), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    with torch.no_grad():
        data_dict = dataset[0]
        batch_dict = dataset.collate_batch([data_dict])
        load_data_to_gpu(batch_dict)
        pred_dicts, _ = model.forward(batch_dict)

    pred = pred_dicts[0]
    boxes = pred["pred_boxes"].detach().cpu().numpy()
    scores = pred["pred_scores"].detach().cpu().numpy()
    labels = pred["pred_labels"].detach().cpu().numpy()

    detections = []
    det_index = 0
    for box, score, label_index in zip(boxes, scores, labels):
        if float(score) < args.score_threshold:
            continue
        det_index += 1
        cx, cy, cz, dx, dy, dz, heading = [float(value) for value in box.tolist()]
        bbox_min = [cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0]
        bbox_max = [cx + dx / 2.0, cy + dy / 2.0, cz + dz / 2.0]
        detections.append(
            {
                "detection_id": f"{args.model_name}-{det_index:04d}",
                "centroid": [cx, cy, cz],
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "point_count": 0,
                "cell_count": 0,
                "label": cfg.CLASS_NAMES[int(label_index) - 1],
                "score": round(float(score), 6),
                "heading_rad": heading,
            }
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


if __name__ == "__main__":
    raise SystemExit(main())
