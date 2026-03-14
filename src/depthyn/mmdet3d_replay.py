from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from depthyn.comparison import run_detector_comparison
from depthyn.config import DetectorConfig, ReplayConfig
from depthyn.ml_prep import export_ml_replay_bundle


class MMDet3DReplayError(RuntimeError):
    """Raised when the external MMDetection3D replay runner fails."""


def run_mmdet3d_manifest_inference(
    *,
    manifest_path: Path,
    output_path: Path,
    backend_python: str | None,
    backend_repo: Path | None,
    config_path: Path | None,
    checkpoint_path: Path | None,
    score_threshold: float,
    model_name: str,
    device: str,
) -> dict[str, object]:
    if config_path is None:
        raise MMDet3DReplayError("MMDetection3D manifest inference requires --ml-config.")
    if checkpoint_path is None:
        raise MMDet3DReplayError(
            "MMDetection3D manifest inference requires --ml-checkpoint."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path = Path(__file__).resolve().parents[2] / "tools" / "mmdet3d_runner.py"
    command = [
        backend_python or sys.executable,
        str(runner_path),
        "--manifest-json",
        str(manifest_path),
        "--output-json",
        str(output_path),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--score-threshold",
        str(score_threshold),
        "--model-name",
        model_name,
        "--device",
        device,
    ]
    if backend_repo is not None:
        command.extend(["--mmdet3d-repo", str(backend_repo)])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise MMDet3DReplayError(
            f"MMDetection3D replay inference failed: {details}"
        )

    return json.loads(output_path.read_text(encoding="utf-8"))


def run_stage1_mmdet3d_compare(
    *,
    input_dir: Path,
    output_dir: Path,
    mode: str,
    zone_config: Path | None,
    max_frames: int | None,
    preview_point_limit: int,
    voxel_size_m: float,
    cluster_cell_size_m: float,
    track_max_distance_m: float,
    min_range_m: float,
    max_range_m: float,
    z_min_m: float,
    z_max_m: float,
    default_intensity: float,
    backend_python: str | None,
    backend_repo: Path | None,
    config_path: Path | None,
    checkpoint_path: Path | None,
    score_threshold: float,
    model_name: str,
    device: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = output_dir / "ml-replay"
    comparison_dir = output_dir / "comparison"
    predictions_path = output_dir / f"{model_name}-predictions.json"

    manifest = export_ml_replay_bundle(
        input_dir=input_dir,
        output_dir=export_dir,
        max_frames=max_frames,
        voxel_size_m=voxel_size_m,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        default_intensity=default_intensity,
    )

    prediction_payload = run_mmdet3d_manifest_inference(
        manifest_path=export_dir / "manifest.json",
        output_path=predictions_path,
        backend_python=backend_python,
        backend_repo=backend_repo,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        score_threshold=score_threshold,
        model_name=model_name,
        device=device,
    )

    comparison = run_detector_comparison(
        ReplayConfig(
            input_dir=input_dir,
            output_json=comparison_dir / "placeholder.json",
            mode=mode,
            zone_config=zone_config,
            max_frames=max_frames,
            preview_point_limit=preview_point_limit,
            voxel_size_m=voxel_size_m,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            cluster_cell_size_m=cluster_cell_size_m,
            track_max_distance_m=track_max_distance_m,
        ),
        [
            DetectorConfig(kind="baseline"),
            DetectorConfig(
                kind="precomputed",
                label=model_name,
                prediction_path=predictions_path,
            ),
        ],
        comparison_dir,
    )

    return {
        "project": "Depthyn",
        "pipeline_item": "stage1b_mmdet3d_compare",
        "model_name": model_name,
        "manifest_path": str(export_dir / "manifest.json"),
        "predictions_path": str(predictions_path),
        "comparison_path": str(comparison_dir / "comparison.json"),
        "frame_count": manifest["frame_count"],
        "prediction_summary": {
            "frames_processed": prediction_payload.get("frames_processed"),
            "total_detections": prediction_payload.get("total_detections"),
        },
        "comparison": comparison,
    }
