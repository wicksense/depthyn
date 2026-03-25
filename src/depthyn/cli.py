from __future__ import annotations

import argparse
import json
from pathlib import Path

from depthyn.comparison import run_detector_comparison
from depthyn.config import DetectorConfig, ReplayConfig
from depthyn.mmdet3d_replay import (
    run_mmdet3d_manifest_inference,
    run_stage1_mmdet3d_compare,
)
from depthyn.ml_prep import export_ml_replay_bundle
from depthyn.pipeline import run_replay, write_summary
from depthyn.viewer import serve_viewer


def _add_mmdet3d_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mmdet3d-python",
        default=None,
        help="Python executable inside the MMDetection3D environment.",
    )
    parser.add_argument(
        "--mmdet3d-repo",
        type=Path,
        default=None,
        help="Optional path to an MMDetection3D repository checkout.",
    )
    parser.add_argument(
        "--ml-config",
        type=Path,
        default=None,
        help="Model config file for CenterPoint, DSVT, or another MMDetection3D model.",
    )
    parser.add_argument(
        "--ml-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file for the selected MMDetection3D model.",
    )
    parser.add_argument(
        "--ml-score-threshold",
        type=float,
        default=0.25,
        help="Score threshold applied to ML detector outputs.",
    )
    parser.add_argument(
        "--ml-device",
        default="cuda:0",
        help="Torch device passed through to MMDetection3D, for example cuda:0 or cpu.",
    )


def _add_ml_export_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames exported or processed.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.3,
        help="Voxel size in meters for downsampling while loading frames.",
    )
    parser.add_argument(
        "--min-range",
        type=float,
        default=1.0,
        help="Minimum range in meters for exported points.",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=60.0,
        help="Maximum range in meters for exported points.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=-2.5,
        help="Minimum Z in meters for exported points.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=4.5,
        help="Maximum Z in meters for exported points.",
    )
    parser.add_argument(
        "--default-intensity",
        type=float,
        default=0.0,
        help="Intensity value written into exported XYZI frames.",
    )


def _build_detector_config(
    kind: str,
    backend_python: str | None,
    backend_repo: Path | None,
    prediction_path: Path | None,
    config_path: Path | None,
    checkpoint_path: Path | None,
    score_threshold: float,
    device: str,
) -> DetectorConfig:
    return DetectorConfig(
        kind=kind,
        backend_python=backend_python,
        backend_repo=backend_repo,
        prediction_path=prediction_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        score_threshold=score_threshold,
        device=device,
    )


def _build_compare_detector_config(name: str, args: argparse.Namespace) -> DetectorConfig:
    if name == "pointpillars":
        return _build_detector_config(
            name,
            args.mmdet3d_python,
            args.mmdet3d_repo,
            None,
            args.pointpillars_config,
            args.pointpillars_checkpoint,
            args.ml_score_threshold,
            args.ml_device,
        )
    if name == "centerpoint":
        return _build_detector_config(
            name,
            args.mmdet3d_python,
            args.mmdet3d_repo,
            None,
            args.centerpoint_config,
            args.centerpoint_checkpoint,
            args.ml_score_threshold,
            args.ml_device,
        )
    if name == "dsvt":
        return _build_detector_config(
            name,
            args.mmdet3d_python,
            args.mmdet3d_repo,
            None,
            args.dsvt_config,
            args.dsvt_checkpoint,
            args.ml_score_threshold,
            args.ml_device,
        )
    if name == "centerpoint-onnx":
        return _build_detector_config(
            name,
            None,
            None,
            None,
            None,
            None,
            args.ml_score_threshold,
            args.ml_device,
        )
    if name == "precomputed":
        return _build_detector_config(
            name,
            None,
            None,
            args.precomputed_path,
            None,
            None,
            args.ml_score_threshold,
            args.ml_device,
        )
    return _build_detector_config(
        name,
        args.mmdet3d_python,
        args.mmdet3d_repo,
        None,
        None,
        None,
        args.ml_score_threshold,
        args.ml_device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="depthyn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay_parser = subparsers.add_parser(
        "replay",
        help="Run the replay pipeline on LiDAR data (CSV or Ouster pcap).",
    )
    replay_parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing converted CSV files or Ouster pcap files.",
    )
    replay_parser.add_argument(
        "--source-type",
        choices=("auto", "csv", "pcap"),
        default="auto",
        help="Input source type. 'auto' detects based on file extensions.",
    )
    replay_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/depthyn-replay-summary.json"),
        help="Where to write the JSON run summary.",
    )
    replay_parser.add_argument(
        "--mode",
        choices=("mobile", "stationary"),
        default="mobile",
        help="Use stationary mode to enable background warmup and foreground suppression.",
    )
    replay_parser.add_argument(
        "--world-align",
        action="store_true",
        help="Transform mobile replay outputs into a GPS-anchored world frame.",
    )
    replay_parser.add_argument(
        "--gps-path",
        type=Path,
        default=None,
        help="Optional GPS CSV path for world alignment. Defaults to auto-discovery in the input directory.",
    )
    replay_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames processed.",
    )
    replay_parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.3,
        help="Voxel size in meters for downsampling while loading frames.",
    )
    replay_parser.add_argument(
        "--cluster-cell-size",
        type=float,
        default=0.5,
        help="XY cell size in meters used for occupancy clustering.",
    )
    replay_parser.add_argument(
        "--track-max-distance",
        type=float,
        default=3.0,
        help="Maximum centroid distance in meters for track association.",
    )
    replay_parser.add_argument(
        "--detector",
        choices=("baseline", "precomputed", "centerpoint-onnx", "pointpillars", "centerpoint", "dsvt"),
        default="baseline",
        help="Detection backend to run during replay.",
    )
    replay_parser.add_argument(
        "--preview-points",
        type=int,
        default=1200,
        help="Maximum downsampled points per frame to embed for viewer playback.",
    )
    replay_parser.add_argument(
        "--detail-points",
        type=int,
        default=0,
        help="Optional higher-detail point budget per frame for richer object/scanline viewing.",
    )
    replay_parser.add_argument(
        "--zone-config",
        type=Path,
        default=None,
        help="Optional JSON file defining XY zones for scene-rule evaluation.",
    )
    replay_parser.add_argument(
        "--detector-on-foreground",
        action="store_true",
        help="In stationary mode, run the detector on foreground-only points even if it normally expects the full scene.",
    )
    replay_parser.add_argument(
        "--precomputed-path",
        type=Path,
        default=None,
        help="Directory or JSON file containing normalized precomputed detections.",
    )
    _add_mmdet3d_options(replay_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Run the same replay across multiple detector backends and write a comparison report.",
    )
    compare_parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing converted CSV files or Ouster pcap files.",
    )
    compare_parser.add_argument(
        "--source-type",
        choices=("auto", "csv", "pcap"),
        default="auto",
        help="Input source type. 'auto' detects based on file extensions.",
    )
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/detector-comparison"),
        help="Directory where per-detector summaries and the comparison report are written.",
    )
    compare_parser.add_argument(
        "--mode",
        choices=("mobile", "stationary"),
        default="mobile",
        help="Replay mode used for all detector backends in the comparison.",
    )
    compare_parser.add_argument(
        "--world-align",
        action="store_true",
        help="Transform replay outputs into a GPS-anchored world frame.",
    )
    compare_parser.add_argument(
        "--gps-path",
        type=Path,
        default=None,
        help="Optional GPS CSV path for world alignment. Defaults to auto-discovery in the input directory.",
    )
    compare_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames processed.",
    )
    compare_parser.add_argument(
        "--detectors",
        nargs="+",
        default=["baseline"],
        choices=("baseline", "precomputed", "centerpoint-onnx", "pointpillars", "centerpoint", "dsvt"),
        help="Detector backends to run side by side.",
    )
    compare_parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.3,
        help="Voxel size in meters for downsampling while loading frames.",
    )
    compare_parser.add_argument(
        "--cluster-cell-size",
        type=float,
        default=0.5,
        help="XY cell size in meters used for occupancy clustering.",
    )
    compare_parser.add_argument(
        "--track-max-distance",
        type=float,
        default=3.0,
        help="Maximum centroid distance in meters for track association.",
    )
    compare_parser.add_argument(
        "--preview-points",
        type=int,
        default=1200,
        help="Maximum downsampled points per frame to embed for viewer playback.",
    )
    compare_parser.add_argument(
        "--zone-config",
        type=Path,
        default=None,
        help="Optional JSON file defining XY zones for scene-rule evaluation.",
    )
    compare_parser.add_argument(
        "--detector-on-foreground",
        action="store_true",
        help="In stationary mode, run all detectors on foreground-only points even if they normally expect the full scene.",
    )
    compare_parser.add_argument(
        "--precomputed-path",
        type=Path,
        default=None,
        help="Directory or JSON file containing normalized precomputed detections.",
    )
    compare_parser.add_argument(
        "--pointpillars-config",
        type=Path,
        default=None,
        help="MMDetection3D config for PointPillars.",
    )
    compare_parser.add_argument(
        "--pointpillars-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for PointPillars.",
    )
    compare_parser.add_argument(
        "--centerpoint-config",
        type=Path,
        default=None,
        help="MMDetection3D config for CenterPoint.",
    )
    compare_parser.add_argument(
        "--centerpoint-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for CenterPoint.",
    )
    compare_parser.add_argument(
        "--dsvt-config",
        type=Path,
        default=None,
        help="MMDetection3D config for DSVT.",
    )
    compare_parser.add_argument(
        "--dsvt-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for DSVT.",
    )
    _add_mmdet3d_options(compare_parser)

    serve_parser = subparsers.add_parser(
        "serve-viewer",
        help="Serve the recorded-data viewer for a replay summary JSON.",
    )
    serve_parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Replay summary JSON produced by the replay command.",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local HTTP server.",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the local HTTP server.",
    )

    # ── Evaluate ──────────────────────────────────────────────────
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate detector against Ouster ground truth classification logs.",
    )
    eval_parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing Ouster pcap files.",
    )
    eval_parser.add_argument(
        "--gt-log",
        type=Path,
        required=True,
        help="Path to Ouster classification log file (JSON-per-line).",
    )
    eval_parser.add_argument(
        "--source-type",
        choices=("auto", "csv", "pcap"),
        default="pcap",
        help="Input source type (default: pcap).",
    )
    eval_parser.add_argument(
        "--mode",
        choices=("mobile", "stationary"),
        default="stationary",
        help="Evaluation mode. Stationary enables background modeling before detection.",
    )
    eval_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation.json"),
        help="Output evaluation report JSON.",
    )
    eval_parser.add_argument(
        "--detector",
        choices=("baseline", "centerpoint-onnx"),
        default="centerpoint-onnx",
        help="Detector to evaluate.",
    )
    eval_parser.add_argument(
        "--ml-score-threshold",
        type=float,
        default=0.25,
        help="Minimum detection confidence score.",
    )
    eval_parser.add_argument(
        "--match-distance",
        type=float,
        default=3.0,
        help="Maximum XY distance for detection-to-GT matching (meters).",
    )
    eval_parser.add_argument(
        "--min-gt-distance",
        type=float,
        default=2.0,
        help="Exclude GT objects closer than this (likely self-returns).",
    )
    eval_parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.3,
        help="Voxel downsampling resolution.",
    )
    eval_parser.add_argument(
        "--cluster-cell-size",
        type=float,
        default=0.5,
        help="XY cell size in meters used for background modeling and clustering.",
    )
    eval_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process.",
    )
    eval_parser.add_argument(
        "--detector-on-foreground",
        action="store_true",
        help="Run the detector on foreground-only points after stationary background suppression.",
    )
    eval_parser.add_argument(
        "--no-class-match",
        action="store_true",
        help="Count spatial matches as TP regardless of class.",
    )

    debug_frame_parser = subparsers.add_parser(
        "debug-frame",
        help="Export a single viewer-friendly debug frame with raw points, detector output, and Gemini GT objects.",
    )
    debug_frame_parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing Ouster pcap files.",
    )
    debug_frame_parser.add_argument(
        "--gt-log",
        type=Path,
        required=True,
        help="Path to Ouster classification log file (JSON-per-line).",
    )
    debug_frame_parser.add_argument(
        "--frame-count",
        type=int,
        required=True,
        help="Gemini/Ouster frame_count to export.",
    )
    debug_frame_parser.add_argument(
        "--source-type",
        choices=("auto", "csv", "pcap"),
        default="pcap",
        help="Input source type (default: pcap).",
    )
    debug_frame_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/debug-frame.json"),
        help="Output debug bundle JSON.",
    )
    debug_frame_parser.add_argument(
        "--detector",
        choices=("baseline", "centerpoint-onnx"),
        default="centerpoint-onnx",
        help="Detector to run on the selected frame.",
    )
    debug_frame_parser.add_argument(
        "--ml-score-threshold",
        type=float,
        default=0.25,
        help="Minimum detection confidence score.",
    )
    debug_frame_parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.3,
        help="Voxel downsampling resolution.",
    )
    debug_frame_parser.add_argument(
        "--detector-on-foreground",
        action="store_true",
        help="Run the detector on foreground-only points. Currently intended for future stationary debug parity work.",
    )

    export_parser = subparsers.add_parser(
        "prepare-ml-replay",
        help="Export filtered replay frames as ML-ready XYZI binaries plus a manifest.",
    )
    export_parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing converted frame CSV files.",
    )
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ml-replay"),
        help="Directory where the exported manifest and frame binaries are written.",
    )
    _add_ml_export_options(export_parser)

    run_mmdet_parser = subparsers.add_parser(
        "run-mmdet3d-replay",
        help="Run MMDetection3D over an exported replay manifest and write normalized predictions.",
    )
    run_mmdet_parser.add_argument(
        "--manifest-json",
        type=Path,
        required=True,
        help="Manifest produced by prepare-ml-replay.",
    )
    run_mmdet_parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/centerpoint-predictions.json"),
        help="Where to write normalized frame predictions.",
    )
    run_mmdet_parser.add_argument(
        "--model-name",
        default="centerpoint",
        help="Model label used in output metadata and imported replay comparisons.",
    )
    _add_mmdet3d_options(run_mmdet_parser)

    compare_mmdet_parser = subparsers.add_parser(
        "compare-mmdet3d-replay",
        help="Export replay frames, run MMDetection3D once, and compare the result against the baseline.",
    )
    compare_mmdet_parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing converted frame CSV files.",
    )
    compare_mmdet_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mmdet3d-stage1"),
        help="Directory where exported frames, predictions, and comparison outputs are written.",
    )
    compare_mmdet_parser.add_argument(
        "--mode",
        choices=("mobile", "stationary"),
        default="mobile",
        help="Replay mode used for the baseline comparison run.",
    )
    compare_mmdet_parser.add_argument(
        "--zone-config",
        type=Path,
        default=None,
        help="Optional JSON file defining XY zones for scene-rule evaluation.",
    )
    compare_mmdet_parser.add_argument(
        "--preview-points",
        type=int,
        default=1200,
        help="Maximum downsampled points per frame to embed for viewer playback.",
    )
    compare_mmdet_parser.add_argument(
        "--cluster-cell-size",
        type=float,
        default=0.75,
        help="XY cell size in meters used for occupancy clustering.",
    )
    compare_mmdet_parser.add_argument(
        "--track-max-distance",
        type=float,
        default=3.0,
        help="Maximum centroid distance in meters for track association.",
    )
    compare_mmdet_parser.add_argument(
        "--model-name",
        default="centerpoint",
        help="Model label used for predictions and comparison output.",
    )
    _add_ml_export_options(compare_mmdet_parser)
    _add_mmdet3d_options(compare_mmdet_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "replay":
        config = ReplayConfig(
            input_dir=args.input_dir,
            output_json=args.output,
            mode=args.mode,
            world_align=args.world_align,
            gps_path=args.gps_path,
            detector_on_foreground=args.detector_on_foreground,
            source_type=args.source_type,
            zone_config=args.zone_config,
            max_frames=args.max_frames,
            preview_point_limit=args.preview_points,
            detail_point_limit=args.detail_points,
            detector=_build_detector_config(
                args.detector,
                args.mmdet3d_python,
                args.mmdet3d_repo,
                args.precomputed_path,
                args.ml_config,
                args.ml_checkpoint,
                args.ml_score_threshold,
                args.ml_device,
            ),
            voxel_size_m=args.voxel_size,
            cluster_cell_size_m=args.cluster_cell_size,
            track_max_distance_m=args.track_max_distance,
        )
        summary = run_replay(config)
        write_summary(summary, config.output_json)
        print(f"Wrote summary to {config.output_json}")
        print(json.dumps(summary["metrics"], indent=2))
        return 0
    if args.command == "compare":
        base_config = ReplayConfig(
            input_dir=args.input_dir,
            output_json=args.output_dir / "placeholder.json",
            mode=args.mode,
            world_align=args.world_align,
            gps_path=args.gps_path,
            detector_on_foreground=args.detector_on_foreground,
            source_type=args.source_type,
            zone_config=args.zone_config,
            max_frames=args.max_frames,
            preview_point_limit=args.preview_points,
            voxel_size_m=args.voxel_size,
            cluster_cell_size_m=args.cluster_cell_size,
            track_max_distance_m=args.track_max_distance,
        )
        detectors = [
            _build_compare_detector_config(name, args)
            for name in args.detectors
        ]
        comparison = run_detector_comparison(base_config, detectors, args.output_dir)
        print(json.dumps(comparison, indent=2))
        return 0
    if args.command == "serve-viewer":
        serve_viewer(args.summary, args.host, args.port)
        return 0
    if args.command == "evaluate":
        from depthyn.evaluation.runner import run_evaluation

        eval_config = ReplayConfig(
            input_dir=args.input_dir,
            output_json=args.output,
            mode=args.mode,
            detector_on_foreground=args.detector_on_foreground,
            source_type=args.source_type,
            max_frames=args.max_frames,
            detector=_build_detector_config(
                args.detector,
                None, None, None, None, None,
                args.ml_score_threshold,
                "cpu",
            ),
            voxel_size_m=args.voxel_size,
            cluster_cell_size_m=args.cluster_cell_size,
        )
        report = run_evaluation(
            eval_config,
            args.gt_log,
            max_distance_m=args.match_distance,
            min_gt_distance_m=args.min_gt_distance,
            class_match=not args.no_class_match,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote evaluation report to {args.output}")
        print(json.dumps(report["metrics"], indent=2))
        return 0
    if args.command == "debug-frame":
        from depthyn.evaluation.debug_export import export_debug_frame

        bundle = export_debug_frame(
            input_dir=args.input_dir,
            gt_log_path=args.gt_log,
            output_path=args.output,
            frame_count=args.frame_count,
            detector=_build_detector_config(
                args.detector,
                None, None, None, None, None,
                args.ml_score_threshold,
                "cpu",
            ),
            source_type=args.source_type,
            voxel_size_m=args.voxel_size,
            detector_on_foreground=args.detector_on_foreground,
        )
        print(f"Wrote debug frame bundle to {args.output}")
        print(json.dumps(bundle["metrics"], indent=2))
        return 0
    if args.command == "prepare-ml-replay":
        manifest = export_ml_replay_bundle(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            voxel_size_m=args.voxel_size,
            min_range_m=args.min_range,
            max_range_m=args.max_range,
            z_min_m=args.z_min,
            z_max_m=args.z_max,
            default_intensity=args.default_intensity,
        )
        print(f"Wrote ML replay bundle to {args.output_dir}")
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "run-mmdet3d-replay":
        payload = run_mmdet3d_manifest_inference(
            manifest_path=args.manifest_json,
            output_path=args.output_json,
            backend_python=args.mmdet3d_python,
            backend_repo=args.mmdet3d_repo,
            config_path=args.ml_config,
            checkpoint_path=args.ml_checkpoint,
            score_threshold=args.ml_score_threshold,
            model_name=args.model_name,
            device=args.ml_device,
        )
        print(f"Wrote normalized predictions to {args.output_json}")
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "compare-mmdet3d-replay":
        result = run_stage1_mmdet3d_compare(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            zone_config=args.zone_config,
            max_frames=args.max_frames,
            preview_point_limit=args.preview_points,
            voxel_size_m=args.voxel_size,
            cluster_cell_size_m=args.cluster_cell_size,
            track_max_distance_m=args.track_max_distance,
            min_range_m=args.min_range,
            max_range_m=args.max_range,
            z_min_m=args.z_min,
            z_max_m=args.z_max,
            default_intensity=args.default_intensity,
            backend_python=args.mmdet3d_python,
            backend_repo=args.mmdet3d_repo,
            config_path=args.ml_config,
            checkpoint_path=args.ml_checkpoint,
            score_threshold=args.ml_score_threshold,
            model_name=args.model_name,
            device=args.ml_device,
        )
        print(json.dumps(result, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
