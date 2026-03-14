from __future__ import annotations

import argparse
import json
from pathlib import Path

from depthyn.comparison import run_detector_comparison
from depthyn.config import DetectorConfig, ReplayConfig
from depthyn.pipeline import run_replay, write_summary
from depthyn.viewer import serve_viewer


def _add_openpcdet_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--openpcdet-python",
        default=None,
        help="Python executable inside the OpenPCDet environment.",
    )
    parser.add_argument(
        "--openpcdet-repo",
        type=Path,
        default=None,
        help="Path to an OpenPCDet repository checkout.",
    )
    parser.add_argument(
        "--ml-config",
        type=Path,
        default=None,
        help="Model config file for PointPillars or CenterPoint.",
    )
    parser.add_argument(
        "--ml-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file for PointPillars or CenterPoint.",
    )
    parser.add_argument(
        "--ml-score-threshold",
        type=float,
        default=0.25,
        help="Score threshold applied to ML detector outputs.",
    )


def _build_detector_config(
    kind: str,
    openpcdet_python: str | None,
    openpcdet_repo: Path | None,
    config_path: Path | None,
    checkpoint_path: Path | None,
    score_threshold: float,
) -> DetectorConfig:
    return DetectorConfig(
        kind=kind,
        openpcdet_python=openpcdet_python,
        openpcdet_repo=openpcdet_repo,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        score_threshold=score_threshold,
    )


def _build_compare_detector_config(name: str, args: argparse.Namespace) -> DetectorConfig:
    if name == "pointpillars":
        return _build_detector_config(
            name,
            args.openpcdet_python,
            args.openpcdet_repo,
            args.pointpillars_config,
            args.pointpillars_checkpoint,
            args.ml_score_threshold,
        )
    if name == "centerpoint":
        return _build_detector_config(
            name,
            args.openpcdet_python,
            args.openpcdet_repo,
            args.centerpoint_config,
            args.centerpoint_checkpoint,
            args.ml_score_threshold,
        )
    return _build_detector_config(
        name,
        args.openpcdet_python,
        args.openpcdet_repo,
        None,
        None,
        args.ml_score_threshold,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="depthyn")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay_parser = subparsers.add_parser(
        "replay",
        help="Run the baseline replay pipeline on converted LiDAR CSV frames.",
    )
    replay_parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing converted frame CSV files.",
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
        default=0.75,
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
        choices=("baseline", "pointpillars", "centerpoint"),
        default="baseline",
        help="Detection backend to run during replay.",
    )
    replay_parser.add_argument(
        "--preview-points",
        type=int,
        default=1200,
        help="Maximum downsampled points per frame to embed for viewer playback.",
    )
    _add_openpcdet_options(replay_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Run the same replay across multiple detector backends and write a comparison report.",
    )
    compare_parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing converted frame CSV files.",
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
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames processed.",
    )
    compare_parser.add_argument(
        "--detectors",
        nargs="+",
        default=["baseline"],
        choices=("baseline", "pointpillars", "centerpoint"),
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
        default=0.75,
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
        "--pointpillars-config",
        type=Path,
        default=None,
        help="OpenPCDet config for PointPillars.",
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
        help="OpenPCDet config for CenterPoint.",
    )
    compare_parser.add_argument(
        "--centerpoint-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for CenterPoint.",
    )
    _add_openpcdet_options(compare_parser)

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "replay":
        config = ReplayConfig(
            input_dir=args.input_dir,
            output_json=args.output,
            mode=args.mode,
            max_frames=args.max_frames,
            preview_point_limit=args.preview_points,
            detector=_build_detector_config(
                args.detector,
                args.openpcdet_python,
                args.openpcdet_repo,
                args.ml_config,
                args.ml_checkpoint,
                args.ml_score_threshold,
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
