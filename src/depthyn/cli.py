from __future__ import annotations

import argparse
import json
from pathlib import Path

from depthyn.config import ReplayConfig
from depthyn.pipeline import run_replay, write_summary
from depthyn.viewer import serve_viewer


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
        "--preview-points",
        type=int,
        default=1200,
        help="Maximum downsampled points per frame to embed for viewer playback.",
    )

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
            voxel_size_m=args.voxel_size,
            cluster_cell_size_m=args.cluster_cell_size,
            track_max_distance_m=args.track_max_distance,
        )
        summary = run_replay(config)
        write_summary(summary, config.output_json)
        print(f"Wrote summary to {config.output_json}")
        print(json.dumps(summary["metrics"], indent=2))
        return 0
    if args.command == "serve-viewer":
        serve_viewer(args.summary, args.host, args.port)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
