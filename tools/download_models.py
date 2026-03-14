"""Download pre-trained ONNX models for Depthyn detector backends."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "centerpoint": {
        "dir": "centerpoint-onnx",
        "files": [
            {
                "name": "pts_voxel_encoder_centerpoint.onnx",
                "url": "https://awf.ml.dev.web.auto/perception/models/centerpoint/v2/pts_voxel_encoder_centerpoint.onnx",
            },
            {
                "name": "pts_backbone_neck_head_centerpoint.onnx",
                "url": "https://awf.ml.dev.web.auto/perception/models/centerpoint/v2/pts_backbone_neck_head_centerpoint.onnx",
            },
        ],
        "source": "Autoware (TIER IV)",
        "license": "Apache-2.0",
    },
}


def download_model(model_name: str, models_root: Path) -> None:
    spec = MODELS.get(model_name)
    if spec is None:
        print(f"Unknown model: {model_name}")
        print(f"Available: {', '.join(MODELS)}")
        sys.exit(1)

    target_dir = models_root / spec["dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Source: {spec['source']}")
    print(f"License: {spec['license']}")
    print(f"Target: {target_dir}")
    print()

    for entry in spec["files"]:
        dest = target_dir / entry["name"]
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"  [skip] {entry['name']} (already exists, {dest.stat().st_size:,} bytes)")
            continue
        print(f"  [download] {entry['name']} ...")
        urllib.request.urlretrieve(entry["url"], str(dest))
        print(f"  [done] {dest.stat().st_size:,} bytes")

    print()
    print("Download complete.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS),
        help="Which model to download.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="Root directory for downloaded models.",
    )
    args = parser.parse_args(argv)
    download_model(args.model, args.models_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
