# Depthyn

3D tracking and scene intelligence from LiDAR.

Depthyn is an open source LiDAR perception platform for two workflows:
- live sensor operation
- recorded replay and analysis

The first milestone in this repo is a dependency-light baseline that can replay
converted LiDAR CSV frames, cluster foreground objects, maintain simple 3D
tracks, and write replay bundles we can inspect in a browser while the rest of
the platform comes online.

## Current Capabilities

- replay converted frame CSVs from `lidar-rpi-gps-pipeline`
- support `mobile` and `stationary` processing modes
- downsample and filter LiDAR frames
- build a simple non-ML baseline using clustering + tracking
- expose a pluggable detector interface for ML backends
- compare the baseline against optional MMDetection3D runs
- write JSON replay bundles for downstream API/UI work
- serve a browser replay viewer for recorded sessions

## Sample Run

From the workspace root:

```bash
PYTHONPATH=src python3 -m depthyn.cli replay \
  SampleData/output-26/converted_csv \
  --output artifacts/sampledata-26-summary.json \
  --mode mobile \
  --max-frames 20
```

For a stationary scene, switch to `--mode stationary` to enable background
warmup and foreground suppression.

To view the recorded replay in a browser:

```bash
PYTHONPATH=src python3 -m depthyn.cli serve-viewer \
  --summary artifacts/sampledata-26-summary.json
```

Then open the printed URL.

## Detector Backends

Depthyn now supports detector selection during replay:
- `baseline`: built-in clustering detector
- `centerpoint`: optional MMDetection3D adapter
- `dsvt`: optional MMDetection3D adapter
- `pointpillars`: optional MMDetection3D adapter

Baseline replay:

```bash
PYTHONPATH=src python3 -m depthyn.cli replay \
  SampleData/output-26/converted_csv \
  --output artifacts/baseline-summary.json \
  --detector baseline \
  --mode mobile \
  --max-frames 20
```

Side-by-side comparison report:

```bash
PYTHONPATH=src python3 -m depthyn.cli compare \
  SampleData/output-26/converted_csv \
  --output-dir artifacts/detector-comparison \
  --detectors baseline centerpoint dsvt \
  --mmdet3d-python /path/to/mmdet3d-env/bin/python \
  --mmdet3d-repo /path/to/mmdetection3d \
  --centerpoint-config /path/to/centerpoint.py \
  --centerpoint-checkpoint /path/to/centerpoint.pth \
  --dsvt-config /path/to/dsvt.py \
  --dsvt-checkpoint /path/to/dsvt.pth
```

If the MMDetection3D environment is not configured yet, the comparison report will
still run and mark the ML detectors as configuration errors instead of failing
the whole command.

Manual backend setup lives in [docs/mmdet3d_setup.md](/home/spriteadmin/Documents/LiDAR-Object-Detection/docs/mmdet3d_setup.md).

## MMDetection3D Notes

The current ML adapters target MMDetection3D rather than OpenPCDet. Depthyn
does not vendor or install MMDetection3D automatically. Instead, Depthyn
shells out to `tools/mmdet3d_runner.py` using the Python executable from an
existing MMDetection3D-capable environment.

That keeps the core project lightweight while still giving us a concrete path
to compare the non-ML baseline against modern 3D detectors such as CenterPoint
and DSVT.

## Why Start With A Non-ML Baseline?

The product needs more than a detector. It needs a stable end-to-end loop:

```text
source -> frame assembly -> filtering -> detections -> tracks -> zones/alerts -> storage/api/ui
```

The baseline in this repo gets that full loop working before we choose and tune
heavier 3D detection models such as CenterPoint or DSVT.

## Repository Layout

- `src/depthyn/`: core package
- `docs/architecture.md`: current system design
- `viewer/`: browser replay UI
- `tests/`: unit tests for clustering, tracking, and replay flow
- `MEMORY.md`: persistent project memory for this workspace
