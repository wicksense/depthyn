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

## Why Start With A Non-ML Baseline?

The product needs more than a detector. It needs a stable end-to-end loop:

```text
source -> frame assembly -> filtering -> detections -> tracks -> zones/alerts -> storage/api/ui
```

The baseline in this repo gets that full loop working before we choose and tune
heavier 3D detection models such as PointPillars or CenterPoint.

## Repository Layout

- `src/depthyn/`: core package
- `docs/architecture.md`: current system design
- `viewer/`: browser replay UI
- `tests/`: unit tests for clustering, tracking, and replay flow
- `MEMORY.md`: persistent project memory for this workspace
