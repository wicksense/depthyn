# Depthyn

3D tracking and scene intelligence from LiDAR.

Depthyn is an open source LiDAR perception platform for two workflows:
- live sensor operation
- recorded replay and analysis

The repo is now organized around the product architecture, not around any one
ML framework. Detection backends are pluggable. Replay, tracking, zones,
alerts, scene state, and visualization are the stable core.

## Product Architecture

```text
LiDAR source
  -> input adapter
  -> unified frame contract
  -> preprocessing
  -> detector backend
  -> tracking
  -> scene intelligence
  -> replay bundle / API / UI
```

Current source path:
- converted CSV replay from `lidar-rpi-gps-pipeline`

Planned source paths:
- live Ouster UDP
- `pcap` replay
- `osf` replay

Detector backends can change without rewriting the rest of the stack:

```text
preprocessed frame
  -> baseline clustering detector
  -> optional ML detector backend
  -> normalized detections
  -> tracks, zones, alerts, viewer
```

## Current Capabilities

- replay converted frame CSVs from `lidar-rpi-gps-pipeline`
- support `mobile` and `stationary` processing modes
- downsample and filter LiDAR frames
- run a non-ML baseline using clustering + tracking
- emit product-facing scene state per frame
- evaluate XY zone rules with enter, dwell, and exit events
- write replay bundles for downstream API/UI work
- serve a browser replay viewer for recorded sessions
- compare the baseline against optional ML detector backends

## Quick Start

Replay sample converted data:

```bash
PYTHONPATH=src python3 -m depthyn.cli replay \
  SampleData/output-26/converted_csv \
  --output artifacts/sampledata-26-summary.json \
  --mode mobile \
  --max-frames 20
```

Replay with a sample zone config:

```bash
PYTHONPATH=src python3 -m depthyn.cli replay \
  SampleData/output-26/converted_csv \
  --output artifacts/sampledata-26-zones.json \
  --mode mobile \
  --max-frames 20 \
  --zone-config examples/zones/sample-yard.json
```

View the replay bundle in a browser:

```bash
PYTHONPATH=src python3 -m depthyn.cli serve-viewer \
  --summary artifacts/sampledata-26-zones.json
```

Then open the printed URL.

## Replay Output

Each frame summary now includes:
- preview points
- detections
- active tracks
- `scene_state` with tracked objects
- zone occupancy
- zone events

Top-level replay bundles also include:
- scene bounds
- playback timing
- zone definitions
- aggregate metrics
- event summaries

That structure is intended to become the basis for future REST/WebSocket APIs.

## Detector Backends

Built in:
- `baseline`: clustering + tracking

Optional:
- `centerpoint`
- `dsvt`
- `pointpillars`

Those ML options are currently adapter backends, not the foundation of the
repo. Depthyn can compare them against the baseline, but the platform itself is
designed so that tracking, zones, replay, and UI are not coupled to any single
detector framework.

Manual ML backend setup lives in [docs/mmdet3d_setup.md](/home/spriteadmin/Documents/LiDAR-Object-Detection/docs/mmdet3d_setup.md).

## Repository Layout

- `src/depthyn/source/`: replay and future live input adapters
- `src/depthyn/perception/`: preprocessing and baseline perception
- `src/depthyn/detectors/`: pluggable detector backends
- `src/depthyn/tracking/`: track management
- `src/depthyn/scene/`: scene-state contracts
- `src/depthyn/rules/`: zone rules and event generation
- `viewer/`: browser replay UI
- `docs/`: architecture and setup notes
- `tests/`: unit tests
- `MEMORY.md`: persistent project memory
