# Depthyn

3D tracking and scene intelligence from LiDAR.

Depthyn is an open source LiDAR perception platform for two workflows:
- live sensor operation
- recorded replay and analysis

The repo is organized around the product architecture, not around any one
ML framework. Detection backends are pluggable. Replay, tracking, zones,
alerts, scene state, and visualization are the stable core.

## Product Architecture

```text
LiDAR source
  -> input adapter (CSV, pcap, live UDP)
  -> unified frame contract
  -> preprocessing
  -> detector backend
  -> tracking
  -> scene intelligence
  -> replay bundle / API / UI
```

Source adapters:
- **Converted CSV** — replay from `lidar-rpi-gps-pipeline` exports
- **Ouster pcap** — replay raw Ouster sensor captures via `ouster-sdk`
- Planned: live Ouster UDP, OSF replay

Detector backends can change without rewriting the rest of the stack:

```text
preprocessed frame
  -> baseline clustering detector
  -> optional ML detector backend (ONNX CenterPoint, etc.)
  -> normalized detections
  -> tracks, zones, alerts, viewer
```

## Current Capabilities

- Replay converted CSV frames or raw Ouster pcap captures
- Auto-detect source type (CSV vs pcap) from input directory
- Support `mobile` and `stationary` processing modes
- Downsample and filter LiDAR frames
- Run a non-ML baseline using clustering + tracking
- Run ONNX CenterPoint inference with GPU acceleration (car, truck, bus, pedestrian, bicycle)
- Emit product-facing scene state per frame
- Evaluate XY zone rules with enter, dwell, and exit events
- Write replay bundles for downstream API/UI work
- Serve 3D and 2D browser viewers for recorded sessions
- Compare the baseline against optional ML detector backends

## Quick Start

### Replay from raw Ouster pcap

```bash
PYTHONPATH=src python3 -m depthyn.cli replay \
  SampleData/26 \
  --detector centerpoint-onnx \
  --output artifacts/pcap-centerpoint.json \
  --max-frames 20
```

### Replay from converted CSV

```bash
PYTHONPATH=src python3 -m depthyn.cli replay \
  SampleData/output-26/converted_csv \
  --output artifacts/sampledata-26-summary.json \
  --mode mobile \
  --max-frames 20
```

### View results in 3D

```bash
PYTHONPATH=src python3 -m depthyn.cli serve-viewer \
  --summary artifacts/pcap-centerpoint.json
```

Then open the printed 3D viewer URL in a browser.

### Replay with zone rules

```bash
PYTHONPATH=src python3 -m depthyn.cli replay \
  SampleData/output-26/converted_csv \
  --output artifacts/sampledata-26-zones.json \
  --mode mobile \
  --max-frames 20 \
  --zone-config examples/zones/sample-yard.json
```

## ONNX CenterPoint

Depthyn includes an in-process ONNX CenterPoint detector using Autoware v2
PointPillars models. No MMDetection3D installation required.

Download the models:

```bash
python3 tools/download_models.py
```

Run with CenterPoint:

```bash
PYTHONPATH=src python3 -m depthyn.cli replay \
  SampleData/26 \
  --detector centerpoint-onnx \
  --output artifacts/centerpoint-results.json \
  --max-frames 30
```

Classes detected: car, truck, bus, pedestrian, bicycle.

Requirements: `numpy`, `onnxruntime-gpu` (or `onnxruntime` for CPU).

## 3D Viewer

WebGL-based 3D point cloud viewer built with Three.js:

- Height-colored point cloud rendering
- Wireframe bounding boxes with per-class colors and heading rotation
- Floating labels with class name and confidence score
- Track trails showing movement history
- Orbit/pan/zoom controls with ground grid
- Frame playback with slider, speed control, keyboard shortcuts (Space, Arrow keys)
- Dark theme UI

The viewer loads the same replay JSON produced by the `replay` command.

## Ouster Pcap Source

Read raw Ouster pcap captures directly without CSV conversion:

```bash
PYTHONPATH=src python3 -m depthyn.cli replay \
  SampleData/26 \
  --detector baseline \
  --max-frames 10
```

Source type is auto-detected from file extensions. Override with `--source-type pcap` or `--source-type csv`.

Requirements: `ouster-sdk`, `numpy`.

## ML Replay Workflow

Depthyn supports a model-host-agnostic workflow for recorded-data ML evaluation:

1. Export filtered replay frames in ML-friendly `XYZI` binary format
2. Run any external detector against those exported frames
3. Import the normalized detections back into Depthyn
4. Compare them against the built-in baseline on the same replay

Export a replay bundle for ML runners:

```bash
PYTHONPATH=src python3 -m depthyn.cli prepare-ml-replay \
  SampleData/output-26/converted_csv \
  --output-dir artifacts/ml-replay \
  --max-frames 20
```

Compare baseline vs imported ML detections:

```bash
PYTHONPATH=src python3 -m depthyn.cli compare \
  SampleData/output-26/converted_csv \
  --output-dir artifacts/detector-comparison \
  --detectors baseline precomputed \
  --precomputed-path /path/to/predictions
```

Prediction input formats:
- a directory of per-frame JSON files named `<frame_id>.json`
- a single JSON file containing frame-to-detections mappings

Each detection entry should include:
- `centroid`
- `bbox_min`
- `bbox_max`

Optional fields:
- `detection_id`
- `label`
- `score`
- `heading_rad`
- `point_count`
- `cell_count`

## Replay Output

Each frame summary includes:
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
- `baseline`: clustering + tracking (no ML required)
- `centerpoint-onnx`: ONNX CenterPoint with PointPillars voxelization (GPU or CPU)
- `precomputed`: imported normalized detections from any external model runner

Optional (require MMDetection3D):
- `centerpoint`
- `dsvt`
- `pointpillars`

Manual ML backend setup lives in [docs/mmdet3d_setup.md](docs/mmdet3d_setup.md).

## Repository Layout

- `src/depthyn/source/`: input adapters (CSV, Ouster pcap)
- `src/depthyn/perception/`: preprocessing and baseline perception
- `src/depthyn/detectors/`: pluggable detector backends
- `src/depthyn/tracking/`: track management
- `src/depthyn/scene/`: scene-state contracts
- `src/depthyn/rules/`: zone rules and event generation
- `viewer/`: browser replay UI (3D and 2D viewers)
- `tools/`: model download and utility scripts
- `docs/`: architecture and setup notes
- `tests/`: unit tests
