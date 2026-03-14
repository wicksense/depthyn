# Depthyn Architecture

Depthyn starts with one shared runtime path for replay and live sensing.

## Runtime Flow

```text
source -> frame -> pose mode -> preprocessing -> detections -> tracking -> scene intelligence
```

## Source Layer

V1 source support:
- converted LiDAR CSV frame replay

Planned next:
- live Ouster UDP ingest
- `pcap` replay
- `osf` replay

Each source should normalize into the same internal frame object so the
downstream perception stack does not care whether points came from a file or a
live sensor.

## Pose Modes

- `mobile`: points are treated as sensor-frame or time-varying data, so the
  baseline runs direct clustering without a persistent background model
- `stationary`: the system warms up a background occupancy map and suppresses
  stable cells to emphasize movers

## Perception Baseline

The current baseline is deliberately simple:
- range and height filtering
- voxel downsampling while loading
- XY occupancy clustering
- 3D box estimates from clustered points
- nearest-neighbor track association with a constant-velocity prediction

This is enough to validate ingest, scene state, summaries, and future UI/API
contracts without blocking on ML training or model integration.

## Detector Interface

Replay now routes detections through a detector abstraction:
- `baseline`: built-in clustering detector using foreground occupancy clusters
- `pointpillars`: optional OpenPCDet-backed detector
- `centerpoint`: optional OpenPCDet-backed detector

The key design point is that tracking, replay summaries, and the viewer do not
care which detector produced the boxes. That lets us compare a no-ML baseline
against learned 3D detectors without changing the rest of the stack.

## OpenPCDet Adapter

The ML adapter is intentionally out-of-process:
- Depthyn serializes the current frame to a temp JSON payload
- an external Python executable in an OpenPCDet environment runs
  `tools/openpcdet_runner.py`
- the runner loads the model, performs inference, and writes normalized JSON
  detections back to Depthyn

Why use an out-of-process adapter:
- the main Depthyn environment is currently dependency-light
- PointPillars and CenterPoint need `torch`, `numpy`, and the OpenPCDet stack
- it keeps the replay and viewer tooling usable even before the ML environment
  is installed

## Comparison Flow

The `compare` command runs multiple detector backends over the same replay input
and writes:
- one replay summary per detector
- one `comparison.json` report with status and key metrics

If an optional ML backend is not configured, comparison records that backend as
an error instead of aborting the whole run.

## Recorded Replay Viewer

Depthyn now includes a lightweight browser viewer for replay bundles:
- top-down point preview
- detection boxes
- active track overlays
- short track trails
- playback controls and frame scrubbing

The viewer is intentionally simple and dependency-light so we can inspect real
data now, before the larger API and operator UI arrive.

## Next Layers

- native ML environment bootstrap so PointPillars and CenterPoint can run locally
- zone rules and alerts
- replay timeline and event storage
- REST/WebSocket API
- browser UI with 2D and 3D views
