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

## Next Layers

- learned 3D detection and classification
- zone rules and alerts
- replay timeline and event storage
- REST/WebSocket API
- browser UI with 2D and 3D views

