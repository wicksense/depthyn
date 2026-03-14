# Depthyn Architecture

Depthyn is organized as a perception platform, not as a model-zoo repo.

## Runtime Stack

```text
source
  -> frame contract
  -> preprocessing
  -> detector backend
  -> tracking
  -> scene state
  -> rules
  -> replay/API/UI
```

That split matters because the product value is not only "detect objects." It
is the full loop of ingest, tracking, scene awareness, replay, and operator
logic.

## Layered View

```text
Depthyn platform
  |
  +-- source adapters
  |     live UDP / pcap / osf / converted replay
  |
  +-- scene pipeline
  |     preprocess -> detect -> track -> scene state
  |
  +-- scene intelligence
  |     zones -> counts -> dwell -> alerts
  |
  +-- delivery
        replay bundle -> API -> browser UI
```

## Source Contract

Current:
- converted LiDAR CSV replay

Next:
- live Ouster UDP
- `pcap`
- `osf`

All sources should normalize into the same internal frame object so the
downstream pipeline does not care where the points came from.

## Operating Modes

- `mobile`
  Use direct clustering/tracking on moving or pose-varying captures.
- `stationary`
  Warm a background map and suppress stable cells so scene rules focus on
  movers.

## Scene Pipeline

The current built-in baseline does this:
- range and height filtering
- voxel downsampling during load
- optional stationary-background suppression
- XY occupancy clustering
- 3D box estimation
- nearest-neighbor track association with constant-velocity prediction

Each frame now emits `scene_state`, which is the beginning of the product-facing
contract that future APIs and UIs will consume.

## Scene Intelligence

Depthyn now includes a first rule layer for axis-aligned XY zones:
- zone occupancy
- `entered` events
- `dwell` events
- `exited` events

These rules sit on top of tracks, which is the same product direction we want
for counts, tripwires, dwell analytics, and alerts.

## Detector Backends

Detector backends are replaceable:
- `baseline`
- `precomputed`
- optional `centerpoint`
- optional `dsvt`
- optional `pointpillars`

The important boundary is:

```text
detector backend -> normalized detections -> tracks -> scene state
```

That keeps the rest of the platform stable even if we swap model frameworks.

## Stage 1 ML Replay Boundary

The current preferred ML-evaluation loop is:

```text
converted replay
  -> Depthyn frame export
  -> external detector runner
  -> normalized prediction JSON
  -> Depthyn precomputed detector
  -> tracking / scene state / rules / comparison / viewer
```

Why this exists:
- it lets us compare ML outputs before committing to one heavy framework
- it keeps the replay and product stack usable without `torch` in the main env
- it gives us a stable normalized contract for imported detections

## Optional ML Adapter

The current ML path is intentionally isolated behind an out-of-process adapter.
That keeps the main Depthyn environment dependency-light while we decide what
the long-term ML backend should be.

The repo should not be thought of as "an MMDetection3D project." MMDetection3D
is only one possible detector host behind the adapter boundary.

## Replay Viewer

The browser replay viewer currently supports:
- top-down point preview
- zone overlays
- detection boxes
- active track overlays
- short track trails
- playback controls
- per-frame zone status

## Near-Term Roadmap

- add Ouster SDK-backed live and recorded source adapters
- promote scene-state bundles into a real API surface
- add event storage and replay search
- add richer rule types beyond rectangular zones
- evaluate the first production ML backend behind the detector interface
