# Project Memory

Last updated: 2026-03-13

## Goal
Build an open source LiDAR perception platform inspired by Ouster Gemini, using `/home/spriteadmin/Documents/LiDAR-Object-Detection` as the working directory and leveraging the VM GPU for local development and evaluation.

## Known Inputs
- Reference capture pipeline repo: `wicksense/lidar-rpi-gps-pipeline`
- Product target reference: Ouster Gemini public product pages
- User noted `gh` is already authenticated on this machine
- Current workspace started empty
- VM is Ubuntu 24.04 on Azure with Python `3.12.3`
- `lspci` shows an NVIDIA RTX 4000 Ada GPU is attached
- User confirmed `nvidia-smi` works on the host shell with driver `570.133.07` and CUDA `12.8`
- Treat GPU acceleration as available; earlier failure likely came from the agent execution environment rather than the VM itself
- Sample dataset available at `SampleData/26`
- Converted outputs from the existing pipeline available at `SampleData/output-26`
- Sample raw session includes manifests, GPS CSV, and multiple Ouster `.pcap` chunks
- Sample converted session includes LAS, COPC/LAZ, OSF, QA JSON, and converted CSV outputs

## Existing Pipeline Summary
- Raspberry Pi captures raw Ouster packets plus GPS logs in the field
- Offline workstation pipeline supports `gps_fusion`, `slam_map`, and `slam_gps_anchor`
- Existing outputs include LAS, optional OSF playback, QA JSON, and manifest-driven session processing
- Current repo is strong on ingestion, timestamping, GPS alignment, and offline map generation
- Current repo does not yet provide Gemini-style live scene monitoring, multi-sensor meshing service, real-time tracking, zones, alerts, or an operations UI
- The broader product scope should support both stationary monitoring and moving/mobile LiDAR recording workflows

## Gemini-Like Capabilities To Recreate
- Real-time point cloud ingestion from Ouster sensors
- Multi-sensor calibration / meshing into a shared scene
- Detection, classification, and tracking
- 2D and 3D scene visualization / digital twin
- Zones, alerts, counts, and analytics
- Event recording and replay
- Device health / diagnostics
- API-first architecture and deployable open source stack

## Immediate Next Steps
1. Inspect the existing Raspberry Pi + Ouster + GPS pipeline repo and extract reusable components.
2. Define the MVP scope for the open source Gemini-like platform.
3. Stand up a repo skeleton with ingestion, perception, storage, and visualization modules.
4. Choose the local inference/runtime stack around the available RTX 4000 Ada GPU.
5. Use `SampleData/26` and `SampleData/output-26` as the initial local test corpus for replay, scene building, and model evaluation.

## Build Progress
- Initialized the `Depthyn` project structure with `pyproject.toml`, README, docs, tests, and a `src/depthyn` package
- Implemented a first working replay pipeline for converted CSV frames
- Implemented two operating modes:
  - `mobile`: direct clustering and tracking
  - `stationary`: background warmup plus foreground suppression
- Implemented a non-ML baseline:
  - frame loading
  - voxel downsampling
  - range and Z filtering
  - XY occupancy clustering
  - simple nearest-neighbor 3D tracking
- Added CLI entrypoint: `python3 -m depthyn.cli replay ...`
- Added unit tests for clustering, tracking, and replay flow
- Verified tests pass locally
- Verified the pipeline runs on `SampleData/output-26/converted_csv`
- Initial sample run on first 10 frames produced:
  - average filtered points per frame: `10263.7`
  - total detections: `201`
  - total tracks: `29`
  - max active tracks: `28`
- Local summary artifact written to `artifacts/sampledata-26-summary.json`
- Git repository initialized locally on branch `main`
- Initial commit created: `8341805` (`Bootstrap Depthyn replay pipeline`)
- GitHub repository created and pushed: `https://github.com/wicksense/depthyn`
- Added recorded-data visualization:
  - replay bundle now includes preview points, active tracks, scene bounds, and playback metadata
  - browser viewer added under `viewer/`
  - local `serve-viewer` command added
- Verified recorded-data viewer path end to end:
  - replay bundle generated successfully for first 20 sample frames
  - `viewer/index.html` served successfully over local HTTP
  - `artifacts/sampledata-26-summary.json` served successfully over local HTTP

## Model Direction
- Start LiDAR-only, not camera-first
- MVP detector candidate: `PointPillars` or `CenterPoint`
- Tracking candidate: lightweight 3D Kalman/Hungarian tracker or CenterPoint tracking-style association
- Zones, alerts, counts, and replay are application logic on top of tracks, not separate ML models
- MVP deployment target is a single LiDAR sensor
- Multi-sensor fusion is a later feature, not an initial requirement
- If/when multi-sensor is added, begin with geometric registration plus track fusion, not learned fusion
- Product framing should stay broad enough for both fixed and mobile capture/replay

## Open Questions
- Which Gemini features should be in MVP vs later phases?
- What data formats and transport does the existing pipeline already emit?
- Whether we should start with recorded pcap/osf data, live sensor streaming, or both.

## Naming Direction
- Avoid generic descriptive repo names as the primary identity
- Avoid names that imply multi-sensor fusion as a core day-one feature
- Prefer a brandable coined name, with a clear descriptive subtitle in the README
- `Trackora` was considered but is not final due to search/name conflicts
- Current subtitle direction: `3D tracking and scene intelligence from LiDAR`

## Current Product Scope
- Trackora should support both live sensor operation and recorded data replay
- Trackora should support both stationary monitoring and moving/mobile LiDAR workflows
- Initial deployment target is one Ouster sensor
- Multi-sensor fusion remains a later feature

## V1 Architecture Direction
- Use one internal frame/scene interface regardless of whether input comes from live Ouster UDP, `pcap`, `osf`, or converted outputs
- Support two pose modes:
  - fixed pose for stationary deployments
  - time-varying pose from SLAM/GPS for mobile recordings
- Build the first working perception loop with preprocessing, motion/static segmentation, clustering, and tracking
- Add learned 3D detection/classification once replay and scene-state plumbing is stable
- Zones, alerts, counts, replay, and analytics are built on top of tracked objects and scene state
