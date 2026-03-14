# Project Memory

Last updated: 2026-03-14

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
- Added a detector abstraction layer:
  - baseline clustering detector is now a first-class backend
  - optional MMDetection3D-backed detector adapter planned for `CenterPoint` and `DSVT`
  - `compare` command added for side-by-side detector runs
- Added helper runner script pattern for out-of-process ML execution
- Current shell environment is still dependency-light:
  - no `pip`
  - no `numpy`
  - no `torch`
  - no `MMDetection3D`
- ML detector execution therefore depends on a separate MMDetection3D-capable Python environment
- Verified locally:
  - `replay --detector baseline` runs successfully on sample data
  - `compare --detectors baseline pointpillars` writes a comparison report
  - unconfigured ML backends currently report clean configuration errors instead of aborting the run
- Pivoted the repo framing from "detector framework first" to "product platform first"
- Added product-facing scene contracts:
  - `src/depthyn/scene/`
  - per-frame `scene_state` payloads in replay bundles
- Added first rule-engine scaffolding:
  - `src/depthyn/rules/`
  - rectangular XY zones
  - `entered`, `dwell`, and `exited` events
- Added optional `--zone-config` support to replay and comparison flows
- Added sample zone config under `examples/zones/sample-yard.json`
- Updated the viewer to render zone overlays and per-frame zone status
- Rewrote docs to describe the platform architecture directly rather than centering MMDetection3D
- Verified locally after the pivot:
  - all 10 unit tests pass
  - sample replay with zone config completes successfully on `SampleData/output-26/converted_csv`
  - zone-aware sample replay metrics for first 5 frames:
    - average filtered points per frame: `10282.8`
    - total detections: `97`
    - total tracks: `25`
    - max active tracks: `25`
    - total zone events: `0`
- Started Stage 1 ML replay work with a framework-agnostic path instead of forcing one model stack into the main environment
- Added `prepare-ml-replay` command:
  - exports filtered replay frames as `XYZI` float32 little-endian binaries
  - writes `manifest.json` for external model runners
- Added `precomputed` detector backend:
  - imports normalized detections from either per-frame JSON files or a single JSON mapping
  - allows external ML outputs to flow through Depthyn tracking, rules, replay, comparison, and viewer
- Replay metrics now include:
  - `label_counts`
  - `avg_detection_score`
- Verified locally for Stage 1:
  - all 13 unit tests pass
  - `prepare-ml-replay` succeeds on `SampleData/output-26/converted_csv`
  - `replay --detector precomputed` succeeds on sample data using normalized imported predictions
  - `compare --detectors baseline precomputed` succeeds on sample data
- Stage 1 is now split into two practical substeps:
  - Stage 1a: ML replay prep/import loop and comparison plumbing
  - Stage 1b: first real learned detector host and model execution
- Added Stage 1b MMDetection3D batch orchestration:
  - `run-mmdet3d-replay` for batch inference over an exported replay manifest
  - `compare-mmdet3d-replay` for export -> inference -> import -> compare in one command
  - `tools/mmdet3d_runner.py` now supports both single-frame and manifest-driven batch inference
- Verified locally for Stage 1b orchestration:
  - all 15 unit tests pass
  - Python compilation passes for `src`, `tests`, and `tools`
  - orchestration logic is tested with mocked MMDetection3D execution
- Not yet verified locally for Stage 1b model execution:
  - real `CenterPoint` inference has not run in this shell yet because the required external MMDetection3D environment is still not available here

## Model Direction
- Start LiDAR-only, not camera-first
- Depthyn is no longer being framed around any one ML framework
- Current optional detector adapter path: `MMDetection3D`
- Practical baseline detector: `CenterPoint`
- Modern detector candidate: `DSVT`
- Optional speed/reference detector: `PointPillars`
- Tracking candidate: lightweight 3D Kalman/Hungarian tracker or CenterPoint tracking-style association
- Zones, alerts, counts, and replay are application logic on top of tracks, not separate ML models
- MVP deployment target is a single LiDAR sensor
- Multi-sensor fusion is a later feature, not an initial requirement
- If/when multi-sensor is added, begin with geometric registration plus track fusion, not learned fusion
- Product framing should stay broad enough for both fixed and mobile capture/replay

## Backend Pivot
- OpenPCDet install/debug effort was high relative to value for this project
- User wants to avoid burning Codex limits on repeated heavy installs and is willing to run install commands directly when needed
- Repo has moved away from OpenPCDet
- Local cleanup removed the temporary `.openpcdet` checkout and Miniforge installer script
- Repo architecture is now platform-first:
  - source adapters
  - scene pipeline
  - tracking
  - rules
  - replay/API/UI
- MMDetection3D remains an optional backend path, not the foundation of the repo

## Open Questions
- Which Gemini features should be in MVP vs later phases?
- What data formats and transport does the existing pipeline already emit?
- Whether we should start with recorded pcap/osf data, live sensor streaming, or both.

## Naming Direction
- Project/repo name: `Depthyn`
- Subtitle direction: `3D tracking and scene intelligence from LiDAR`

## Collaboration Preferences
- After each completed pipeline item, provide the user with the exact commands needed to run and test that item locally.

## Current Product Scope
- Depthyn should support both live sensor operation and recorded data replay
- Depthyn should support both stationary monitoring and moving/mobile LiDAR workflows
- Initial deployment target is one Ouster sensor
- Multi-sensor fusion remains a later feature

## V1 Architecture Direction
- Use one internal frame/scene interface regardless of whether input comes from live Ouster UDP, `pcap`, `osf`, or converted outputs
- Support two pose modes:
  - fixed pose for stationary deployments
  - time-varying pose from SLAM/GPS for mobile recordings
- Build the first working perception loop with preprocessing, motion/static segmentation, clustering, and tracking
- Emit stable `scene_state` payloads that future APIs and UIs can consume
- Build rules on top of tracks and scene state rather than coupling them to a detector framework
- Add learned 3D detection/classification once replay and scene-state plumbing is stable
- Zones, alerts, counts, replay, and analytics are built on top of tracked objects and scene state

## Agreed Development Pipeline

Depthyn will be built in this order and tackled one stage at a time.

### Stage 0: Foundation Replay Stack
- Status: done
- Scope:
  - converted CSV replay
  - baseline clustering detector
  - tracking
  - replay viewer
  - scene-state contract
  - first zone-rule engine
- Exit criteria:
  - replay sample data locally
  - inspect tracks and zones in browser
  - keep tests passing

### Stage 1: ML Detection On Recorded Replay
- Status: in progress
- Scope:
  - add the first real learned detector behind the existing detector interface
  - run it on recorded replay before touching live ingest
  - compare `baseline` vs ML on `SampleData/output-26`
- Preferred first model:
  - `CenterPoint`
- Exit criteria:
  - one real ML backend runs on replayed sample data
  - `compare` writes usable baseline-vs-ML outputs
  - viewer can inspect ML-generated replay bundles
  - imported detections and native model-hosted detections share the same normalized contract

### Stage 2: Detector Evaluation And Tuning
- Status: planned
- Scope:
  - compare baseline and ML behavior on sample recordings
  - document failure modes, class quality, and track stability
  - decide whether `CenterPoint` is good enough or whether to try another backend/model
- Exit criteria:
  - choose the first supported ML detector path for Depthyn
  - record model assumptions and known gaps in docs/memory

### Stage 3: Native Ouster Source Adapters
- Status: planned
- Scope:
  - add Ouster-specific adapters under `src/depthyn/source/`
  - support raw `pcap`, `osf`, and later live UDP
  - keep the rest of Depthyn behind the shared frame contract
- Exit criteria:
  - replay through a raw Ouster path without requiring converted CSV
  - no changes required in tracking/rules/viewer for source swaps

### Stage 4: Live Sensor Runtime
- Status: planned
- Scope:
  - live ingest from one Ouster sensor
  - frame buffering and runtime processing loop
  - reuse the same scene/detector/tracking/rules path as replay
- Exit criteria:
  - one live sensor can feed the runtime continuously
  - scene-state output updates in near real time

### Stage 5: Scene Intelligence Expansion
- Status: planned
- Scope:
  - tripwires
  - counts
  - dwell analytics
  - alert rules
  - event summaries/bookmarks
- Exit criteria:
  - rules produce stable event outputs suitable for UI/API consumers

### Stage 6: API And Operator UI
- Status: planned
- Scope:
  - REST/WebSocket service
  - richer replay controls
  - better operator view for tracks, zones, and events
  - eventual 3D scene view alongside the current 2D replay
- Exit criteria:
  - one service process can publish replay/live scene state to a browser client

### Stage 7: Tracking And Product Hardening
- Status: planned
- Scope:
  - improve tracker quality and smoothing
  - improve class-aware tracks once ML detection is stable
  - add health/status, better artifacts, and deployment docs
- Exit criteria:
  - repeatable local runs
  - clearer product behavior and better debugging surfaces

### Stage 8: Multi-Sensor Fusion
- Status: later
- Scope:
  - only after single-sensor product loop is solid
  - calibration, overlap handling, deduplication, and track handoff
- Exit criteria:
  - multiple sensors can feed one shared scene without rewriting single-sensor logic

## Current Next Step
- Execute Stage 1 first:
  - Stage 1a is now in place
  - Stage 1b orchestration is now in place
  - next is the first verified real `CenterPoint` run on recorded replay
  - compare it against the baseline using the new batch MMDetection3D path
  - keep source ingestion work for the following stage unless ML setup blocks progress
