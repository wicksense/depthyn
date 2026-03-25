import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ─── Color palette ───────────────────────────────────────────────

const COLORS = {
  car:        { hex: "#e8913a", r: 0.91, g: 0.57, b: 0.23 },
  truck:      { hex: "#b07de8", r: 0.69, g: 0.49, b: 0.91 },
  bus:        { hex: "#9d6fd4", r: 0.62, g: 0.44, b: 0.83 },
  pedestrian: { hex: "#3dcc7a", r: 0.24, g: 0.80, b: 0.48 },
  bicycle:    { hex: "#42a5d9", r: 0.26, g: 0.65, b: 0.85 },
  object:     { hex: "#e8913a", r: 0.91, g: 0.57, b: 0.23 },
  track:      { hex: "#3dcc7a", r: 0.24, g: 0.80, b: 0.48 },
  grid:       { hex: "#1a1f2e", r: 0.10, g: 0.12, b: 0.18 },
  ground:     { hex: "#0c0e13" },
  point:      { hex: "#8b6fbf" },
};

function labelColor(label) {
  return COLORS[(label || "object").toLowerCase()] || COLORS.object;
}

function normalizedLabel(label) {
  return (label || "object").toLowerCase();
}

function isLabelVisible(label) {
  return state.classVisibility[normalizedLabel(label)] !== false;
}

function formatLabel(label) {
  const value = normalizedLabel(label);
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function formatEventType(eventType) {
  const value = String(eventType || "event");
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function formatDirection(direction) {
  if (!direction) return "";
  return String(direction)
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function eventTypeColor(eventType) {
  switch (eventType) {
    case "entered":
      return "#7ee787";
    case "dwell":
      return "#ffd166";
    case "exited":
      return "#ff8a80";
    case "crossed":
      return "#59b8ff";
    default:
      return "#8c93b0";
  }
}

function isScanlineMode() {
  return state.viewMode === "scanline";
}

function currentReferenceFrame() {
  return state.referenceFrame || state.bundle?.reference_frame || "sensor";
}

function scanlineModeSupported() {
  return currentReferenceFrame() === "sensor";
}

function filteredEventTimeline() {
  return (state.eventTimeline || []).filter((event) => {
    if (state.eventFilters.type !== "all" && event.event_type !== state.eventFilters.type) {
      return false;
    }
    if (state.eventFilters.label !== "all" && normalizedLabel(event.label) !== state.eventFilters.label) {
      return false;
    }
    if (state.eventFilters.rule !== "all" && event.rule_id !== state.eventFilters.rule) {
      return false;
    }
    return true;
  });
}

function cloneZoneDefinition(zone) {
  return {
    zone_id: zone.zone_id,
    name: zone.name,
    min_xy: [...(zone.min_xy || [0, 0])],
    max_xy: [...(zone.max_xy || [0, 0])],
    kind: zone.kind || "inclusion",
    color: zone.color || "#6f8f77",
    dwell_alert_seconds: Number(zone.dwell_alert_seconds || 0),
    tags: [...(zone.tags || [])],
  };
}

function cloneTripwireDefinition(tripwire) {
  return {
    tripwire_id: tripwire.tripwire_id,
    name: tripwire.name,
    start_xy: [...(tripwire.start_xy || [0, 0])],
    end_xy: [...(tripwire.end_xy || [0, 0])],
    color: tripwire.color || "#59b8ff",
    positive_direction_label: tripwire.positive_direction_label || "negative_to_positive",
    negative_direction_label: tripwire.negative_direction_label || "positive_to_negative",
    tags: [...(tripwire.tags || [])],
  };
}

function emptyRuleState() {
  return { zones: [], tripwires: [] };
}

function currentRuleState() {
  const frame = currentReferenceFrame();
  if (!state.ruleSets[frame]) {
    state.ruleSets[frame] = emptyRuleState();
  }
  return state.ruleSets[frame];
}

function initializeRuleSets(bundle) {
  const availableFrames = new Set(bundle?.available_reference_frames || [bundle?.reference_frame || "sensor"]);
  state.ruleSets = {};
  for (const frame of availableFrames) {
    state.ruleSets[frame] = emptyRuleState();
  }
  const bundleFrame = bundle?.reference_frame || "sensor";
  if (!state.ruleSets[bundleFrame]) {
    state.ruleSets[bundleFrame] = emptyRuleState();
  }
  state.ruleSets[bundleFrame] = {
    zones: (bundle?.zone_definitions || []).map(cloneZoneDefinition),
    tripwires: (bundle?.tripwire_definitions || []).map(cloneTripwireDefinition),
  };
}

function nextRuleId(kind) {
  const rules = currentRuleState();
  const items = kind === "zone" ? rules.zones : rules.tripwires;
  const prefix = kind === "zone" ? "zone" : "tripwire";
  return `${prefix}-${items.length + 1}`;
}

function roundPoint(value) {
  return Math.round(value * 1000) / 1000;
}

function startRuleDrawing(mode) {
  state.authoring = { mode, points: [], hoverPoint: null };
  stopPlayback();
  buildRuleOverlays();
  updateSidebar();
}

function cancelRuleDrawing() {
  state.authoring = { mode: null, points: [], hoverPoint: null };
  buildRuleOverlays();
  updateSidebar();
}

function addAuthoredZone(pointA, pointB) {
  const rules = currentRuleState();
  rules.zones.push({
    zone_id: nextRuleId("zone"),
    name: `Zone ${rules.zones.length + 1}`,
    min_xy: [roundPoint(Math.min(pointA[0], pointB[0])), roundPoint(Math.min(pointA[1], pointB[1]))],
    max_xy: [roundPoint(Math.max(pointA[0], pointB[0])), roundPoint(Math.max(pointA[1], pointB[1]))],
    kind: "inclusion",
    color: "#6f8f77",
    dwell_alert_seconds: 0,
    tags: [],
  });
}

function addAuthoredTripwire(pointA, pointB) {
  const rules = currentRuleState();
  rules.tripwires.push({
    tripwire_id: nextRuleId("tripwire"),
    name: `Tripwire ${rules.tripwires.length + 1}`,
    start_xy: [roundPoint(pointA[0]), roundPoint(pointA[1])],
    end_xy: [roundPoint(pointB[0]), roundPoint(pointB[1])],
    color: "#59b8ff",
    positive_direction_label: "negative_to_positive",
    negative_direction_label: "positive_to_negative",
    tags: [],
  });
}

function commitRulePoint(point) {
  if (!state.authoring.mode) return;
  state.authoring.points.push(point);
  if (state.authoring.mode === "zone" && state.authoring.points.length >= 2) {
    addAuthoredZone(state.authoring.points[0], state.authoring.points[1]);
    if (state.rulePreviewApplied) {
      applyRulesToReplay();
    }
    cancelRuleDrawing();
    return;
  }
  if (state.authoring.mode === "tripwire" && state.authoring.points.length >= 2) {
    addAuthoredTripwire(state.authoring.points[0], state.authoring.points[1]);
    if (state.rulePreviewApplied) {
      applyRulesToReplay();
    }
    cancelRuleDrawing();
    return;
  }
  buildRuleOverlays();
  updateSidebar();
}

function exportRulesJson() {
  const rules = currentRuleState();
  const payload = {
    reference_frame: currentReferenceFrame(),
    zones: rules.zones.map(cloneZoneDefinition),
    tripwires: rules.tripwires.map(cloneTripwireDefinition),
  };
  downloadText(
    `depthyn-rules-${currentReferenceFrame()}.json`,
    JSON.stringify(payload, null, 2),
    "application/json",
  );
}

function cloneBundle(bundle) {
  return JSON.parse(JSON.stringify(bundle));
}

function trackListForReference(frame, referenceFrame) {
  if (referenceFrame === "sensor") {
    return frame.sensor_active_tracks || frame.active_tracks || [];
  }
  return frame.active_tracks || frame.sensor_active_tracks || [];
}

function pointInsideZone(point, zone) {
  return (
    point[0] >= zone.min_xy[0]
    && point[0] <= zone.max_xy[0]
    && point[1] >= zone.min_xy[1]
    && point[1] <= zone.max_xy[1]
  );
}

function orientation2D(a, b, c) {
  return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]));
}

function pointOnSegment(a, b, c, epsilon = 1e-6) {
  if (Math.abs(orientation2D(a, b, c)) > epsilon) return false;
  return (
    c[0] >= Math.min(a[0], b[0]) - epsilon
    && c[0] <= Math.max(a[0], b[0]) + epsilon
    && c[1] >= Math.min(a[1], b[1]) - epsilon
    && c[1] <= Math.max(a[1], b[1]) + epsilon
  );
}

function segmentsIntersect2D(a, b, c, d, epsilon = 1e-6) {
  const abc = orientation2D(a, b, c);
  const abd = orientation2D(a, b, d);
  const cda = orientation2D(c, d, a);
  const cdb = orientation2D(c, d, b);
  if ((abc * abd) < -epsilon && (cda * cdb) < -epsilon) {
    return true;
  }
  return (
    pointOnSegment(a, b, c, epsilon)
    || pointOnSegment(a, b, d, epsilon)
    || pointOnSegment(c, d, a, epsilon)
    || pointOnSegment(c, d, b, epsilon)
  );
}

function crossingDirection(previousPoint, currentPoint, tripwire, epsilon = 1e-6) {
  if (!segmentsIntersect2D(previousPoint, currentPoint, tripwire.start_xy, tripwire.end_xy, epsilon)) {
    return null;
  }
  const edgeX = tripwire.end_xy[0] - tripwire.start_xy[0];
  const edgeY = tripwire.end_xy[1] - tripwire.start_xy[1];
  const motionX = currentPoint[0] - previousPoint[0];
  const motionY = currentPoint[1] - previousPoint[1];
  const positiveNormalX = edgeY;
  const positiveNormalY = -edgeX;
  const projection = (motionX * positiveNormalX) + (motionY * positiveNormalY);
  if (projection > epsilon) return tripwire.positive_direction_label;
  if (projection < -epsilon) return tripwire.negative_direction_label;
  return null;
}

function evaluateRulesForBundle(bundle, referenceFrame, rules) {
  const derived = cloneBundle(bundle);
  const activeMemberships = new Map();
  const dwellEmitted = new Set();
  const lastPositions = new Map();
  const allEvents = [];

  for (const frame of derived.frame_summaries) {
    const tracks = [...trackListForReference(frame, referenceFrame)].sort((a, b) => a.track_id - b.track_id);
    const frameEvents = [];
    const zoneOccupancy = [];
    const currentMemberships = new Set();

    for (const zone of rules.zones) {
      const insideTrackIds = [];
      for (const track of tracks) {
        if (!pointInsideZone(track.centroid, zone)) continue;
        insideTrackIds.push(track.track_id);
        const membershipKey = `${track.track_id}::${zone.zone_id}`;
        currentMemberships.add(membershipKey);
        if (!activeMemberships.has(membershipKey)) {
          activeMemberships.set(membershipKey, frame.timestamp_ns);
          frameEvents.push({
            event_type: "entered",
            zone_id: zone.zone_id,
            zone_name: zone.name,
            track_id: track.track_id,
            timestamp_ns: frame.timestamp_ns,
            rule_kind: "zone",
          });
          continue;
        }

        const dwellSeconds = (frame.timestamp_ns - activeMemberships.get(membershipKey)) / 1_000_000_000;
        if (
          Number(zone.dwell_alert_seconds || 0) > 0
          && dwellSeconds >= Number(zone.dwell_alert_seconds || 0)
          && !dwellEmitted.has(membershipKey)
        ) {
          dwellEmitted.add(membershipKey);
          frameEvents.push({
            event_type: "dwell",
            zone_id: zone.zone_id,
            zone_name: zone.name,
            track_id: track.track_id,
            timestamp_ns: frame.timestamp_ns,
            rule_kind: "zone",
            dwell_seconds: Math.round(dwellSeconds * 1000) / 1000,
          });
        }
      }
      zoneOccupancy.push({
        zone_id: zone.zone_id,
        zone_name: zone.name,
        kind: zone.kind || "inclusion",
        color: zone.color || "#6f8f77",
        track_ids: insideTrackIds,
        object_count: insideTrackIds.length,
        dwell_alert_seconds: Number(zone.dwell_alert_seconds || 0),
      });
    }

    for (const membershipKey of [...activeMemberships.keys()].sort()) {
      if (currentMemberships.has(membershipKey)) continue;
      const [trackIdText, zoneId] = membershipKey.split("::");
      const zone = rules.zones.find((item) => item.zone_id === zoneId);
      if (!zone) continue;
      const dwellSeconds = (frame.timestamp_ns - activeMemberships.get(membershipKey)) / 1_000_000_000;
      activeMemberships.delete(membershipKey);
      dwellEmitted.delete(membershipKey);
      frameEvents.push({
        event_type: "exited",
        zone_id: zone.zone_id,
        zone_name: zone.name,
        track_id: Number(trackIdText),
        timestamp_ns: frame.timestamp_ns,
        rule_kind: "zone",
        dwell_seconds: Math.round(dwellSeconds * 1000) / 1000,
      });
    }

    for (const track of tracks) {
      const previousPoint = lastPositions.get(track.track_id);
      if (!previousPoint) continue;
      const prev2D = [previousPoint[0], previousPoint[1]];
      const current2D = [track.centroid[0], track.centroid[1]];
      for (const tripwire of rules.tripwires) {
        const direction = crossingDirection(prev2D, current2D, tripwire);
        if (!direction) continue;
        frameEvents.push({
          event_type: "crossed",
          zone_id: tripwire.tripwire_id,
          zone_name: tripwire.name,
          track_id: track.track_id,
          timestamp_ns: frame.timestamp_ns,
          rule_kind: "tripwire",
          direction,
        });
      }
    }

    lastPositions.clear();
    for (const track of tracks) {
      lastPositions.set(track.track_id, [...track.centroid]);
    }

    allEvents.push(...frameEvents);
    frame.scene_state.zones = zoneOccupancy;
    frame.scene_state.events = frameEvents;
    frame.scene_state.zone_count = zoneOccupancy.length;
    frame.scene_state.event_count = frameEvents.length;
  }

  derived.zone_definitions = rules.zones.map(cloneZoneDefinition);
  derived.tripwire_definitions = rules.tripwires.map(cloneTripwireDefinition);
  derived.event_summaries = allEvents;
  derived.metrics = {
    ...(derived.metrics || {}),
    total_zone_events: allEvents.length,
  };
  derived.rule_preview = {
    applied: true,
    reference_frame: referenceFrame,
  };
  return derived;
}

function applyRulesToReplay() {
  if (!state.sourceBundle) return;
  state.bundle = evaluateRulesForBundle(state.sourceBundle, currentReferenceFrame(), currentRuleState());
  state.rulePreviewApplied = true;
  state.eventTimeline = buildEventTimeline(state.bundle);
}

function resetRulePreview() {
  if (!state.sourceBundle) return;
  state.bundle = cloneBundle(state.sourceBundle);
  state.rulePreviewApplied = false;
  state.eventTimeline = buildEventTimeline(state.bundle);
}

// ─── State ───────────────────────────────────────────────────────

const state = {
  sourceBundle: null,
  bundle: null,
  frameIndex: 0,
  playing: false,
  timer: null,
  selected: null,  // selected object id
  viewMode: "spatial",
  referenceFrame: "sensor",
  classVisibility: {},
  eventTimeline: [],
  eventFilters: {
    type: "all",
    label: "all",
    rule: "all",
  },
  ruleSets: {},
  authoring: {
    mode: null,
    points: [],
    hoverPoint: null,
  },
  rulePreviewApplied: false,
  pointOwnership: {
    byDetectionId: new Map(),
  },
  overlays: {
    rawPoints: true,
    objectPoints: true,
    selectedPoints: true,
    boxes: true,
    trails: true,
    motionVector: true,
    events: true,
    worldAxes: true,
    egoMarker: true,
  },
};

// ─── Three.js setup ──────────────────────────────────────────────

const container = document.getElementById("canvas-container");
const rangeCanvas = document.getElementById("range-canvas");
const rangeCtx = rangeCanvas.getContext("2d");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0c0e13);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0c0e13, 0.006);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.5, 500);
camera.position.set(0, -45, 35);
camera.up.set(0, 0, 1);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 3;
controls.maxDistance = 200;
controls.target.set(0, 0, 0);

// Raycaster for click-to-inspect
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.5;
const mouse = new THREE.Vector2();
let mouseDownPos = null;

// ─── Lights ──────────────────────────────────────────────────────

scene.add(new THREE.AmbientLight(0x404860, 1.0));
const dirLight = new THREE.DirectionalLight(0xc0c8e0, 0.4);
dirLight.position.set(30, -20, 50);
scene.add(dirLight);

// ─── Ground grid ─────────────────────────────────────────────────

let worldAxesGroup = null;

function createGroundGrid() {
  const gridGroup = new THREE.Group();
  const axesGroup = new THREE.Group();

  const grid = new THREE.GridHelper(200, 40, 0x1a2030, 0x141826);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.01;
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  gridGroup.add(grid);

  const axLen = 6;
  const axMat = (color) => new THREE.LineBasicMaterial({ color, linewidth: 2, transparent: true, opacity: 0.6 });
  const makeLine = (pts, color) => {
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    return new THREE.Line(g, axMat(color));
  };
  axesGroup.add(makeLine([new THREE.Vector3(0,0,0), new THREE.Vector3(axLen,0,0)], 0xff4444));
  axesGroup.add(makeLine([new THREE.Vector3(0,0,0), new THREE.Vector3(0,axLen,0)], 0x44ff44));
  axesGroup.add(makeLine([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,axLen)], 0x4488ff));

  scene.add(gridGroup);
  scene.add(axesGroup);
  worldAxesGroup = axesGroup;
}
createGroundGrid();

// ─── Scene objects (rebuilt per frame) ───────────────────────────

let pointCloud = null;
let objectPointCloud = null;
let objectSilhouetteCloud = null;
let highlightPointCloud = null;
let boxGroup = new THREE.Group();
let trailGroup = new THREE.Group();
let motionGroup = new THREE.Group();
let eventGroup = new THREE.Group();
let egoGroup = new THREE.Group();
let ruleGroup = new THREE.Group();
scene.add(boxGroup);
scene.add(trailGroup);
scene.add(motionGroup);
scene.add(eventGroup);
scene.add(egoGroup);
scene.add(ruleGroup);

const labelContainer = document.createElement("div");
labelContainer.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:5;overflow:hidden;";
document.body.appendChild(labelContainer);

// Store box meshes for raycasting
let clickableBoxes = [];

// ─── Circle sprite for round points ─────────────────────────────

const circleTexture = (() => {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
})();

// ─── Point cloud ─────────────────────────────────────────────────

function assignPointOwner(point, detections) {
  if (!detections?.length) return null;

  let best = null;
  let bestDist = Infinity;
  for (const detection of detections) {
    if (!pointInDetectionVolume(point, detection)) continue;
    const dx = point[0] - detection.centroid[0];
    const dy = point[1] - detection.centroid[1];
    const dz = point[2] - detection.centroid[2];
    const dist = dx * dx + dy * dy + dz * dz;
    if (dist < bestDist) {
      best = detection;
      bestDist = dist;
    }
  }
  return best;
}

function rawPointsForFrame(frame) {
  if (!frame) return [];
  if (currentReferenceFrame() === "sensor") {
    return frame.sensor_preview_points || frame.preview_points || [];
  }
  return frame.preview_points || [];
}

function detailPointsForFrame(frame) {
  const detail = currentReferenceFrame() === "sensor"
    ? (frame?.sensor_detail_points || frame?.detail_points || [])
    : (frame?.detail_points || []);
  return detail.length ? detail : rawPointsForFrame(frame);
}

function detectionsForFrame(frame) {
  if (!frame) return [];
  if (currentReferenceFrame() === "sensor") {
    return frame.sensor_detections || frame.detections || [];
  }
  return frame.detections || [];
}

function activeTracksForFrame(frame) {
  if (!frame) return [];
  if (currentReferenceFrame() === "sensor") {
    return frame.sensor_active_tracks || frame.active_tracks || [];
  }
  return frame.active_tracks || [];
}

function framePoseForCurrentReference(frame) {
  return currentReferenceFrame() === "world" ? frame?.frame_pose : null;
}

function scanlinePointsForFrame(frame) {
  return frame?.scanline_points || [];
}

function spatialUsesSensorSamples(frame) {
  const referenceFrame = currentReferenceFrame();
  return referenceFrame === "sensor" && scanlinePointsForFrame(frame).length > 0;
}

function samplePosition(sample) {
  return [sample[2], sample[3], sample[4]];
}

function sampleRange(sample) {
  return sample[5];
}

function colorLerp(from, to, alpha) {
  return [
    from[0] + (to[0] - from[0]) * alpha,
    from[1] + (to[1] - from[1]) * alpha,
    from[2] + (to[2] - from[2]) * alpha,
  ];
}

function sensorSampleBaseColor(frame, sample) {
  const shape = scanlineShapeForFrame(frame);
  const rowT = shape ? sample[0] / Math.max(1, shape[0] - 1) : 0.5;
  const rangeT = Math.max(0, Math.min(1, sampleRange(sample) / 60));
  const near = [0.98, 0.28, 0.36];
  const mid = [0.78, 0.30, 0.62];
  const far = [0.46, 0.34, 0.86];
  const rangeMix = rangeT < 0.55
    ? colorLerp(near, mid, rangeT / 0.55)
    : colorLerp(mid, far, (rangeT - 0.55) / 0.45);
  const lift = 0.06 + rowT * 0.12;
  return [
    Math.min(1, rangeMix[0] + lift * 0.35),
    Math.min(1, rangeMix[1] + lift * 0.15),
    Math.min(1, rangeMix[2] + lift * 0.55),
  ];
}

function scanlineShapeForFrame(frame) {
  return frame?.scanline_shape || state.bundle?.scanline_metadata?.shape || null;
}

function scanlineShifts() {
  return state.bundle?.scanline_metadata?.pixel_shift_by_row || [];
}

function mod(value, base) {
  return ((value % base) + base) % base;
}

function destaggeredSampleColumn(frame, sample) {
  const shape = scanlineShapeForFrame(frame);
  if (!shape) return sample[1];
  const shifts = scanlineShifts();
  const shift = shifts[sample[0]] || 0;
  return mod(sample[1] + shift, shape[1]);
}

function scanlineCellGeometry(frame, sample, width, height, padding = 0) {
  const shape = scanlineShapeForFrame(frame);
  if (!shape) {
    return { x: 0, y: 0, cellWidth: width, cellHeight: height };
  }
  const rows = Math.max(1, shape[0]);
  const cols = Math.max(1, shape[1]);
  const innerWidth = Math.max(1, width - padding * 2);
  const innerHeight = Math.max(1, height - padding * 2);
  const cellWidth = innerWidth / cols;
  const cellHeight = innerHeight / rows;
  const destaggeredCol = destaggeredSampleColumn(frame, sample);
  return {
    x: padding + destaggeredCol * cellWidth,
    y: padding + sample[0] * cellHeight,
    cellWidth,
    cellHeight,
  };
}

function circularColumnSegments(columns, totalCols) {
  if (!columns.length || totalCols <= 0) return [];
  const sorted = [...new Set(columns.map((value) => mod(value, totalCols)))].sort((a, b) => a - b);
  if (sorted.length === 1) {
    return [[sorted[0], sorted[0]]];
  }
  let maxGap = -1;
  let maxGapIndex = 0;
  for (let index = 0; index < sorted.length; index++) {
    const current = sorted[index];
    const next = sorted[(index + 1) % sorted.length] + (index === sorted.length - 1 ? totalCols : 0);
    const gap = next - current;
    if (gap > maxGap) {
      maxGap = gap;
      maxGapIndex = index;
    }
  }
  const start = sorted[(maxGapIndex + 1) % sorted.length];
  const end = sorted[maxGapIndex];
  if (start <= end) {
    return [[start, end]];
  }
  return [
    [0, end],
    [start, totalCols - 1],
  ];
}

function drawScanlineSampleSet(frame, samples, color, alpha, width, height, options = {}) {
  if (!samples.length) return;
  const padding = options.padding || 0;
  const scale = options.scale || 1;
  rangeCtx.fillStyle = color;
  rangeCtx.globalAlpha = alpha;
  for (const sample of samples) {
    const cell = scanlineCellGeometry(frame, sample, width, height, padding);
    const drawWidth = Math.max(1, cell.cellWidth * scale);
    const drawHeight = Math.max(1, cell.cellHeight * scale);
    rangeCtx.fillRect(
      cell.x - (drawWidth - cell.cellWidth) / 2,
      cell.y - (drawHeight - cell.cellHeight) / 2,
      drawWidth,
      drawHeight,
    );
  }
  rangeCtx.globalAlpha = 1;
}

function buildPointCloud(frame, detections = [], selectedInfo = null) {
  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
    pointCloud.material.dispose();
  }
  if (objectPointCloud) {
    scene.remove(objectPointCloud);
    objectPointCloud.geometry.dispose();
    objectPointCloud.material.dispose();
    objectPointCloud = null;
  }
  if (objectSilhouetteCloud) {
    scene.remove(objectSilhouetteCloud);
    objectSilhouetteCloud.geometry.dispose();
    objectSilhouetteCloud.material.dispose();
    objectSilhouetteCloud = null;
  }
  if (highlightPointCloud) {
    scene.remove(highlightPointCloud);
    highlightPointCloud.geometry.dispose();
    highlightPointCloud.material.dispose();
    highlightPointCloud = null;
  }
  const sensorSamples = spatialUsesSensorSamples(frame) ? scanlinePointsForFrame(frame) : [];
  const displayPoints = sensorSamples.length ? sensorSamples.map(samplePosition) : rawPointsForFrame(frame);
  const ownershipPoints = sensorSamples.length ? sensorSamples.map(samplePosition) : detailPointsForFrame(frame);

  if (!displayPoints?.length && !ownershipPoints?.length) {
    state.pointOwnership = { byDetectionId: new Map() };
    return;
  }

  const positions = new Float32Array(displayPoints.length * 3);
  const colors = new Float32Array(displayPoints.length * 3);
  const objectPoints = [];
  const highlightPoints = [];
  const byDetectionId = new Map();
  const visibleDetections = detections.filter((detection) => isLabelVisible(detection.label));
  const selectedDetId = selectedInfo?.detection?.detection_id || null;
  const selectedVolume = selectedInfo?.volume || null;

  let zMin = Infinity, zMax = -Infinity;
  for (const point of displayPoints) {
    if (point[2] < zMin) zMin = point[2];
    if (point[2] > zMax) zMax = point[2];
  }
  const zSpan = Math.max(0.1, zMax - zMin);

  const jitter = sensorSamples.length ? 0.0 : 0.08;
  for (let i = 0; i < displayPoints.length; i++) {
    const px = displayPoints[i][0] + (Math.random() - 0.5) * jitter;
    const py = displayPoints[i][1] + (Math.random() - 0.5) * jitter;
    const pz = displayPoints[i][2] + (Math.random() - 0.5) * jitter;
    positions[i * 3] = px;
    positions[i * 3 + 1] = py;
    positions[i * 3 + 2] = pz;

    if (sensorSamples.length) {
      const [r, g, b] = sensorSampleBaseColor(frame, sensorSamples[i]);
      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    } else {
      const t = (displayPoints[i][2] - zMin) / zSpan;
      const r = 0.25 + t * 0.65;
      const g = 0.15 + t * 0.20;
      const b = 0.55 + (1 - t) * 0.25;
      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }
  }

  const ownershipSourcePositions = sensorSamples.length ? sensorSamples.map(samplePosition) : ownershipPoints;
  for (let i = 0; i < ownershipSourcePositions.length; i++) {
    const owner = assignPointOwner(ownershipSourcePositions[i], visibleDetections);
    if (owner) {
      const ownerColor = labelColor(owner.label);
      const ownedPoint = {
        position: [ownershipSourcePositions[i][0], ownershipSourcePositions[i][1], ownershipSourcePositions[i][2]],
        sourcePosition: ownershipSourcePositions[i],
        color: ownerColor,
        isSelected: owner.detection_id === selectedDetId,
      };
      objectPoints.push(ownedPoint);
      const existing = byDetectionId.get(owner.detection_id) || [];
      existing.push(ownedPoint);
      byDetectionId.set(owner.detection_id, existing);
    }

    if (selectedVolume && pointInDetectionVolume(ownershipSourcePositions[i], selectedVolume)) {
      highlightPoints.push([ownershipSourcePositions[i][0], ownershipSourcePositions[i][1], ownershipSourcePositions[i][2]]);
    }
  }

  state.pointOwnership = {
    byDetectionId,
  };

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: sensorSamples.length ? 0.2 : 0.3,
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    opacity: selectedVolume
      ? (sensorSamples.length ? 0.22 : 0.12)
      : (sensorSamples.length ? 0.48 : (objectPoints.length ? 0.38 : 0.85)),
    depthWrite: false,
    map: circleTexture,
    alphaMap: circleTexture,
    alphaTest: 0.1,
  });

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  if (objectPoints.length) {
    const ownedPositions = new Float32Array(objectPoints.length * 3);
    const ownedColors = new Float32Array(objectPoints.length * 3);
    const silhouetteColors = new Float32Array(objectPoints.length * 3);
    for (let i = 0; i < objectPoints.length; i++) {
      const owned = objectPoints[i];
      ownedPositions[i * 3] = owned.position[0];
      ownedPositions[i * 3 + 1] = owned.position[1];
      ownedPositions[i * 3 + 2] = owned.position[2];

      const intensity = selectedVolume ? (owned.isSelected ? 1.0 : 0.42) : 0.88;
      ownedColors[i * 3] = owned.color.r * intensity;
      ownedColors[i * 3 + 1] = owned.color.g * intensity;
      ownedColors[i * 3 + 2] = owned.color.b * intensity;

      const silhouetteIntensity = owned.isSelected ? 0.92 : 0.68;
      silhouetteColors[i * 3] = Math.min(1, owned.color.r * silhouetteIntensity + 0.08);
      silhouetteColors[i * 3 + 1] = Math.min(1, owned.color.g * silhouetteIntensity + 0.06);
      silhouetteColors[i * 3 + 2] = Math.min(1, owned.color.b * silhouetteIntensity + 0.10);
    }

    const silhouetteGeometry = new THREE.BufferGeometry();
    silhouetteGeometry.setAttribute("position", new THREE.Float32BufferAttribute(ownedPositions.slice(), 3));
    silhouetteGeometry.setAttribute("color", new THREE.Float32BufferAttribute(silhouetteColors, 3));
    const silhouetteMaterial = new THREE.PointsMaterial({
      size: sensorSamples.length ? (selectedVolume ? 0.82 : 0.64) : (selectedVolume ? 0.92 : 0.78),
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: selectedVolume ? (sensorSamples.length ? 0.3 : 0.42) : (sensorSamples.length ? 0.24 : 0.18),
      depthWrite: false,
      map: circleTexture,
      alphaMap: circleTexture,
      alphaTest: 0.05,
    });
    objectSilhouetteCloud = new THREE.Points(silhouetteGeometry, silhouetteMaterial);
    scene.add(objectSilhouetteCloud);

    const ownedGeometry = new THREE.BufferGeometry();
    ownedGeometry.setAttribute("position", new THREE.Float32BufferAttribute(ownedPositions, 3));
    ownedGeometry.setAttribute("color", new THREE.Float32BufferAttribute(ownedColors, 3));
    const ownedMaterial = new THREE.PointsMaterial({
      size: sensorSamples.length ? (selectedVolume ? 0.32 : 0.28) : (selectedVolume ? 0.44 : 0.4),
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: selectedVolume ? (sensorSamples.length ? 0.82 : 0.72) : (sensorSamples.length ? 0.96 : 0.9),
      depthWrite: false,
      map: circleTexture,
      alphaMap: circleTexture,
      alphaTest: 0.1,
    });
    objectPointCloud = new THREE.Points(ownedGeometry, ownedMaterial);
    scene.add(objectPointCloud);
  }

  if (highlightPoints.length) {
    const hiPositions = new Float32Array(highlightPoints.length * 3);
    const hiColors = new Float32Array(highlightPoints.length * 3);
    for (let i = 0; i < highlightPoints.length; i++) {
      hiPositions[i * 3] = highlightPoints[i][0];
      hiPositions[i * 3 + 1] = highlightPoints[i][1];
      hiPositions[i * 3 + 2] = highlightPoints[i][2];
      hiColors[i * 3] = 1.0;
      hiColors[i * 3 + 1] = 0.96;
      hiColors[i * 3 + 2] = 0.72;
    }

    const hiGeometry = new THREE.BufferGeometry();
    hiGeometry.setAttribute("position", new THREE.Float32BufferAttribute(hiPositions, 3));
    hiGeometry.setAttribute("color", new THREE.Float32BufferAttribute(hiColors, 3));
    const hiMaterial = new THREE.PointsMaterial({
      size: sensorSamples.length ? 0.34 : 0.42,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
      map: circleTexture,
      alphaMap: circleTexture,
      alphaTest: 0.1,
    });
    highlightPointCloud = new THREE.Points(hiGeometry, hiMaterial);
    scene.add(highlightPointCloud);
  }
}

// ─── Ego marker ──────────────────────────────────────────────────

function buildEgoMarker() {
  egoGroup.clear();

  const lidarGlowGeo = new THREE.RingGeometry(0.35, 0.55, 32);
  const lidarGlowMat = new THREE.MeshBasicMaterial({
    color: 0x3b8bff,
    transparent: true,
    opacity: 0.6,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const lidarGlow = new THREE.Mesh(lidarGlowGeo, lidarGlowMat);
  lidarGlow.rotation.x = Math.PI / 2;
  egoGroup.add(lidarGlow);

  const vehicleLength = 4.6;
  const vehicleWidth = 2.0;
  const halfL = vehicleLength / 2;
  const halfW = vehicleWidth / 2;
  const outlinePts = [
    new THREE.Vector3(-halfL, -halfW, 0.02),
    new THREE.Vector3(halfL, -halfW, 0.02),
    new THREE.Vector3(halfL, halfW, 0.02),
    new THREE.Vector3(-halfL, halfW, 0.02),
    new THREE.Vector3(-halfL, -halfW, 0.02),
  ];
  const outlineGeo = new THREE.BufferGeometry().setFromPoints(outlinePts);
  const outlineMat = new THREE.LineBasicMaterial({
    color: 0x6ea3ff,
    transparent: true,
    opacity: 0.55,
  });
  egoGroup.add(new THREE.Line(outlineGeo, outlineMat));

  const noseGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(halfL, -halfW, 0.02),
    new THREE.Vector3(halfL + 0.8, 0, 0.02),
    new THREE.Vector3(halfL, halfW, 0.02),
  ]);
  egoGroup.add(new THREE.Line(noseGeo, outlineMat));

  const arrowDir = new THREE.Vector3(1, 0, 0);
  const arrow = new THREE.ArrowHelper(
    arrowDir,
    new THREE.Vector3(0, 0, 0.1),
    6.5,
    0xff6a3d,
    0.9,
    0.45,
  );
  egoGroup.add(arrow);

  const leftWingGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0.05),
    new THREE.Vector3(0, 3.1, 0.05),
  ]);
  const leftWingMat = new THREE.LineBasicMaterial({
    color: 0x3dcc7a,
    transparent: true,
    opacity: 0.35,
  });
  egoGroup.add(new THREE.Line(leftWingGeo, leftWingMat));

  egoGroup.userData = {
    labels: [
      { text: "Forward (+X)", color: "#ff6a3d", pos: new THREE.Vector3(7.4, 0, 0.35) },
      { text: "Rear", color: "#7f8aad", pos: new THREE.Vector3(-4.6, 0, 0.25) },
      { text: "Left (+Y)", color: "#3dcc7a", pos: new THREE.Vector3(0, 4.2, 0.25) },
      { text: "Right (-Y)", color: "#7f8aad", pos: new THREE.Vector3(0, -4.2, 0.25) },
      { text: "LiDAR", color: "#6ea3ff", pos: new THREE.Vector3(0, 0, 0.9) },
    ],
  };
}

function applyFramePose(frame) {
  const pose = framePoseForCurrentReference(frame);
  if (!pose) {
    egoGroup.position.set(0, 0, 0);
    egoGroup.rotation.set(0, 0, 0);
    return;
  }
  const [x, y, z] = pose.position_m || [0, 0, 0];
  egoGroup.position.set(x, y, z);
  egoGroup.rotation.set(0, 0, pose.heading_rad || 0);
}

function applyOverlayVisibility() {
  if (pointCloud) {
    pointCloud.visible = state.overlays.rawPoints;
  }
  if (objectSilhouetteCloud) {
    objectSilhouetteCloud.visible = state.overlays.objectPoints;
  }
  if (objectPointCloud) {
    objectPointCloud.visible = state.overlays.objectPoints;
  }
  if (highlightPointCloud) {
    highlightPointCloud.visible = state.overlays.selectedPoints;
  }
  if (boxGroup) {
    boxGroup.visible = state.overlays.boxes;
  }
  if (trailGroup) {
    trailGroup.visible = state.overlays.trails;
  }
  if (motionGroup) {
    motionGroup.visible = state.overlays.motionVector;
  }
  if (eventGroup) {
    eventGroup.visible = state.overlays.events;
  }
  if (worldAxesGroup) {
    worldAxesGroup.visible = state.overlays.worldAxes;
  }
  if (egoGroup) {
    egoGroup.visible = state.overlays.egoMarker;
  }
}

// ─── 3D bounding boxes ──────────────────────────────────────────

function buildBoxes(detections, activeTracks) {
  boxGroup.clear();
  labelContainer.innerHTML = "";
  clickableBoxes = [];

  if (!detections || !detections.length) return;

  // Build a map of track info by matching detection centroids to tracks
  const trackByDetId = new Map();
  if (activeTracks) {
    for (const track of activeTracks) {
      trackByDetId.set(track.track_id, track);
    }
  }

  const selectedInfo = resolveSelectedObject(state.bundle?.frame_summaries?.[state.frameIndex]);

  for (const det of detections) {
    if (!isLabelVisible(det.label)) continue;
    const color = labelColor(det.label);
    const threeColor = new THREE.Color(color.r, color.g, color.b);
    const isSelected = selectedInfo?.detection?.detection_id === det.detection_id;
    const isDimmed = Boolean(selectedInfo) && !isSelected;

    const min = det.bbox_min;
    const max = det.bbox_max;
    const cx = (min[0] + max[0]) / 2;
    const cy = (min[1] + max[1]) / 2;
    const cz = (min[2] + max[2]) / 2;
    const sx = Math.abs(max[0] - min[0]);
    const sy = Math.abs(max[1] - min[1]);
    const sz = Math.abs(max[2] - min[2]);

    // Wireframe box
    const boxGeo = new THREE.BoxGeometry(sx, sy, sz);
    const edges = new THREE.EdgesGeometry(boxGeo);
    const lineMat = new THREE.LineBasicMaterial({
      color: isSelected ? 0xffefbf : threeColor,
      linewidth: isSelected ? 3 : 2,
      transparent: true,
      opacity: isSelected ? 0.82 : (isDimmed ? 0.025 : 0.16),
    });
    const wireframe = new THREE.LineSegments(edges, lineMat);
    wireframe.position.set(cx, cy, cz);
    if (det.heading_rad != null) {
      wireframe.rotation.z = det.heading_rad;
    }
    boxGroup.add(wireframe);

    if (isSelected) {
      const glowMat = new THREE.LineBasicMaterial({
        color: 0xffc95a,
        transparent: true,
        opacity: 0.18,
      });
      const glow = new THREE.LineSegments(edges, glowMat);
      glow.position.set(cx, cy, cz);
      glow.scale.set(1.05, 1.05, 1.05);
      if (det.heading_rad != null) {
        glow.rotation.z = det.heading_rad;
      }
      boxGroup.add(glow);
    }

    // Transparent clickable box for raycasting
    const clickGeo = new THREE.BoxGeometry(sx, sy, sz);
    const clickMat = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const clickMesh = new THREE.Mesh(clickGeo, clickMat);
    clickMesh.position.set(cx, cy, cz);
    if (det.heading_rad != null) {
      clickMesh.rotation.z = det.heading_rad;
    }
    clickMesh.userData = { detection: det, tracks: activeTracks };
    boxGroup.add(clickMesh);
    clickableBoxes.push(clickMesh);

    // Semi-transparent fill on bottom face
    const fillGeo = new THREE.PlaneGeometry(sx, sy);
    const fillMat = new THREE.MeshBasicMaterial({
      color: isSelected ? 0xffc95a : threeColor,
      transparent: true,
      opacity: isSelected ? 0.06 : 0.0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const fill = new THREE.Mesh(fillGeo, fillMat);
    fill.position.set(cx, cy, min[2] + 0.01);
    if (det.heading_rad != null) {
      fill.rotation.z = det.heading_rad;
    }
    boxGroup.add(fill);

    // Vertical line from ground to box bottom
    const poleGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(cx, cy, -2.5),
      new THREE.Vector3(cx, cy, min[2]),
    ]);
    const poleMat = new THREE.LineBasicMaterial({
      color: isSelected ? 0xffc95a : threeColor,
      transparent: true,
      opacity: isSelected ? 0.16 : 0.0,
    });
    if (isSelected) {
      boxGroup.add(new THREE.Line(poleGeo, poleMat));
    }

    // Store data for 3D label projection
    wireframe.userData = {
      label: det.label || "object",
      score: det.score,
      detId: det.detection_id,
      color: color.hex,
      isSelected,
      topPos: new THREE.Vector3(cx, cy, max[2] + 0.5),
    };
  }
}

// ─── Track trails ───────────────────────────────────────────────

function buildTrails(bundle, frameIndex) {
  trailGroup.clear();
  if (!bundle) return;

  const selectedInfo = resolveSelectedObject(bundle.frame_summaries[frameIndex]);
  const selectedTrackId = selectedInfo?.track?.track_id ?? null;
  const history = new Map();
  const labelMap = new Map();
  const start = Math.max(0, frameIndex - 120);
  for (let i = start; i <= frameIndex; i++) {
    const frame = bundle.frame_summaries[i];
    for (const track of activeTracksForFrame(frame)) {
      if (!history.has(track.track_id)) history.set(track.track_id, []);
      history.get(track.track_id).push(track.centroid);
      if (track.label) labelMap.set(track.track_id, track.label);
    }
  }

  for (const [trackId, trail] of history) {
    if (trail.length < 2) continue;
    const isSelected = selectedTrackId === trackId;
    const visibleTrail = isSelected ? trail : trail.slice(-30);
    const points = visibleTrail.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    const trackLabel = labelMap.get(trackId) || "object";
    if (!isLabelVisible(trackLabel)) continue;
    const color = labelColor(trackLabel);
    const isDimmed = Boolean(selectedInfo) && !isSelected;
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({
      color: isSelected ? new THREE.Color(1.0, 0.79, 0.35) : new THREE.Color(color.r, color.g, color.b),
      transparent: true,
      opacity: isSelected ? 0.95 : (isDimmed ? 0.08 : 0.4),
      linewidth: isSelected ? 2 : 1,
    });
    trailGroup.add(new THREE.Line(geo, mat));
  }
}

function buildMotionCue(frame, selectedInfo) {
  motionGroup.clear();
  if (!frame || !selectedInfo?.track) return;

  const track = selectedInfo.track;
  const velocity = track.velocity_mps || [0, 0, 0];
  const speed = Math.hypot(velocity[0], velocity[1], velocity[2]);
  if (!Number.isFinite(speed) || speed < 0.05 || speed > 100) return;

  const origin = new THREE.Vector3(
    track.centroid[0],
    track.centroid[1],
    track.centroid[2] + 0.35,
  );
  const direction = new THREE.Vector3(velocity[0], velocity[1], velocity[2]).normalize();
  const length = Math.min(10, Math.max(1.5, speed * 1.8));
  const arrow = new THREE.ArrowHelper(
    direction,
    origin,
    length,
    0xffc95a,
    Math.min(1.6, length * 0.22),
    Math.min(0.9, length * 0.12),
  );
  motionGroup.add(arrow);

  const lineGeo = new THREE.BufferGeometry().setFromPoints([
    origin,
    origin.clone().add(direction.clone().multiplyScalar(length)),
  ]);
  const line = new THREE.Line(
    lineGeo,
    new THREE.LineBasicMaterial({
      color: 0xffe08a,
      transparent: true,
      opacity: 0.85,
    }),
  );
  motionGroup.add(line);

  motionGroup.userData = {
    labels: [
      {
        text: `${speed.toFixed(1)} m/s`,
        color: "#ffdb7a",
        pos: origin.clone().add(direction.clone().multiplyScalar(length + 0.7)),
      },
    ],
  };
}

function zoneCenter(zone, zoneDefinitions) {
  const definition = zoneDefinitions.get(zone.zone_id);
  if (!definition) return [0, 0, 0.35];
  const minXY = definition.min_xy || [0, 0];
  const maxXY = definition.max_xy || [0, 0];
  return [
    (minXY[0] + maxXY[0]) / 2,
    (minXY[1] + maxXY[1]) / 2,
    0.35,
  ];
}

function buildEvents(frame, bundle) {
  eventGroup.clear();
  eventGroup.userData = { labels: [] };
  if (!frame?.scene_state?.events?.length) return;

  const zoneDefinitions = new Map((bundle.zone_definitions || []).map((zone) => [zone.zone_id, zone]));
  for (const event of frame.scene_state.events) {
    const track = activeTracksForFrame(frame).find((item) => item.track_id === event.track_id);
    const zone = frame.scene_state.zones?.find((item) => item.zone_id === event.zone_id);
    const center = track
      ? [track.centroid[0], track.centroid[1], Math.max(0.35, track.centroid[2] + 0.5)]
      : zone
        ? zoneCenter(zone, zoneDefinitions)
        : [0, 0, 0.35];

    const color = Number.parseInt(eventTypeColor(event.event_type).slice(1), 16);
    const beaconGeo = new THREE.SphereGeometry(0.2, 12, 12);
    const beaconMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.95,
    });
    const beacon = new THREE.Mesh(beaconGeo, beaconMat);
    beacon.position.set(center[0], center[1], center[2]);
    eventGroup.add(beacon);

    const poleGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(center[0], center[1], center[2] - 0.55),
      new THREE.Vector3(center[0], center[1], center[2] + 0.55),
    ]);
    const poleMat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.55,
    });
    eventGroup.add(new THREE.Line(poleGeo, poleMat));

    eventGroup.userData.labels.push({
      text: `${event.zone_name}: ${formatEventType(event.event_type)}${event.direction ? ` · ${formatDirection(event.direction)}` : ""}`,
      color: `#${color.toString(16).padStart(6, "0")}`,
      pos: new THREE.Vector3(center[0], center[1], center[2] + 0.85),
    });
  }
}

function buildRuleOverlays() {
  ruleGroup.clear();
  ruleGroup.userData = { labels: [] };
  if (!state.bundle || isScanlineMode()) return;

  const rules = currentRuleState();
  for (const zone of rules.zones) {
    const minXY = zone.min_xy || [0, 0];
    const maxXY = zone.max_xy || [0, 0];
    const color = zone.color || "#6f8f77";
    const corners = [
      new THREE.Vector3(minXY[0], minXY[1], 0.06),
      new THREE.Vector3(maxXY[0], minXY[1], 0.06),
      new THREE.Vector3(maxXY[0], maxXY[1], 0.06),
      new THREE.Vector3(minXY[0], maxXY[1], 0.06),
      new THREE.Vector3(minXY[0], minXY[1], 0.06),
    ];
    const geometry = new THREE.BufferGeometry().setFromPoints(corners);
    const line = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }),
    );
    ruleGroup.add(line);

    const fillGeometry = new THREE.PlaneGeometry(
      Math.max(0.1, Math.abs(maxXY[0] - minXY[0])),
      Math.max(0.1, Math.abs(maxXY[1] - minXY[1])),
    );
    const fill = new THREE.Mesh(
      fillGeometry,
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08, side: THREE.DoubleSide }),
    );
    fill.position.set((minXY[0] + maxXY[0]) / 2, (minXY[1] + maxXY[1]) / 2, 0.04);
    ruleGroup.add(fill);

    ruleGroup.userData.labels.push({
      text: zone.name,
      color,
      pos: new THREE.Vector3((minXY[0] + maxXY[0]) / 2, (minXY[1] + maxXY[1]) / 2, 0.25),
    });
  }

  for (const tripwire of rules.tripwires) {
    const start = tripwire.start_xy || [0, 0];
    const end = tripwire.end_xy || [0, 0];
    const color = tripwire.color || "#59b8ff";
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(start[0], start[1], 0.08),
        new THREE.Vector3(end[0], end[1], 0.08),
      ]),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.92 }),
    );
    ruleGroup.add(line);

    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const length = Math.max(0.001, Math.hypot(dx, dy));
    const ux = dx / length;
    const uy = dy / length;
    const arrowBase = new THREE.Vector3(end[0] - ux * 0.9, end[1] - uy * 0.9, 0.08);
    const perpX = -uy;
    const perpY = ux;
    const arrow = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(end[0], end[1], 0.08),
        new THREE.Vector3(arrowBase.x + perpX * 0.25, arrowBase.y + perpY * 0.25, 0.08),
        new THREE.Vector3(arrowBase.x - perpX * 0.25, arrowBase.y - perpY * 0.25, 0.08),
        new THREE.Vector3(end[0], end[1], 0.08),
      ]),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.92 }),
    );
    ruleGroup.add(arrow);

    ruleGroup.userData.labels.push({
      text: tripwire.name,
      color,
      pos: new THREE.Vector3((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, 0.25),
    });
  }

  if (state.authoring.mode && state.authoring.points.length) {
    const start = state.authoring.points[0];
    const end = state.authoring.hoverPoint || start;
    const previewColor = state.authoring.mode === "zone" ? "#7ee787" : "#59b8ff";
    if (state.authoring.mode === "zone") {
      const corners = [
        new THREE.Vector3(start[0], start[1], 0.09),
        new THREE.Vector3(end[0], start[1], 0.09),
        new THREE.Vector3(end[0], end[1], 0.09),
        new THREE.Vector3(start[0], end[1], 0.09),
        new THREE.Vector3(start[0], start[1], 0.09),
      ];
      ruleGroup.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(corners),
          new THREE.LineDashedMaterial({ color: previewColor, dashSize: 0.4, gapSize: 0.2, transparent: true, opacity: 0.9 }),
        ),
      );
      ruleGroup.children.at(-1)?.computeLineDistances?.();
    } else {
      ruleGroup.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(start[0], start[1], 0.09),
            new THREE.Vector3(end[0], end[1], 0.09),
          ]),
          new THREE.LineDashedMaterial({ color: previewColor, dashSize: 0.4, gapSize: 0.2, transparent: true, opacity: 0.9 }),
        ),
      );
      ruleGroup.children.at(-1)?.computeLineDistances?.();
    }
  }
}

// ─── Project 3D labels to screen ─────────────────────────────────

function updateLabels() {
  labelContainer.innerHTML = "";
  if (!state.bundle) return;
  if (isScanlineMode()) return;

  const halfW = window.innerWidth / 2;
  const halfH = window.innerHeight / 2;

  for (const child of boxGroup.children) {
    if (!child.userData || !child.userData.topPos) continue;
    const data = child.userData;
    if (state.selected && !data.isSelected) continue;

    const projected = data.topPos.clone().project(camera);
    if (projected.z > 1) continue;

    const x = projected.x * halfW + halfW;
    const y = -projected.y * halfH + halfH;

    if (x < -100 || x > window.innerWidth + 100 || y < -50 || y > window.innerHeight + 50) continue;

    const el = document.createElement("div");
    el.className = "label-3d";
    el.style.left = x + "px";
    el.style.top = y + "px";
    el.style.color = data.color;
    el.style.background = "rgba(12, 14, 19, 0.75)";
    el.style.borderLeft = `3px solid ${data.color}`;

    const scoreText = data.score != null ? ` ${Math.round(data.score * 100)}%` : "";
    el.textContent = data.label + scoreText;
    labelContainer.appendChild(el);
  }

  for (const item of egoGroup.userData.labels || []) {
    if (!state.overlays.egoMarker) continue;
    const projected = egoGroup.localToWorld(item.pos.clone()).project(camera);
    if (projected.z > 1) continue;

    const x = projected.x * halfW + halfW;
    const y = -projected.y * halfH + halfH;
    if (x < -140 || x > window.innerWidth + 140 || y < -60 || y > window.innerHeight + 60) continue;

    const el = document.createElement("div");
    el.className = "label-3d label-ego";
    el.style.left = x + "px";
    el.style.top = y + "px";
    el.style.color = item.color;
    el.textContent = item.text;
    labelContainer.appendChild(el);
  }

  if (state.overlays.motionVector) {
    for (const item of motionGroup.userData?.labels || []) {
      const projected = item.pos.clone().project(camera);
      if (projected.z > 1) continue;

      const x = projected.x * halfW + halfW;
      const y = -projected.y * halfH + halfH;
      if (x < -140 || x > window.innerWidth + 140 || y < -60 || y > window.innerHeight + 60) continue;

      const el = document.createElement("div");
      el.className = "label-3d label-ego";
      el.style.left = x + "px";
      el.style.top = y + "px";
      el.style.color = item.color;
      el.textContent = item.text;
      labelContainer.appendChild(el);
    }
  }

  if (state.overlays.events) {
    for (const item of eventGroup.userData?.labels || []) {
      const projected = item.pos.clone().project(camera);
      if (projected.z > 1) continue;

      const x = projected.x * halfW + halfW;
      const y = -projected.y * halfH + halfH;
      if (x < -180 || x > window.innerWidth + 180 || y < -60 || y > window.innerHeight + 60) continue;

      const el = document.createElement("div");
      el.className = "label-3d label-ego";
      el.style.left = x + "px";
      el.style.top = y + "px";
      el.style.color = item.color;
      el.textContent = item.text;
      labelContainer.appendChild(el);
    }
  }

  for (const item of ruleGroup.userData?.labels || []) {
    const projected = item.pos.clone().project(camera);
    if (projected.z > 1) continue;

    const x = projected.x * halfW + halfW;
    const y = -projected.y * halfH + halfH;
    if (x < -180 || x > window.innerWidth + 180 || y < -60 || y > window.innerHeight + 60) continue;

    const el = document.createElement("div");
    el.className = "label-3d label-ego";
    el.style.left = x + "px";
    el.style.top = y + "px";
    el.style.color = item.color;
    el.textContent = item.text;
    labelContainer.appendChild(el);
  }
}

// ─── Click-to-inspect panel ──────────────────────────────────────

function showInspectPanel(det, tracks) {
  const panel = document.getElementById("inspect-panel");
  const content = document.getElementById("inspect-content");

  const color = labelColor(det.label);
  const min = det.bbox_min;
  const max = det.bbox_max;
  const dims = {
    length: Math.abs(max[0] - min[0]).toFixed(2),
    width: Math.abs(max[1] - min[1]).toFixed(2),
    height: Math.abs(max[2] - min[2]).toFixed(2),
  };

  // Find matching track
  let matchedTrack = null;
  if (tracks) {
    const detCx = (min[0] + max[0]) / 2;
    const detCy = (min[1] + max[1]) / 2;
    let bestDist = Infinity;
    for (const t of tracks) {
      const dx = t.centroid[0] - detCx;
      const dy = t.centroid[1] - detCy;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if (dist < bestDist && dist < 3.0) {
        bestDist = dist;
        matchedTrack = t;
      }
    }
  }

  let velocity = "—";
  let speed = "—";
  let lifetime = "—";
  let hits = "—";
  let distance = "—";

  if (matchedTrack) {
    const v = matchedTrack.velocity_mps;
    const rawSpeed = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    // Cap display at 100 m/s — higher values indicate tracker noise
    if (rawSpeed < 100) {
      speed = rawSpeed.toFixed(1) + " m/s";
      velocity = `[${v[0].toFixed(1)}, ${v[1].toFixed(1)}, ${v[2].toFixed(1)}]`;
    } else {
      speed = "noisy";
      velocity = "noisy";
    }

    const durationMs = (matchedTrack.last_seen_ns - matchedTrack.first_seen_ns) / 1e6;
    if (durationMs < 0) {
      lifetime = "—";
    } else if (durationMs > 1000) {
      lifetime = (durationMs / 1000).toFixed(1) + " s";
    } else {
      lifetime = Math.round(durationMs) + " ms";
    }
    hits = `${matchedTrack.hits} / ${matchedTrack.age_frames} frames`;
    distance = matchedTrack.total_distance_m.toFixed(2) + " m";
  }

  const position = det.centroid
    ? `[${det.centroid[0].toFixed(1)}, ${det.centroid[1].toFixed(1)}, ${det.centroid[2].toFixed(1)}]`
    : "—";

  const rows = [
    ["Class", `<span style="color:${color.hex};font-weight:700">${det.label || "object"}</span>`],
    ["Confidence", det.score != null ? `${Math.round(det.score * 100)}%` : "—"],
    ["Position", position],
    ["Dimensions", `${dims.length} × ${dims.width} × ${dims.height} m`],
    ["Speed", speed],
    ["Velocity", velocity],
    ["Lifetime", lifetime],
    ["Hits", hits],
    ["Distance", distance],
  ];

  const hero = `
    <div class="inspect-hero">
      <div class="inspect-pill">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color.hex}"></span>
        ${formatLabel(det.label)}
      </div>
      <div class="inspect-title">
        <div class="inspect-name">${formatLabel(det.label)}</div>
        <div class="inspect-score">${det.score != null ? `${Math.round(det.score * 100)}%` : "Tracked"}</div>
      </div>
    </div>
    <div class="inspect-scanline-block">
      <canvas id="inspect-scanline" width="252" height="120"></canvas>
    </div>
  `;

  content.innerHTML = hero + rows.map(([k, v]) =>
    `<div class="row"><dt>${k}</dt><dd>${v}</dd></div>`
  ).join("");

  panel.style.display = "block";
  const currentFrame = state.bundle?.frame_summaries?.[state.frameIndex];
  const structuredSamples = currentFrame ? ownedScanlineSamplesForDetection(currentFrame, det.detection_id) : [];
  if (structuredSamples.length && currentFrame?.scanline_shape) {
    renderInspectStructuredScanline(currentFrame, structuredSamples, det.label);
  } else {
    renderInspectScanline(ownedPointsForDetection(det.detection_id), det.label);
  }
}

function hideInspectPanel() {
  document.getElementById("inspect-panel").style.display = "none";
  state.selected = null;
  if (state.bundle) {
    showFrame();
  }
}

function pointInVolume(point, volume) {
  return (
    point[0] >= volume.bbox_min[0] &&
    point[0] <= volume.bbox_max[0] &&
    point[1] >= volume.bbox_min[1] &&
    point[1] <= volume.bbox_max[1] &&
    point[2] >= volume.bbox_min[2] &&
    point[2] <= volume.bbox_max[2]
  );
}

function volumeDimensions(volume) {
  return [
    Math.abs(volume.bbox_max[0] - volume.bbox_min[0]),
    Math.abs(volume.bbox_max[1] - volume.bbox_min[1]),
    Math.abs(volume.bbox_max[2] - volume.bbox_min[2]),
  ];
}

function pointInDetectionVolume(point, volume) {
  if (!volume?.centroid || volume.heading_rad == null) {
    return pointInVolume(point, volume);
  }

  const [dx, dy, dz] = [
    point[0] - volume.centroid[0],
    point[1] - volume.centroid[1],
    point[2] - volume.centroid[2],
  ];
  const cos = Math.cos(-volume.heading_rad);
  const sin = Math.sin(-volume.heading_rad);
  const localX = dx * cos - dy * sin;
  const localY = dx * sin + dy * cos;
  const [sx, sy, sz] = volumeDimensions(volume);
  return (
    Math.abs(localX) <= sx / 2 &&
    Math.abs(localY) <= sy / 2 &&
    Math.abs(dz) <= sz / 2
  );
}

function ownedPointsForDetection(detectionId) {
  return state.pointOwnership.byDetectionId.get(detectionId) || [];
}

function inverseRotateXY(point, headingRad) {
  const cos = Math.cos(-headingRad);
  const sin = Math.sin(-headingRad);
  return [
    point[0] * cos - point[1] * sin,
    point[0] * sin + point[1] * cos,
    point[2],
  ];
}

function sensorSpaceDetection(frame, detection) {
  const pose = frame?.frame_pose;
  if (!pose) return detection;
  const [tx, ty, tz] = pose.position_m || [0, 0, 0];
  const localCentroid = inverseRotateXY(
    [
      detection.centroid[0] - tx,
      detection.centroid[1] - ty,
      detection.centroid[2] - tz,
    ],
    pose.heading_rad || 0,
  );
  const sizeX = detection.bbox_max[0] - detection.bbox_min[0];
  const sizeY = detection.bbox_max[1] - detection.bbox_min[1];
  const sizeZ = detection.bbox_max[2] - detection.bbox_min[2];
  return {
    ...detection,
    centroid: localCentroid,
    bbox_min: [
      localCentroid[0] - sizeX / 2,
      localCentroid[1] - sizeY / 2,
      localCentroid[2] - sizeZ / 2,
    ],
    bbox_max: [
      localCentroid[0] + sizeX / 2,
      localCentroid[1] + sizeY / 2,
      localCentroid[2] + sizeZ / 2,
    ],
    heading_rad: detection.heading_rad == null
      ? null
      : detection.heading_rad - (pose.heading_rad || 0),
  };
}

function detectionDimensions(detection) {
  return {
    sx: detection.bbox_max[0] - detection.bbox_min[0],
    sy: detection.bbox_max[1] - detection.bbox_min[1],
    sz: detection.bbox_max[2] - detection.bbox_min[2],
  };
}

function detectionBoundingCorners(detection) {
  const { sx, sy, sz } = detectionDimensions(detection);
  const heading = detection.heading_rad || 0;
  const cos = Math.cos(heading);
  const sin = Math.sin(heading);
  const halfX = sx / 2;
  const halfY = sy / 2;
  const halfZ = sz / 2;
  const corners = [];
  for (const dx of [-halfX, halfX]) {
    for (const dy of [-halfY, halfY]) {
      for (const dz of [-halfZ, halfZ]) {
        corners.push([
          detection.centroid[0] + dx * cos - dy * sin,
          detection.centroid[1] + dx * sin + dy * cos,
          detection.centroid[2] + dz,
        ]);
      }
    }
  }
  return corners;
}

function azimuthToDestaggeredColumn(angle, totalCols) {
  const normalized = mod((angle + Math.PI) / (Math.PI * 2), 1);
  return mod(Math.round(normalized * Math.max(1, totalCols - 1)), totalCols);
}

function rowElevations(frame) {
  const shape = scanlineShapeForFrame(frame);
  if (!shape) return [];
  const rows = shape[0];
  const sums = new Array(rows).fill(0);
  const counts = new Array(rows).fill(0);
  for (const sample of scanlinePointsForFrame(frame)) {
    const elevation = Math.atan2(sample[4], Math.hypot(sample[2], sample[3]));
    sums[sample[0]] += elevation;
    counts[sample[0]] += 1;
  }
  const result = new Array(rows).fill(0);
  let lastKnown = 0;
  for (let index = 0; index < rows; index++) {
    if (counts[index] > 0) {
      lastKnown = sums[index] / counts[index];
    }
    result[index] = lastKnown;
  }
  for (let index = rows - 1; index >= 0; index--) {
    if (counts[index] > 0) {
      lastKnown = sums[index] / counts[index];
    }
    if (counts[index] === 0) {
      result[index] = lastKnown;
    }
  }
  return result;
}

function nearestRowForElevation(elevations, target) {
  if (!elevations.length) return 0;
  let bestIndex = 0;
  let bestDistance = Infinity;
  for (let index = 0; index < elevations.length; index++) {
    const distance = Math.abs(elevations[index] - target);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function scanlineDetectionBounds(frame, detection, ownedSamples) {
  const shape = scanlineShapeForFrame(frame);
  if (!shape) return null;
  const sensorDetection = sensorSpaceDetection(frame, detection);
  const corners = detectionBoundingCorners(sensorDetection);
  const cols = corners.map((corner) =>
    azimuthToDestaggeredColumn(Math.atan2(corner[1], corner[0]), shape[1])
  );
  const segments = circularColumnSegments(cols, shape[1]);
  let rowMin = 0;
  let rowMax = shape[0] - 1;
  if (ownedSamples.length) {
    rowMin = Math.min(...ownedSamples.map((sample) => sample[0]));
    rowMax = Math.max(...ownedSamples.map((sample) => sample[0]));
  } else {
    const elevations = rowElevations(frame);
    const cornerElevations = corners.map((corner) =>
      Math.atan2(corner[2], Math.hypot(corner[0], corner[1]))
    );
    rowMin = nearestRowForElevation(elevations, Math.min(...cornerElevations));
    rowMax = nearestRowForElevation(elevations, Math.max(...cornerElevations));
    if (rowMin > rowMax) {
      [rowMin, rowMax] = [rowMax, rowMin];
    }
  }
  return { rowMin, rowMax, segments };
}

function assignScanlineOwner(sample, detections) {
  if (!detections?.length) return null;
  const point = [sample[2], sample[3], sample[4]];
  let best = null;
  let bestDist = Infinity;
  for (const detection of detections) {
    if (!pointInDetectionVolume(point, detection)) continue;
    const dx = point[0] - detection.centroid[0];
    const dy = point[1] - detection.centroid[1];
    const dz = point[2] - detection.centroid[2];
    const dist = dx * dx + dy * dy + dz * dz;
    if (dist < bestDist) {
      best = detection;
      bestDist = dist;
    }
  }
  return best;
}

function ownedScanlineSamplesForDetection(frame, detectionId) {
  const samples = scanlinePointsForFrame(frame);
  if (!samples.length) return [];
  const frameDetections = detectionsForFrame(frame);
  const detection = frameDetections.find((item) => item.detection_id === detectionId);
  if (!detection) return [];
  const visibleDetections = frameDetections
    .filter((item) => isLabelVisible(item.label))
    .map((item) => currentReferenceFrame() === "world" ? sensorSpaceDetection(frame, item) : item);
  return samples.filter((sample) => {
    const owner = assignScanlineOwner(sample, visibleDetections);
    return owner?.detection_id === detectionId;
  });
}

function renderInspectScanline(points, label) {
  const canvas = document.getElementById("inspect-scanline");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(8, 10, 14, 0.96)";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (height / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  if (!points.length) {
    ctx.fillStyle = "rgba(212, 216, 228, 0.56)";
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText("No owned points in preview", 14, height / 2);
    return;
  }

  const color = labelColor(label).hex;
  const azimuths = points.map((point) => Math.atan2(point.sourcePosition[1], point.sourcePosition[0]));
  const elevations = points.map((point) => Math.atan2(point.sourcePosition[2], Math.hypot(point.sourcePosition[0], point.sourcePosition[1])));
  const azMin = Math.min(...azimuths);
  const azMax = Math.max(...azimuths);
  const elMin = Math.min(...elevations);
  const elMax = Math.max(...elevations);
  const azSpan = Math.max(0.01, azMax - azMin);
  const elSpan = Math.max(0.01, elMax - elMin);

  ctx.fillStyle = color;
  for (let i = 0; i < points.length; i++) {
    const x = ((azimuths[i] - azMin) / azSpan) * (width - 18) + 9;
    const y = height - (((elevations[i] - elMin) / elSpan) * (height - 18) + 9);
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.arc(x, y, 1.8, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  ctx.fillStyle = "rgba(212, 216, 228, 0.72)";
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText("Scanline view", 12, 18);
  ctx.fillText(`${points.length} pts`, width - 58, 18);
}

function renderInspectStructuredScanline(frame, samples, label) {
  const canvas = document.getElementById("inspect-scanline");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(8, 10, 14, 0.96)";
  ctx.fillRect(0, 0, width, height);

  const shape = scanlineShapeForFrame(frame);
  if (!samples.length || !shape) {
    renderInspectScanline([], label);
    return;
  }

  const color = labelColor(label).hex;
  ctx.fillStyle = color;
  for (const sample of samples) {
    const cell = scanlineCellGeometry(frame, sample, width, height, 9);
    ctx.globalAlpha = 0.82;
    ctx.fillRect(
      cell.x,
      cell.y,
      Math.max(1, cell.cellWidth),
      Math.max(1, cell.cellHeight),
    );
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = "rgba(212, 216, 228, 0.72)";
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText("Sensor scanline", 12, 18);
  ctx.fillText(`${samples.length} pts`, width - 58, 18);
}

function pointToAngles(point) {
  return {
    azimuth: Math.atan2(point[1], point[0]),
    elevation: Math.atan2(point[2], Math.hypot(point[0], point[1])),
  };
}

function rangeFrameExtents(points) {
  if (!points.length) {
    return { elevationMin: -0.35, elevationMax: 0.25 };
  }
  const elevations = points.map((point) => pointToAngles(point).elevation);
  return {
    elevationMin: Math.min(...elevations),
    elevationMax: Math.max(...elevations),
  };
}

function rangePointToCanvas(point, width, height, extents) {
  const { azimuth, elevation } = pointToAngles(point);
  const elevationSpan = Math.max(0.01, extents.elevationMax - extents.elevationMin);
  return {
    x: ((azimuth + Math.PI) / (Math.PI * 2)) * width,
    y: height - (((elevation - extents.elevationMin) / elevationSpan) * height),
  };
}

function drawRangeGrid(width, height) {
  rangeCtx.strokeStyle = "rgba(255,255,255,0.08)";
  rangeCtx.lineWidth = 1;
  const horizontal = 6;
  const vertical = 8;
  for (let i = 1; i < horizontal; i++) {
    const y = (height / horizontal) * i;
    rangeCtx.beginPath();
    rangeCtx.moveTo(0, y);
    rangeCtx.lineTo(width, y);
    rangeCtx.stroke();
  }
  for (let i = 1; i < vertical; i++) {
    const x = (width / vertical) * i;
    rangeCtx.beginPath();
    rangeCtx.moveTo(x, 0);
    rangeCtx.lineTo(x, height);
    rangeCtx.stroke();
  }
}

function drawRangePointSet(points, color, radius, alpha, width, height, extents) {
  if (!points.length) return;
  rangeCtx.fillStyle = color;
  rangeCtx.globalAlpha = alpha;
  for (const point of points) {
    const canvasPoint = rangePointToCanvas(point, width, height, extents);
    rangeCtx.beginPath();
    rangeCtx.arc(canvasPoint.x, canvasPoint.y, radius, 0, Math.PI * 2);
    rangeCtx.fill();
  }
  rangeCtx.globalAlpha = 1;
}

function drawRangeDetections(frame, width, height, extents, selectedInfo) {
  const selectedDetId = selectedInfo?.detection?.detection_id || null;
  const scanlineSamples = scanlinePointsForFrame(frame);
  const scanShape = scanlineShapeForFrame(frame);
  for (const detection of detectionsForFrame(frame)) {
    if (!isLabelVisible(detection.label)) continue;
    const isSelected = detection.detection_id === selectedDetId;
    const color = isSelected ? "#fff0c2" : labelColor(detection.label).hex;
    if (scanlineSamples.length && scanShape) {
      const ownedSamples = ownedScanlineSamplesForDetection(frame, detection.detection_id);
      const bounds = scanlineDetectionBounds(frame, detection, ownedSamples);
      if (!bounds) continue;
      const { rowMin, rowMax, segments } = bounds;
      const rowHeight = height / Math.max(1, scanShape[0]);
      const colWidth = width / Math.max(1, scanShape[1]);
      rangeCtx.strokeStyle = color;
      rangeCtx.lineWidth = isSelected ? 2.2 : 1.2;
      rangeCtx.fillStyle = color;
      rangeCtx.font = isSelected ? "700 12px Inter, sans-serif" : "600 11px Inter, sans-serif";
      const scoreText = detection.score != null ? ` ${Math.round(detection.score * 100)}%` : "";
      let labelDrawn = false;
      for (const [startCol, endCol] of segments) {
        const minX = Math.max(4, startCol * colWidth - 4);
        const maxX = Math.min(width - 4, (endCol + 1) * colWidth + 4);
        const minY = Math.max(4, rowMin * rowHeight - 4);
        const maxY = Math.min(height - 4, (rowMax + 1) * rowHeight + 4);
        rangeCtx.globalAlpha = isSelected ? 0.95 : 0.55;
        rangeCtx.strokeRect(minX, minY, Math.max(6, maxX - minX), Math.max(6, maxY - minY));
        rangeCtx.globalAlpha = isSelected ? 0.16 : 0.06;
        rangeCtx.fillRect(minX, minY, Math.max(6, maxX - minX), Math.max(6, maxY - minY));
        rangeCtx.globalAlpha = 1;
        if (!labelDrawn) {
          rangeCtx.fillText(`${formatLabel(detection.label)}${scoreText}`, minX, Math.max(14, minY - 6));
          labelDrawn = true;
        }
      }
      continue;
    }

    const canvasPoints = ownedPointsForDetection(detection.detection_id).map((point) =>
      rangePointToCanvas(point.sourcePosition, width, height, extents)
    );
    if (!canvasPoints.length) continue;
    const xs = canvasPoints.map((point) => point.x);
    const ys = canvasPoints.map((point) => point.y);
    const minX = Math.max(8, Math.min(...xs) - 8);
    const maxX = Math.min(width - 8, Math.max(...xs) + 8);
    const minY = Math.max(8, Math.min(...ys) - 8);
    const maxY = Math.min(height - 8, Math.max(...ys) + 8);
    rangeCtx.strokeStyle = color;
    rangeCtx.lineWidth = isSelected ? 2.2 : 1.2;
    rangeCtx.globalAlpha = isSelected ? 0.95 : 0.55;
    rangeCtx.strokeRect(minX, minY, Math.max(6, maxX - minX), Math.max(6, maxY - minY));
    rangeCtx.globalAlpha = isSelected ? 0.16 : 0.06;
    rangeCtx.fillStyle = color;
    rangeCtx.fillRect(minX, minY, Math.max(6, maxX - minX), Math.max(6, maxY - minY));
    rangeCtx.globalAlpha = 1;
    rangeCtx.fillStyle = color;
    rangeCtx.font = isSelected ? "700 12px Inter, sans-serif" : "600 11px Inter, sans-serif";
    const scoreText = detection.score != null ? ` ${Math.round(detection.score * 100)}%` : "";
    rangeCtx.fillText(`${formatLabel(detection.label)}${scoreText}`, minX, Math.max(14, minY - 6));
  }
}

function renderRangeView(frame, selectedInfo) {
  if (!scanlineModeSupported()) return;
  if (!rangeCanvas) return;
  const width = Math.max(1, container.clientWidth);
  const height = Math.max(1, container.clientHeight);
  if (rangeCanvas.width !== width || rangeCanvas.height !== height) {
    rangeCanvas.width = width;
    rangeCanvas.height = height;
  }

  rangeCtx.clearRect(0, 0, width, height);
  rangeCtx.fillStyle = "#07090d";
  rangeCtx.fillRect(0, 0, width, height);
  drawRangeGrid(width, height);

  const points = detailPointsForFrame(frame);
  const scanlineSamples = scanlinePointsForFrame(frame);
  const scanShape = scanlineShapeForFrame(frame);
  const extents = rangeFrameExtents(points);
  const selectedDetId = selectedInfo?.detection?.detection_id || null;
  const visibleDetections = detectionsForFrame(frame).filter((detection) => isLabelVisible(detection.label));

  if (state.overlays.rawPoints) {
    if (scanlineSamples.length && scanShape) {
      drawScanlineSampleSet(frame, scanlineSamples, "rgba(139,111,191,1)", 0.42, width, height);
    } else {
      drawRangePointSet(points, "rgba(139,111,191,1)", 1.1, 0.38, width, height, extents);
    }
  }

  if (state.overlays.objectPoints) {
    for (const detection of visibleDetections) {
      const isSelected = detection.detection_id === selectedDetId;
      if (scanlineSamples.length && scanShape) {
        const ownedSamples = ownedScanlineSamplesForDetection(frame, detection.detection_id);
        if (!ownedSamples.length) continue;
        drawScanlineSampleSet(
          frame,
          ownedSamples,
          labelColor(detection.label).hex,
          isSelected ? 0.95 : (selectedInfo ? 0.32 : 0.82),
          width,
          height,
          { scale: isSelected ? 1.4 : 1.2 },
        );
      } else {
        const ownedPoints = ownedPointsForDetection(detection.detection_id).map((point) => point.sourcePosition);
        if (!ownedPoints.length) continue;
        drawRangePointSet(
          ownedPoints,
          labelColor(detection.label).hex,
          isSelected ? 2.0 : 1.6,
          isSelected ? 0.95 : (selectedInfo ? 0.32 : 0.78),
          width,
          height,
          extents,
        );
      }
    }
  }

  if (state.overlays.selectedPoints && selectedInfo?.detection) {
    if (scanlineSamples.length && scanShape) {
      const ownedSamples = ownedScanlineSamplesForDetection(frame, selectedInfo.detection.detection_id);
      drawScanlineSampleSet(frame, ownedSamples, "#fff5b8", 0.95, width, height, { scale: 1.7 });
    } else {
      const ownedPoints = ownedPointsForDetection(selectedInfo.detection.detection_id).map((point) => point.sourcePosition);
      drawRangePointSet(ownedPoints, "#fff5b8", 2.4, 0.95, width, height, extents);
    }
  }

  if (state.overlays.boxes) {
    drawRangeDetections(frame, width, height, extents, selectedInfo);
  }

  rangeCtx.fillStyle = "rgba(212, 216, 228, 0.76)";
  rangeCtx.font = "600 12px Inter, sans-serif";
  rangeCtx.fillText("Full-frame scanline view", 18, 24);
  rangeCtx.fillText(`${points.length} pts`, width - 86, 24);
}

function updateViewModeUI() {
  const spatialButton = document.getElementById("btn-view-spatial");
  const scanlineButton = document.getElementById("btn-view-scanline");
  if (!scanlineModeSupported() && state.viewMode === "scanline") {
    state.viewMode = "spatial";
  }
  const spatialActive = state.viewMode === "spatial";
  spatialButton.classList.toggle("is-active", spatialActive);
  scanlineButton.classList.toggle("is-active", !spatialActive && scanlineModeSupported());
  scanlineButton.disabled = !scanlineModeSupported();
  scanlineButton.title = scanlineModeSupported()
    ? ""
    : "Scanline mode is currently only supported for sensor-frame replays.";
  renderer.domElement.style.display = spatialActive ? "block" : "none";
  rangeCanvas.style.display = spatialActive ? "none" : "block";
  labelContainer.style.display = spatialActive ? "block" : "none";
}

function updateReferenceFrameUI() {
  const sensorButton = document.getElementById("btn-ref-sensor");
  const worldButton = document.getElementById("btn-ref-world");
  const available = new Set(state.bundle?.available_reference_frames || ["sensor"]);
  const current = currentReferenceFrame();

  sensorButton.disabled = !available.has("sensor");
  worldButton.disabled = !available.has("world");
  sensorButton.classList.toggle("is-active", current === "sensor");
  worldButton.classList.toggle("is-active", current === "world");
  worldButton.title = available.has("world")
    ? ""
    : "World view is only available for replays generated with GPS world alignment.";
}

function findMatchedTrackForDetection(detection, tracks) {
  if (!tracks?.length) return null;
  let best = null;
  let bestDist = Infinity;
  for (const track of tracks) {
    const dx = track.centroid[0] - detection.centroid[0];
    const dy = track.centroid[1] - detection.centroid[1];
    const dist = Math.hypot(dx, dy);
    if (dist < bestDist && dist < 3.0) {
      best = track;
      bestDist = dist;
    }
  }
  return best;
}

function findMatchedDetectionForTrack(track, detections) {
  if (!detections?.length) return null;
  let best = null;
  let bestDist = Infinity;
  for (const detection of detections) {
    const dx = detection.centroid[0] - track.centroid[0];
    const dy = detection.centroid[1] - track.centroid[1];
    const dist = Math.hypot(dx, dy);
    if (dist < bestDist && dist < 3.0) {
      best = detection;
      bestDist = dist;
    }
  }
  return best;
}

function resolveSelectedObject(frame) {
  if (!frame || !state.selected) return null;
  const frameDetections = detectionsForFrame(frame);
  const frameTracks = activeTracksForFrame(frame);

  if (state.selected.kind === "detection") {
    const detection = frameDetections.find((item) => item.detection_id === state.selected.id);
    if (!detection) return null;
    return {
      detection,
      track: findMatchedTrackForDetection(detection, frameTracks),
      volume: detection,
    };
  }

  if (state.selected.kind === "track") {
    const track = frameTracks.find((item) => item.track_id === state.selected.id);
    if (!track) return null;
    const detection = findMatchedDetectionForTrack(track, frameDetections);
    return {
      detection,
      track,
      volume: detection || track,
    };
  }

  return null;
}

function showTrackPanel(track) {
  const panel = document.getElementById("inspect-panel");
  const content = document.getElementById("inspect-content");
  if (!track) {
    panel.style.display = "none";
    return;
  }

  const v = track.velocity_mps;
  const speed = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  const rows = [
    ["Track", `#${track.track_id}`],
    ["Class", track.label || "object"],
    ["Position", `[${track.centroid[0].toFixed(1)}, ${track.centroid[1].toFixed(1)}, ${track.centroid[2].toFixed(1)}]`],
    ["Speed", speed < 100 ? `${speed.toFixed(1)} m/s` : "noisy"],
    ["Hits", `${track.hits} / ${track.age_frames} frames`],
    ["Distance", `${track.total_distance_m.toFixed(2)} m`],
  ];

  const color = labelColor(track.label);
  const hero = `
    <div class="inspect-hero">
      <div class="inspect-pill">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color.hex}"></span>
        Track ${track.track_id}
      </div>
      <div class="inspect-title">
        <div class="inspect-name">${formatLabel(track.label)}</div>
        <div class="inspect-score">${speed < 100 ? `${speed.toFixed(1)} m/s` : "noisy"}</div>
      </div>
    </div>
    <div class="inspect-scanline-block">
      <canvas id="inspect-scanline" width="252" height="120"></canvas>
    </div>
  `;

  content.innerHTML = hero + rows.map(([k, v]) =>
    `<div class="row"><dt>${k}</dt><dd>${v}</dd></div>`
  ).join("");
  panel.style.display = "block";

  const currentFrame = state.bundle?.frame_summaries?.[state.frameIndex];
  const matchedDetection = findMatchedDetectionForTrack(track, detectionsForFrame(currentFrame));
  const structuredSamples = matchedDetection && currentFrame
    ? ownedScanlineSamplesForDetection(currentFrame, matchedDetection.detection_id)
    : [];
  if (structuredSamples.length && currentFrame?.scanline_shape) {
    renderInspectStructuredScanline(currentFrame, structuredSamples, matchedDetection?.label || track.label);
  } else {
    renderInspectScanline(
      matchedDetection ? ownedPointsForDetection(matchedDetection.detection_id) : [],
      matchedDetection?.label || track.label,
    );
  }
}

function selectObject(selection) {
  state.selected = selection;
  const frame = state.bundle?.frame_summaries?.[state.frameIndex];
  const selectedInfo = resolveSelectedObject(frame);
  if (!selection) {
    hideInspectPanel();
    showFrame();
    return;
  }
  if (selectedInfo?.detection) {
    showInspectPanel(selectedInfo.detection, activeTracksForFrame(frame));
  } else if (selectedInfo?.track) {
    showTrackPanel(selectedInfo.track);
  } else {
    hideInspectPanel();
  }
  showFrame();
}

// ─── Sidebar panels ─────────────────────────────────────────────

function buildEventTimeline(bundle) {
  if (!bundle?.event_summaries?.length) return [];
  const frameByTimestamp = new Map(
    bundle.frame_summaries.map((frame, index) => [frame.timestamp_ns, index]),
  );
  const trackLabelById = new Map(
    (bundle.track_summaries || []).map((track) => [track.track_id, track.label || "object"]),
  );
  return bundle.event_summaries
    .map((event, index) => ({
      ...event,
      key: `${event.timestamp_ns}-${event.track_id}-${event.zone_id}-${event.rule_kind || "zone"}-${index}`,
      frame_index: frameByTimestamp.get(event.timestamp_ns) ?? 0,
      label: trackLabelById.get(event.track_id) || "object",
      rule_id: event.zone_id,
      rule_name: event.zone_name,
      rule_kind: event.rule_kind || "zone",
    }))
    .sort((left, right) => left.frame_index - right.frame_index);
}

function renderEventControls(bundle) {
  const typeEl = document.getElementById("event-filter-type");
  const classEl = document.getElementById("event-filter-class");
  const ruleEl = document.getElementById("event-filter-rule");
  if (!typeEl || !classEl || !ruleEl) return;

  if (!bundle) {
    typeEl.innerHTML = '<option value="all">All</option>';
    classEl.innerHTML = '<option value="all">All</option>';
    ruleEl.innerHTML = '<option value="all">All</option>';
    typeEl.value = "all";
    classEl.value = "all";
    ruleEl.value = "all";
    return;
  }

  const timeline = state.eventTimeline || [];
  const eventTypes = [...new Set(timeline.map((event) => event.event_type))].sort();
  const eventLabels = [...new Set(timeline.map((event) => normalizedLabel(event.label)))].sort();
  const eventRules = [...new Map(
    timeline.map((event) => [event.rule_id, event.rule_name]),
  ).entries()].sort((left, right) => left[1].localeCompare(right[1]));

  typeEl.innerHTML = ['<option value="all">All</option>']
    .concat(eventTypes.map((value) => `<option value="${value}">${formatEventType(value)}</option>`))
    .join("");
  classEl.innerHTML = ['<option value="all">All</option>']
    .concat(eventLabels.map((value) => `<option value="${value}">${formatLabel(value)}</option>`))
    .join("");
  ruleEl.innerHTML = ['<option value="all">All</option>']
    .concat(eventRules.map(([value, label]) => `<option value="${value}">${label}</option>`))
    .join("");

  typeEl.value = eventTypes.includes(state.eventFilters.type) ? state.eventFilters.type : "all";
  classEl.value = eventLabels.includes(state.eventFilters.label) ? state.eventFilters.label : "all";
  ruleEl.value = eventRules.some(([value]) => value === state.eventFilters.rule) ? state.eventFilters.rule : "all";
}

function exportEventsAsJson() {
  const timeline = filteredEventTimeline();
  const payload = {
    project: state.bundle?.project || "Depthyn",
    detector: state.bundle?.metrics?.detector_name || "unknown",
    reference_frame: currentReferenceFrame(),
    filters: { ...state.eventFilters },
    event_count: timeline.length,
    events: timeline,
  };
  downloadText("depthyn-events.json", JSON.stringify(payload, null, 2), "application/json");
}

function exportEventsAsCsv() {
  const timeline = filteredEventTimeline();
  const rows = [
    ["frame_index", "timestamp_ns", "event_type", "rule_kind", "rule_name", "track_id", "label", "direction", "dwell_seconds"],
    ...timeline.map((event) => [
      String(event.frame_index),
      String(event.timestamp_ns),
      event.event_type || "",
      event.rule_kind || "",
      event.rule_name || "",
      String(event.track_id),
      event.label || "",
      event.direction || "",
      event.dwell_seconds != null ? String(event.dwell_seconds) : "",
    ]),
  ];
  const csv = rows
    .map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(","))
    .join("\n");
  downloadText("depthyn-events.csv", csv, "text/csv");
}

function downloadText(filename, text, mimeType) {
  const blob = new Blob([text], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function renderEventMarkers(bundle) {
  const markersEl = document.getElementById("frame-event-markers");
  if (!markersEl) return;
  const frameCount = Math.max(1, bundle?.frame_summaries?.length || 1);
  const timeline = filteredEventTimeline();
  if (!timeline.length) {
    markersEl.innerHTML = "";
    return;
  }
  markersEl.innerHTML = timeline.map((event) => {
    const left = frameCount <= 1
      ? 0
      : (event.frame_index / (frameCount - 1)) * 100;
    const direction = event.direction ? ` · ${formatDirection(event.direction)}` : "";
    return `<button class="event-marker" data-event-key="${event.key}" title="${formatEventType(event.event_type)} · ${event.rule_name}${direction} · frame ${event.frame_index + 1}" style="left:${left}%;background:${eventTypeColor(event.event_type)}"></button>`;
  }).join("");
}

function renderEventPanel(bundle) {
  const eventListEl = document.getElementById("event-list");
  if (!eventListEl) return;
  const allEvents = state.eventTimeline || [];
  const timeline = filteredEventTimeline();
  if (!allEvents.length) {
    eventListEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;">No scene events in this replay.</div>';
    return;
  }
  if (!timeline.length) {
    eventListEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;">No scene events match the current filters.</div>';
    return;
  }

  const currentFrame = state.frameIndex;
  const windowStart = Math.max(0, currentFrame - 15);
  const windowEnd = currentFrame + 20;
  const nearby = timeline.filter((event) => event.frame_index >= windowStart && event.frame_index <= windowEnd);
  const visibleEvents = nearby.length ? nearby : timeline.slice(0, 24);

  eventListEl.innerHTML = visibleEvents.map((event) => {
    const isCurrent = event.frame_index === currentFrame;
    const badgeColor = eventTypeColor(event.event_type);
    const label = formatLabel(event.label);
    const dwell = event.dwell_seconds != null ? ` · ${event.dwell_seconds.toFixed(1)}s` : "";
    const direction = event.direction ? `<span class="event-direction">${formatDirection(event.direction)}</span>` : "";
    return `<div class="event-card${isCurrent ? " is-current" : ""}">
      <button class="event-jump" data-event-key="${event.key}">Jump</button>
      <div class="event-main">
        <div class="event-title">
          <span class="event-badge" style="background:${badgeColor}">${formatEventType(event.event_type)}</span>
          <span>${event.rule_name}</span>
        </div>
        <div class="event-subtitle">
          <span>Track #${event.track_id} · ${label}${dwell}</span>
          ${direction}
        </div>
      </div>
      <div class="event-meta">F${event.frame_index + 1}</div>
    </div>`;
  }).join("");
}

function renderRulePanel(bundle) {
  const ruleListEl = document.getElementById("rule-list");
  const statusEl = document.getElementById("rule-draw-status");
  if (!ruleListEl || !statusEl) return;
  if (!bundle) {
    statusEl.textContent = "Open a replay JSON to start authoring.";
    ruleListEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;">No replay loaded.</div>';
    return;
  }

  const rules = currentRuleState();
  const reference = currentReferenceFrame();
  if (state.authoring.mode === "zone") {
    statusEl.textContent = state.authoring.points.length === 0
      ? `Zone draw active in ${reference}. Click the first ground corner.`
      : `Zone draw active in ${reference}. Click the opposite corner.`;
  } else if (state.authoring.mode === "tripwire") {
    statusEl.textContent = state.authoring.points.length === 0
      ? `Tripwire draw active in ${reference}. Click the start point.`
      : `Tripwire draw active in ${reference}. Click the end point.`;
  } else {
    statusEl.textContent = `${reference.charAt(0).toUpperCase() + reference.slice(1)} rules: ${rules.zones.length} zones, ${rules.tripwires.length} tripwires.${state.rulePreviewApplied ? " Replay preview is active." : ""}`;
  }

  const cards = [];
  for (const zone of rules.zones) {
    cards.push(`
      <div class="rule-card" data-rule-kind="zone" data-rule-id="${zone.zone_id}">
        <div class="rule-card-header">
          <span class="rule-kind-badge">Zone</span>
          <button class="rule-delete" data-rule-kind="zone" data-rule-id="${zone.zone_id}">Delete</button>
        </div>
        <div class="rule-form">
          <label><span>Name</span><input data-field="name" value="${zone.name}"></label>
          <label><span>Color</span><input data-field="color" value="${zone.color}"></label>
          <label><span>Dwell</span><input data-field="dwell_alert_seconds" type="number" min="0" step="0.1" value="${zone.dwell_alert_seconds ?? 0}"></label>
        </div>
        <div class="rule-geometry">min_xy: [${zone.min_xy.map((v) => v.toFixed(2)).join(", ")}] · max_xy: [${zone.max_xy.map((v) => v.toFixed(2)).join(", ")}]</div>
      </div>
    `);
  }
  for (const tripwire of rules.tripwires) {
    cards.push(`
      <div class="rule-card" data-rule-kind="tripwire" data-rule-id="${tripwire.tripwire_id}">
        <div class="rule-card-header">
          <span class="rule-kind-badge">Tripwire</span>
          <button class="rule-delete" data-rule-kind="tripwire" data-rule-id="${tripwire.tripwire_id}">Delete</button>
        </div>
        <div class="rule-form">
          <label><span>Name</span><input data-field="name" value="${tripwire.name}"></label>
          <label><span>Color</span><input data-field="color" value="${tripwire.color}"></label>
          <label><span>+</span><input data-field="positive_direction_label" value="${tripwire.positive_direction_label}"></label>
          <label><span>-</span><input data-field="negative_direction_label" value="${tripwire.negative_direction_label}"></label>
        </div>
        <div class="rule-geometry">start_xy: [${tripwire.start_xy.map((v) => v.toFixed(2)).join(", ")}] · end_xy: [${tripwire.end_xy.map((v) => v.toFixed(2)).join(", ")}]</div>
      </div>
    `);
  }

  ruleListEl.innerHTML = cards.length
    ? cards.join("")
    : '<div style="color:var(--text-muted);font-size:12px;">No authored rules for this reference frame yet.</div>';
}

function jumpToEventByKey(eventKey) {
  const event = (state.eventTimeline || []).find((item) => item.key === eventKey);
  if (!event) return;
  setFrameIndex(event.frame_index);
  selectObject({ kind: "track", id: event.track_id });
}

function updateSidebar() {
  const statsEl = document.getElementById("scene-stats");
  const objectsEl = document.getElementById("object-list");
  const classControlsEl = document.getElementById("class-controls");
  const statusEl = document.getElementById("status-label");

  const bundle = state.bundle;
  const frame = bundle ? bundle.frame_summaries[state.frameIndex] : null;

  if (!bundle || !frame) {
    statsEl.innerHTML = "";
    classControlsEl.innerHTML = "";
    objectsEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;">Open a replay JSON to begin.</div>';
    renderRulePanel(null);
    renderEventControls(null);
    renderEventPanel(null);
    renderEventMarkers(null);
    return;
  }

  statusEl.textContent = `${bundle.frames_processed} frames · ${bundle.metrics.detector_name || "baseline"}`;

  const referenceFrame = currentReferenceFrame();
  const frameAxes = referenceFrame === "world"
    ? "+X east · +Y north"
    : "+X forward · +Y left";
  const eventCount = frame.scene_state?.events?.length || 0;
  const frameDetections = detectionsForFrame(frame);
  const frameTracks = activeTracksForFrame(frame);
  const rows = [
    ["Frame", `${state.frameIndex + 1} / ${bundle.frame_summaries.length}`],
    ["Points", frame.points_after_filtering],
    ["Detections", frame.detection_count],
    ["Tracks", frameTracks.length],
    ["Events", eventCount],
    ["Mode", bundle.config.mode],
    ["Reference", referenceFrame],
    ["Frame axes", frameAxes],
  ];
  if (bundle.rule_preview?.applied) {
    rows.push(["Rule preview", `applied in ${bundle.rule_preview.reference_frame}`]);
  }
  if (!scanlineModeSupported()) {
    rows.push(["Scanline", "disabled in world mode"]);
  }
  statsEl.innerHTML = rows.map(([k, v]) =>
    `<div class="row"><dt>${k}</dt><dd>${v}</dd></div>`
  ).join("");

  const classCounts = new Map();
  for (const det of frameDetections) {
    const key = normalizedLabel(det.label);
    classCounts.set(key, (classCounts.get(key) || 0) + 1);
  }
  for (const track of frameTracks) {
    const key = normalizedLabel(track.label);
    classCounts.set(key, (classCounts.get(key) || 0) + 1);
  }
  const orderedLabels = [...classCounts.keys()].sort();
  classControlsEl.innerHTML = orderedLabels.map((label) => {
    const color = labelColor(label);
    const checked = isLabelVisible(label) ? "checked" : "";
    return `<label class="class-row">
      <input type="checkbox" data-class-label="${label}" ${checked}>
      <span class="obj-dot" style="background:${color.hex}"></span>
      <span class="class-label">${formatLabel(label)}</span>
      <span class="class-count">${classCounts.get(label)}</span>
    </label>`;
  }).join("");

  // Object cards: detections first, then tracks
  const items = [];
  for (const det of frameDetections) {
    if (!isLabelVisible(det.label)) continue;
    items.push({
      type: "det",
      label: det.label || "object",
      score: det.score,
      id: det.detection_id,
      color: labelColor(det.label).hex,
    });
  }
  for (const track of frameTracks) {
    const trackLabel = track.label || "object";
    if (!isLabelVisible(trackLabel)) continue;
    const trackColor = labelColor(trackLabel);
    const v = track.velocity_mps;
    const speed = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    const speedText = speed > 100 ? "noisy" : speed > 0.1 ? `${speed.toFixed(1)} m/s` : "stationary";
    items.push({
      type: "track",
      label: `Track #${track.track_id}`,
      sublabel: trackLabel,
      score: null,
      id: `track-${track.track_id}`,
      color: trackColor.hex,
      meta: `${track.hits} hits · ${speedText}`,
    });
  }

  if (items.length === 0) {
    objectsEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;">No objects in this frame.</div>';
  } else {
    objectsEl.innerHTML = items.map(it => {
      const scoreText = it.score != null ? `${Math.round(it.score * 100)}%` : (it.meta || "");
      const isSelected = state.selected && (
        (it.type === "det" && state.selected.kind === "detection" && state.selected.id === it.id) ||
        (it.type === "track" && state.selected.kind === "track" && state.selected.id === Number(it.id.replace("track-", "")))
      );
      return `<div class="obj-card${isSelected ? " is-selected" : ""}">
        <button class="obj-focus" data-kind="${it.type === "det" ? "detection" : "track"}" data-id="${it.id}" title="Focus object">Focus</button>
        <div class="obj-dot" style="background:${it.color}"></div>
        <span class="obj-label">${it.label}</span>
        <span class="obj-meta">${scoreText}</span>
      </div>`;
    }).join("");
  }

  renderRulePanel(bundle);
  renderEventControls(bundle);
  renderEventPanel(bundle);
  renderEventMarkers(bundle);
}

function buildLegend() {
  const el = document.getElementById("legend-list");
  const items = [
    ["Point cloud", "#8b6fbf"],
    ["Object points", "#ffd48a"],
    ["Selected points", "#fff5b8"],
    ["Ego forward", "#ff6a3d"],
    ["Motion vector", "#ffdb7a"],
    ["Event marker", "#7ee787"],
    ["Car", COLORS.car.hex],
    ["Truck", COLORS.truck.hex],
    ["Pedestrian", COLORS.pedestrian.hex],
    ["Bicycle", COLORS.bicycle.hex],
    ["Track trail", COLORS.track.hex],
  ];
  el.innerHTML = items.map(([label, color]) =>
    `<li><span class="legend-swatch" style="background:${color}"></span>${label}</li>`
  ).join("");
}
buildLegend();

function initializeOverlayControls() {
  const pointToggles = [
    ["toggle-raw-points", "rawPoints"],
    ["toggle-object-points", "objectPoints"],
    ["toggle-selected-points", "selectedPoints"],
    ["toggle-boxes", "boxes"],
    ["toggle-trails", "trails"],
    ["toggle-motion", "motionVector"],
    ["toggle-events", "events"],
  ];
  const worldAxesToggle = document.getElementById("toggle-world-axes");
  const egoMarkerToggle = document.getElementById("toggle-ego-marker");

  for (const [id, key] of pointToggles) {
    const toggle = document.getElementById(id);
    toggle.checked = state.overlays[key];
    toggle.addEventListener("change", (event) => {
      state.overlays[key] = event.target.checked;
      applyOverlayVisibility();
      updateLabels();
    });
  }
  worldAxesToggle.checked = state.overlays.worldAxes;
  egoMarkerToggle.checked = state.overlays.egoMarker;

  worldAxesToggle.addEventListener("change", (event) => {
    state.overlays.worldAxes = event.target.checked;
    applyOverlayVisibility();
    updateLabels();
  });
  egoMarkerToggle.addEventListener("change", (event) => {
    state.overlays.egoMarker = event.target.checked;
    applyOverlayVisibility();
    updateLabels();
  });
}
initializeOverlayControls();

function initializeClassVisibility(bundle) {
  const labels = new Set(["car", "truck", "bus", "pedestrian", "bicycle", "object"]);
  const counts = bundle?.metrics?.label_counts || {};
  for (const label of Object.keys(counts)) {
    labels.add(normalizedLabel(label));
  }
  for (const frame of bundle?.frame_summaries || []) {
    for (const det of frame.sensor_detections || frame.detections || []) {
      labels.add(normalizedLabel(det.label));
    }
    for (const track of frame.sensor_active_tracks || frame.active_tracks || []) {
      labels.add(normalizedLabel(track.label));
    }
  }
  for (const label of labels) {
    if (!(label in state.classVisibility)) {
      state.classVisibility[label] = true;
    }
  }
}

// ─── Frame management ───────────────────────────────────────────

function setBundle(bundle) {
  state.sourceBundle = cloneBundle(bundle);
  state.bundle = cloneBundle(bundle);
  state.frameIndex = 0;
  stopPlayback();
  state.selected = null;
  state.referenceFrame = state.bundle.reference_frame || "sensor";
  state.classVisibility = {};
  state.eventTimeline = buildEventTimeline(state.bundle);
  state.eventFilters = { type: "all", label: "all", rule: "all" };
  state.rulePreviewApplied = false;
  initializeRuleSets(state.bundle);
  cancelRuleDrawing();
  document.getElementById("inspect-panel").style.display = "none";
  initializeClassVisibility(state.bundle);

  const slider = document.getElementById("frame-slider");
  slider.min = 0;
  slider.max = Math.max(0, state.bundle.frame_summaries.length - 1);
  slider.value = 0;

  // Center camera on scene
  const b = state.bundle.scene_bounds;
  const cx = (b.min[0] + b.max[0]) / 2;
  const cy = (b.min[1] + b.max[1]) / 2;
  controls.target.set(cx, cy, 0);
  camera.position.set(cx, cy - 50, 40);
  buildEgoMarker();
  applyOverlayVisibility();
  updateReferenceFrameUI();
  updateViewModeUI();

  showFrame();
}

function showFrame() {
  if (!state.bundle) return;
  const frame = state.bundle.frame_summaries[state.frameIndex];
  const selectedInfo = resolveSelectedObject(frame);
  const frameDetections = detectionsForFrame(frame);
  const frameTracks = activeTracksForFrame(frame);
  applyFramePose(frame);
  buildPointCloud(frame, frameDetections, selectedInfo);
  buildBoxes(frameDetections, frameTracks);
  buildTrails(state.bundle, state.frameIndex);
  buildMotionCue(frame, selectedInfo);
  buildEvents(frame, state.bundle);
  buildRuleOverlays();
  applyOverlayVisibility();
  updateSidebar();
  renderRangeView(frame, selectedInfo);
  if (selectedInfo?.detection) {
    const structuredSamples = ownedScanlineSamplesForDetection(frame, selectedInfo.detection.detection_id);
    if (structuredSamples.length && frame.scanline_shape) {
      renderInspectStructuredScanline(frame, structuredSamples, selectedInfo.detection.label);
    } else {
      renderInspectScanline(ownedPointsForDetection(selectedInfo.detection.detection_id), selectedInfo.detection.label);
    }
  } else if (selectedInfo?.track) {
    const matchedDetection = findMatchedDetectionForTrack(selectedInfo.track, frameDetections);
    const structuredSamples = matchedDetection
      ? ownedScanlineSamplesForDetection(frame, matchedDetection.detection_id)
      : [];
    if (structuredSamples.length && frame.scanline_shape) {
      renderInspectStructuredScanline(frame, structuredSamples, matchedDetection?.label || selectedInfo.track.label);
    } else {
      renderInspectScanline(
        matchedDetection ? ownedPointsForDetection(matchedDetection.detection_id) : [],
        matchedDetection?.label || selectedInfo.track.label,
      );
    }
  }

  document.getElementById("frame-counter").textContent =
    `${state.frameIndex + 1} / ${state.bundle.frame_summaries.length}`;
  document.getElementById("frame-slider").value = state.frameIndex;
}

function setFrameIndex(idx) {
  if (!state.bundle) return;
  state.frameIndex = Math.max(0, Math.min(state.bundle.frame_summaries.length - 1, idx));
  showFrame();
}

function pointOnGroundPlane(clientX, clientY) {
  mouse.x = (clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const target = new THREE.Vector3();
  const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  if (!raycaster.ray.intersectPlane(plane, target)) {
    return null;
  }
  return [roundPoint(target.x), roundPoint(target.y), 0];
}

// ─── Playback ───────────────────────────────────────────────────

function startPlayback() {
  if (!state.bundle || state.playing) return;
  const baseInterval = state.bundle.playback.median_frame_interval_ms || 100;
  const speed = Number(document.getElementById("speed-select").value) || 1;
  const interval = Math.max(20, baseInterval / speed);
  state.playing = true;
  document.getElementById("btn-play").textContent = "Pause";
  state.timer = setInterval(() => {
    if (state.frameIndex >= state.bundle.frame_summaries.length - 1) {
      stopPlayback();
      return;
    }
    setFrameIndex(state.frameIndex + 1);
  }, interval);
}

function stopPlayback() {
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  state.playing = false;
  document.getElementById("btn-play").textContent = "Play";
}

// ─── Event binding ──────────────────────────────────────────────

document.getElementById("btn-play").addEventListener("click", () => {
  state.playing ? stopPlayback() : startPlayback();
});
document.getElementById("btn-prev").addEventListener("click", () => setFrameIndex(state.frameIndex - 1));
document.getElementById("btn-next").addEventListener("click", () => setFrameIndex(state.frameIndex + 1));
document.getElementById("frame-slider").addEventListener("input", (e) => setFrameIndex(Number(e.target.value)));
document.getElementById("speed-select").addEventListener("change", () => {
  if (state.playing) { stopPlayback(); startPlayback(); }
});

document.getElementById("file-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const text = await file.text();
  setBundle(JSON.parse(text));
});

document.getElementById("btn-view-spatial").addEventListener("click", () => {
  state.viewMode = "spatial";
  updateViewModeUI();
  showFrame();
});

document.getElementById("btn-view-scanline").addEventListener("click", () => {
  if (!scanlineModeSupported()) return;
  state.viewMode = "scanline";
  updateViewModeUI();
  showFrame();
});

document.getElementById("btn-ref-sensor").addEventListener("click", () => {
  if (!state.bundle) return;
  cancelRuleDrawing();
  state.referenceFrame = "sensor";
  if (state.rulePreviewApplied) {
    applyRulesToReplay();
  }
  updateReferenceFrameUI();
  updateViewModeUI();
  showFrame();
});

document.getElementById("btn-ref-world").addEventListener("click", () => {
  if (!state.bundle) return;
  const available = new Set(state.bundle.available_reference_frames || []);
  if (!available.has("world")) return;
  cancelRuleDrawing();
  state.referenceFrame = "world";
  if (state.rulePreviewApplied) {
    applyRulesToReplay();
  }
  updateReferenceFrameUI();
  updateViewModeUI();
  showFrame();
});

document.getElementById("inspect-close").addEventListener("click", hideInspectPanel);
document.getElementById("class-controls").addEventListener("change", (e) => {
  const toggle = e.target.closest("[data-class-label]");
  if (!toggle) return;
  const label = toggle.dataset.classLabel;
  state.classVisibility[label] = toggle.checked;
  const frame = state.bundle?.frame_summaries?.[state.frameIndex];
  const selectedInfo = resolveSelectedObject(frame);
  const selectedLabel = selectedInfo?.detection?.label || selectedInfo?.track?.label;
  if (selectedLabel && !isLabelVisible(selectedLabel)) {
    state.selected = null;
    document.getElementById("inspect-panel").style.display = "none";
  }
  showFrame();
});
document.getElementById("object-list").addEventListener("click", (e) => {
  const focusButton = e.target.closest(".obj-focus");
  if (!focusButton || !state.bundle) return;
  const kind = focusButton.dataset.kind;
  const rawId = focusButton.dataset.id;
  if (kind === "detection") {
    selectObject({ kind: "detection", id: rawId });
    return;
  }
  if (kind === "track") {
    selectObject({ kind: "track", id: Number(rawId.replace("track-", "")) });
  }
});

document.getElementById("event-list").addEventListener("click", (e) => {
  const jumpButton = e.target.closest(".event-jump");
  if (!jumpButton) return;
  jumpToEventByKey(jumpButton.dataset.eventKey);
});

document.getElementById("event-filter-type").addEventListener("change", (e) => {
  state.eventFilters.type = e.target.value;
  updateSidebar();
});

document.getElementById("event-filter-class").addEventListener("change", (e) => {
  state.eventFilters.label = e.target.value;
  updateSidebar();
});

document.getElementById("event-filter-rule").addEventListener("change", (e) => {
  state.eventFilters.rule = e.target.value;
  updateSidebar();
});

document.getElementById("btn-export-events-json").addEventListener("click", () => {
  exportEventsAsJson();
});

document.getElementById("btn-export-events-csv").addEventListener("click", () => {
  exportEventsAsCsv();
});

document.getElementById("btn-add-zone").addEventListener("click", () => {
  if (!state.bundle) return;
  if (isScanlineMode()) {
    state.viewMode = "spatial";
    updateViewModeUI();
  }
  startRuleDrawing("zone");
});

document.getElementById("btn-add-tripwire").addEventListener("click", () => {
  if (!state.bundle) return;
  if (isScanlineMode()) {
    state.viewMode = "spatial";
    updateViewModeUI();
  }
  startRuleDrawing("tripwire");
});

document.getElementById("btn-cancel-rule-draw").addEventListener("click", () => {
  cancelRuleDrawing();
  showFrame();
});

document.getElementById("btn-export-rules-json").addEventListener("click", () => {
  if (!state.bundle) return;
  exportRulesJson();
});

document.getElementById("btn-apply-rules").addEventListener("click", () => {
  if (!state.bundle) return;
  cancelRuleDrawing();
  applyRulesToReplay();
  showFrame();
});

document.getElementById("btn-reset-rule-preview").addEventListener("click", () => {
  if (!state.bundle) return;
  cancelRuleDrawing();
  resetRulePreview();
  showFrame();
});

document.getElementById("rule-list").addEventListener("input", (e) => {
  const card = e.target.closest(".rule-card");
  const input = e.target.closest("input[data-field]");
  if (!card || !input) return;
  const rules = currentRuleState();
  const collection = card.dataset.ruleKind === "zone" ? rules.zones : rules.tripwires;
  const idField = card.dataset.ruleKind === "zone" ? "zone_id" : "tripwire_id";
  const item = collection.find((entry) => String(entry[idField]) === card.dataset.ruleId);
  if (!item) return;
  item[input.dataset.field] = input.type === "number" ? Number(input.value || 0) : input.value;
  if (state.rulePreviewApplied) {
    applyRulesToReplay();
  }
  buildRuleOverlays();
  updateSidebar();
});

document.getElementById("rule-list").addEventListener("click", (e) => {
  const button = e.target.closest(".rule-delete");
  if (!button) return;
  const rules = currentRuleState();
  if (button.dataset.ruleKind === "zone") {
    rules.zones = rules.zones.filter((zone) => zone.zone_id !== button.dataset.ruleId);
  } else {
    rules.tripwires = rules.tripwires.filter((tripwire) => tripwire.tripwire_id !== button.dataset.ruleId);
  }
  if (state.rulePreviewApplied) {
    applyRulesToReplay();
  }
  buildRuleOverlays();
  updateSidebar();
});

document.getElementById("frame-event-markers").addEventListener("click", (e) => {
  const marker = e.target.closest(".event-marker");
  if (!marker) return;
  jumpToEventByKey(marker.dataset.eventKey);
});

updateReferenceFrameUI();
updateViewModeUI();

// Click-to-inspect on 3D boxes (distinguish click from orbit drag)
renderer.domElement.addEventListener("mousedown", (e) => {
  mouseDownPos = { x: e.clientX, y: e.clientY };
});

renderer.domElement.addEventListener("mousemove", (e) => {
  if (!state.authoring.mode || isScanlineMode()) return;
  state.authoring.hoverPoint = pointOnGroundPlane(e.clientX, e.clientY);
  buildRuleOverlays();
  updateLabels();
});

renderer.domElement.addEventListener("mouseup", (e) => {
  if (!state.bundle || !mouseDownPos) return;

  const dx = e.clientX - mouseDownPos.x;
  const dy = e.clientY - mouseDownPos.y;
  if (dx * dx + dy * dy > 9) return; // moved more than 3px = drag, not click

  if (state.authoring.mode && !isScanlineMode()) {
    const point = pointOnGroundPlane(e.clientX, e.clientY);
    if (point) {
      commitRulePoint(point);
      showFrame();
      mouseDownPos = null;
      return;
    }
  }

  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(clickableBoxes);

  if (intersects.length > 0) {
    const hit = intersects[0].object;
    const det = hit.userData.detection;
    selectObject({ kind: "detection", id: det.detection_id });
  } else {
    selectObject(null);
  }
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  if (state.bundle) {
    showFrame();
  }
});

window.addEventListener("keydown", (e) => {
  if (e.key === " ") { e.preventDefault(); state.playing ? stopPlayback() : startPlayback(); }
  if (e.key === "ArrowLeft") setFrameIndex(state.frameIndex - 1);
  if (e.key === "ArrowRight") setFrameIndex(state.frameIndex + 1);
  if (e.key === "Escape") hideInspectPanel();
});

// ─── Bootstrap ──────────────────────────────────────────────────

async function bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const dataUrl = params.get("data");
  if (dataUrl) {
    try {
      const res = await fetch(dataUrl);
      const bundle = await res.json();
      setBundle(bundle);
    } catch (err) {
      document.getElementById("status-label").textContent = `Load error: ${err.message}`;
    }
  }
}

// ─── Render loop ─────────────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  updateLabels();
}

updateViewModeUI();
bootstrap();
animate();
