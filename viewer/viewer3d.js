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

// ─── State ───────────────────────────────────────────────────────

const state = {
  bundle: null,
  frameIndex: 0,
  playing: false,
  timer: null,
  selected: null,  // selected object id
  classVisibility: {},
  overlays: {
    worldAxes: true,
    egoMarker: true,
  },
};

// ─── Three.js setup ──────────────────────────────────────────────

const container = document.getElementById("canvas-container");
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
let highlightPointCloud = null;
let boxGroup = new THREE.Group();
let trailGroup = new THREE.Group();
let egoGroup = new THREE.Group();
scene.add(boxGroup);
scene.add(trailGroup);
scene.add(egoGroup);

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

function buildPointCloud(points, selectedVolume = null) {
  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
    pointCloud.material.dispose();
  }
  if (highlightPointCloud) {
    scene.remove(highlightPointCloud);
    highlightPointCloud.geometry.dispose();
    highlightPointCloud.material.dispose();
    highlightPointCloud = null;
  }
  if (!points || !points.length) return;

  const positions = new Float32Array(points.length * 3);
  const colors = new Float32Array(points.length * 3);
  const highlightPoints = [];

  let zMin = Infinity, zMax = -Infinity;
  for (const p of points) {
    if (p[2] < zMin) zMin = p[2];
    if (p[2] > zMax) zMax = p[2];
  }
  const zSpan = Math.max(0.1, zMax - zMin);

  // Add subtle jitter to break voxel grid pattern
  const jitter = 0.08;
  for (let i = 0; i < points.length; i++) {
    const px = points[i][0] + (Math.random() - 0.5) * jitter;
    const py = points[i][1] + (Math.random() - 0.5) * jitter;
    const pz = points[i][2] + (Math.random() - 0.5) * jitter;
    positions[i * 3] = px;
    positions[i * 3 + 1] = py;
    positions[i * 3 + 2] = pz;

    const t = (points[i][2] - zMin) / zSpan;
    const r = 0.25 + t * 0.65;
    const g = 0.15 + t * 0.20;
    const b = 0.55 + (1 - t) * 0.25;
    colors[i * 3]     = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;

    if (selectedVolume && pointInVolume(points[i], selectedVolume)) {
      highlightPoints.push([px, py, pz]);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.3,
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    opacity: selectedVolume ? 0.18 : 0.85,
    depthWrite: false,
    map: circleTexture,
    alphaMap: circleTexture,
    alphaTest: 0.1,
  });

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

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
      size: 0.42,
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
  const pose = frame?.frame_pose;
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
      color: isSelected ? 0xfff1b3 : threeColor,
      linewidth: isSelected ? 3 : 2,
      transparent: true,
      opacity: isSelected ? 1.0 : (isDimmed ? 0.2 : 0.9),
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
        opacity: 0.45,
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
      opacity: isSelected ? 0.18 : (isDimmed ? 0.02 : 0.08),
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
      opacity: isSelected ? 0.45 : (isDimmed ? 0.06 : 0.2),
    });
    boxGroup.add(new THREE.Line(poleGeo, poleMat));

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

  const history = new Map();
  const labelMap = new Map();
  const start = Math.max(0, frameIndex - 30);
  for (let i = start; i <= frameIndex; i++) {
    const frame = bundle.frame_summaries[i];
    for (const track of frame.active_tracks) {
      if (!history.has(track.track_id)) history.set(track.track_id, []);
      history.get(track.track_id).push(track.centroid);
      if (track.label) labelMap.set(track.track_id, track.label);
    }
  }

  const selectedInfo = resolveSelectedObject(bundle.frame_summaries[frameIndex]);

  for (const [trackId, trail] of history) {
    if (trail.length < 2) continue;
    const points = trail.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    const trackLabel = labelMap.get(trackId) || "object";
    if (!isLabelVisible(trackLabel)) continue;
    const color = labelColor(trackLabel);
    const isSelected = selectedInfo?.track?.track_id === trackId;
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

// ─── Project 3D labels to screen ─────────────────────────────────

function updateLabels() {
  labelContainer.innerHTML = "";
  if (!state.bundle) return;

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
  `;

  content.innerHTML = hero + rows.map(([k, v]) =>
    `<div class="row"><dt>${k}</dt><dd>${v}</dd></div>`
  ).join("");

  panel.style.display = "block";
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

  if (state.selected.kind === "detection") {
    const detection = frame.detections.find((item) => item.detection_id === state.selected.id);
    if (!detection) return null;
    return {
      detection,
      track: findMatchedTrackForDetection(detection, frame.active_tracks),
      volume: detection,
    };
  }

  if (state.selected.kind === "track") {
    const track = frame.active_tracks.find((item) => item.track_id === state.selected.id);
    if (!track) return null;
    const detection = findMatchedDetectionForTrack(track, frame.detections);
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
  `;

  content.innerHTML = hero + rows.map(([k, v]) =>
    `<div class="row"><dt>${k}</dt><dd>${v}</dd></div>`
  ).join("");
  panel.style.display = "block";
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
    showInspectPanel(selectedInfo.detection, frame?.active_tracks || []);
  } else if (selectedInfo?.track) {
    showTrackPanel(selectedInfo.track);
  } else {
    hideInspectPanel();
  }
  showFrame();
}

// ─── Sidebar panels ─────────────────────────────────────────────

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
    return;
  }

  statusEl.textContent = `${bundle.frames_processed} frames · ${bundle.metrics.detector_name || "baseline"}`;

  const referenceFrame = bundle.reference_frame || "sensor";
  const frameAxes = referenceFrame === "world"
    ? "+X east · +Y north"
    : "+X forward · +Y left";
  const rows = [
    ["Frame", `${state.frameIndex + 1} / ${bundle.frame_summaries.length}`],
    ["Points", frame.points_after_filtering],
    ["Detections", frame.detection_count],
    ["Tracks", frame.active_tracks.length],
    ["Mode", bundle.config.mode],
    ["Reference", referenceFrame],
    ["Frame axes", frameAxes],
  ];
  statsEl.innerHTML = rows.map(([k, v]) =>
    `<div class="row"><dt>${k}</dt><dd>${v}</dd></div>`
  ).join("");

  const classCounts = new Map();
  for (const det of frame.detections) {
    const key = normalizedLabel(det.label);
    classCounts.set(key, (classCounts.get(key) || 0) + 1);
  }
  for (const track of frame.active_tracks) {
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
  for (const det of frame.detections) {
    if (!isLabelVisible(det.label)) continue;
    items.push({
      type: "det",
      label: det.label || "object",
      score: det.score,
      id: det.detection_id,
      color: labelColor(det.label).hex,
    });
  }
  for (const track of frame.active_tracks) {
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
}

function buildLegend() {
  const el = document.getElementById("legend-list");
  const items = [
    ["Point cloud", "#8b6fbf"],
    ["Ego forward", "#ff6a3d"],
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
  const worldAxesToggle = document.getElementById("toggle-world-axes");
  const egoMarkerToggle = document.getElementById("toggle-ego-marker");

  worldAxesToggle.checked = state.overlays.worldAxes;
  egoMarkerToggle.checked = state.overlays.egoMarker;

  worldAxesToggle.addEventListener("change", (event) => {
    state.overlays.worldAxes = event.target.checked;
    applyOverlayVisibility();
  });
  egoMarkerToggle.addEventListener("change", (event) => {
    state.overlays.egoMarker = event.target.checked;
    applyOverlayVisibility();
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
    for (const det of frame.detections || []) {
      labels.add(normalizedLabel(det.label));
    }
    for (const track of frame.active_tracks || []) {
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
  state.bundle = bundle;
  state.frameIndex = 0;
  stopPlayback();
  state.selected = null;
  state.classVisibility = {};
  document.getElementById("inspect-panel").style.display = "none";
  initializeClassVisibility(bundle);

  const slider = document.getElementById("frame-slider");
  slider.min = 0;
  slider.max = Math.max(0, bundle.frame_summaries.length - 1);
  slider.value = 0;

  // Center camera on scene
  const b = bundle.scene_bounds;
  const cx = (b.min[0] + b.max[0]) / 2;
  const cy = (b.min[1] + b.max[1]) / 2;
  controls.target.set(cx, cy, 0);
  camera.position.set(cx, cy - 50, 40);
  buildEgoMarker();
  applyOverlayVisibility();

  showFrame();
}

function showFrame() {
  if (!state.bundle) return;
  const frame = state.bundle.frame_summaries[state.frameIndex];
  const selectedInfo = resolveSelectedObject(frame);

  applyFramePose(frame);
  buildPointCloud(frame.preview_points, selectedInfo?.volume || null);
  buildBoxes(frame.detections, frame.active_tracks);
  buildTrails(state.bundle, state.frameIndex);
  updateSidebar();

  document.getElementById("frame-counter").textContent =
    `${state.frameIndex + 1} / ${state.bundle.frame_summaries.length}`;
  document.getElementById("frame-slider").value = state.frameIndex;
}

function setFrameIndex(idx) {
  if (!state.bundle) return;
  state.frameIndex = Math.max(0, Math.min(state.bundle.frame_summaries.length - 1, idx));
  showFrame();
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

// Click-to-inspect on 3D boxes (distinguish click from orbit drag)
renderer.domElement.addEventListener("mousedown", (e) => {
  mouseDownPos = { x: e.clientX, y: e.clientY };
});

renderer.domElement.addEventListener("mouseup", (e) => {
  if (!state.bundle || !mouseDownPos) return;

  const dx = e.clientX - mouseDownPos.x;
  const dy = e.clientY - mouseDownPos.y;
  if (dx * dx + dy * dy > 9) return; // moved more than 3px = drag, not click

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

bootstrap();
animate();
