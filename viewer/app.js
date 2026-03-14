const state = {
  bundle: null,
  frameIndex: 0,
  playing: false,
  timer: null,
};

const canvas = document.getElementById("scene-canvas");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("frame-slider");
const playPauseButton = document.getElementById("play-pause");
const stepBackButton = document.getElementById("step-back");
const stepForwardButton = document.getElementById("step-forward");
const speedSelect = document.getElementById("speed-select");
const fileInput = document.getElementById("file-input");
const statusPanel = document.getElementById("status-panel");
const framePanel = document.getElementById("frame-panel");

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  render();
}

function setBundle(bundle) {
  state.bundle = bundle;
  state.frameIndex = 0;
  slider.min = 0;
  slider.max = Math.max(0, bundle.frame_summaries.length - 1);
  slider.value = 0;
  stopPlayback();
  updatePanels();
  render();
}

function getFrame() {
  if (!state.bundle || !state.bundle.frame_summaries.length) {
    return null;
  }
  return state.bundle.frame_summaries[state.frameIndex];
}

function render() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  drawBackground(rect.width, rect.height);

  const bundle = state.bundle;
  const frame = getFrame();
  if (!bundle || !frame) {
    drawEmptyState(rect.width, rect.height);
    return;
  }

  const bounds = bundle.scene_bounds;
  const project = makeProjector(bounds, rect.width, rect.height);

  drawGrid(project, rect.width, rect.height);
  drawPoints(project, frame.preview_points);
  drawDetections(project, frame.detections);
  drawTrails(project, bundle.frame_summaries, state.frameIndex);
  drawTracks(project, frame.active_tracks);
}

function drawBackground(width, height) {
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "rgba(255, 253, 248, 0.95)");
  gradient.addColorStop(1, "rgba(219, 230, 224, 0.92)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
}

function drawEmptyState(width, height) {
  ctx.fillStyle = "#456257";
  ctx.font = "600 24px IBM Plex Sans";
  ctx.textAlign = "center";
  ctx.fillText("Open a Depthyn replay JSON to start playback", width / 2, height / 2);
}

function makeProjector(bounds, width, height) {
  const pad = 36;
  const minX = bounds.min[0];
  const maxX = bounds.max[0];
  const minY = bounds.min[1];
  const maxY = bounds.max[1];
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);
  const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY);
  const offsetX = (width - spanX * scale) / 2;
  const offsetY = (height - spanY * scale) / 2;

  return (point) => {
    const x = offsetX + (point[0] - minX) * scale;
    const y = height - offsetY - (point[1] - minY) * scale;
    return [x, y];
  };
}

function drawGrid(project, width, height) {
  ctx.strokeStyle = "rgba(18, 49, 38, 0.07)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= width; x += 80) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y <= height; y += 80) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawPoints(project, points) {
  ctx.fillStyle = "rgba(35, 54, 50, 0.28)";
  for (const point of points) {
    const [x, y] = project(point);
    ctx.fillRect(x, y, 2, 2);
  }
}

function drawDetections(project, detections) {
  ctx.strokeStyle = "rgba(201, 91, 43, 0.92)";
  ctx.fillStyle = "rgba(239, 180, 107, 0.14)";
  ctx.lineWidth = 2;
  for (const detection of detections) {
    const [x1, y1] = project(detection.bbox_min);
    const [x2, y2] = project(detection.bbox_max);
    const left = Math.min(x1, x2);
    const top = Math.min(y1, y2);
    const width = Math.max(8, Math.abs(x2 - x1));
    const height = Math.max(8, Math.abs(y2 - y1));
    ctx.fillRect(left, top, width, height);
    ctx.strokeRect(left, top, width, height);
  }
}

function drawTracks(project, tracks) {
  ctx.fillStyle = "rgba(11, 122, 117, 0.96)";
  ctx.strokeStyle = "rgba(245, 252, 248, 0.9)";
  ctx.lineWidth = 2;
  ctx.font = "12px IBM Plex Sans";
  for (const track of tracks) {
    const [x, y] = project(track.centroid);
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#0d2c24";
    ctx.fillText(`#${track.track_id}`, x + 10, y - 10);
    ctx.fillStyle = "rgba(11, 122, 117, 0.96)";
  }
}

function drawTrails(project, frames, frameIndex) {
  const history = new Map();
  const start = Math.max(0, frameIndex - 25);
  for (let index = start; index <= frameIndex; index += 1) {
    const frame = frames[index];
    for (const track of frame.active_tracks) {
      if (!history.has(track.track_id)) {
        history.set(track.track_id, []);
      }
      history.get(track.track_id).push(track.centroid);
    }
  }

  ctx.strokeStyle = "rgba(11, 122, 117, 0.34)";
  ctx.lineWidth = 2;
  for (const trail of history.values()) {
    if (trail.length < 2) {
      continue;
    }
    ctx.beginPath();
    trail.forEach((point, index) => {
      const [x, y] = project(point);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  }
}

function updatePanels() {
  const bundle = state.bundle;
  const frame = getFrame();
  if (!bundle || !frame) {
    renderPanel(statusPanel, [["Status", "No data loaded"]]);
    renderPanel(framePanel, []);
    return;
  }

  renderPanel(statusPanel, [
    ["Project", bundle.project || "Depthyn"],
    ["Frames", String(bundle.frames_processed)],
    ["Tracks", String(bundle.metrics.total_tracks)],
    ["Detections", String(bundle.metrics.total_detections)],
    ["Interval", `${bundle.playback.median_frame_interval_ms} ms`],
  ]);

  renderPanel(framePanel, [
    ["Frame", `${frame.frame_index + 1} / ${bundle.frame_summaries.length}`],
    ["Frame ID", frame.frame_id],
    ["Mode", bundle.config.mode],
    ["Points", String(frame.points_after_filtering)],
    ["Foreground", String(frame.foreground_points)],
    ["Detections", String(frame.detection_count)],
    ["Active tracks", String(frame.active_tracks.length)],
    ["Stage", frame.stage],
  ]);
}

function renderPanel(node, entries) {
  node.innerHTML = "";
  for (const [label, value] of entries) {
    const wrapper = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    wrapper.appendChild(dt);
    wrapper.appendChild(dd);
    node.appendChild(wrapper);
  }
}

function setFrameIndex(index) {
  if (!state.bundle) {
    return;
  }
  state.frameIndex = Math.min(
    state.bundle.frame_summaries.length - 1,
    Math.max(0, index)
  );
  slider.value = state.frameIndex;
  updatePanels();
  render();
}

function playStep() {
  if (!state.bundle) {
    stopPlayback();
    return;
  }
  if (state.frameIndex >= state.bundle.frame_summaries.length - 1) {
    stopPlayback();
    return;
  }
  setFrameIndex(state.frameIndex + 1);
}

function startPlayback() {
  if (!state.bundle || state.playing) {
    return;
  }
  const baseInterval = state.bundle.playback.median_frame_interval_ms || 100;
  const speed = Number(speedSelect.value) || 1;
  const interval = Math.max(20, baseInterval / speed);
  state.playing = true;
  playPauseButton.textContent = "Pause";
  state.timer = window.setInterval(playStep, interval);
}

function stopPlayback() {
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
  state.playing = false;
  playPauseButton.textContent = "Play";
}

async function loadBundleFromUrl(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load replay JSON: ${response.status}`);
  }
  const bundle = await response.json();
  setBundle(bundle);
}

function bindEvents() {
  slider.addEventListener("input", () => setFrameIndex(Number(slider.value)));
  playPauseButton.addEventListener("click", () => {
    if (state.playing) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });
  stepBackButton.addEventListener("click", () => setFrameIndex(state.frameIndex - 1));
  stepForwardButton.addEventListener("click", () => setFrameIndex(state.frameIndex + 1));
  speedSelect.addEventListener("change", () => {
    if (state.playing) {
      stopPlayback();
      startPlayback();
    }
  });
  fileInput.addEventListener("change", async (event) => {
    const [file] = event.target.files;
    if (!file) {
      return;
    }
    const text = await file.text();
    const bundle = JSON.parse(text);
    setBundle(bundle);
  });
  window.addEventListener("resize", resizeCanvas);
}

async function bootstrap() {
  bindEvents();
  resizeCanvas();
  const params = new URLSearchParams(window.location.search);
  const dataUrl = params.get("data");
  if (dataUrl) {
    try {
      await loadBundleFromUrl(dataUrl);
    } catch (error) {
      renderPanel(statusPanel, [["Load error", error.message]]);
    }
  }
}

bootstrap();
