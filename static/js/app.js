/* ===================================================================
   Smart Traffic Light Control System — Front-End Logic
   YOLOv8s | Government Override | Continuous Simulation
   =================================================================== */

const bootstrap = window.bootstrapData || { recentCycles: [] };

const state = {
  laneCount: 4,
  latestResult: null,
  intervalId: null,
  timeoutId: null,
  simulationRunning: false,
  stopRequested: false,
  cycleCount: 0,
  overrideActive: false,
};

/* ------------------------------------------------------------------ */
/*  DOM references                                                     */
/* ------------------------------------------------------------------ */

const uploadGrid = document.getElementById("upload-grid");
const laneCountInput = document.getElementById("lane-count");
const statusText = document.getElementById("status-text");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const trafficForm = document.getElementById("traffic-form");
const laneResults = document.getElementById("lane-results");
const simulationGrid = document.getElementById("simulation-grid");
const historyList = document.getElementById("history-list");
const priorityLane = document.getElementById("priority-lane");
const cycleTotal = document.getElementById("cycle-total");
const totalPedestrians = document.getElementById("total-pedestrians");
const decisionText = document.getElementById("decision-text");
const simulationText = document.getElementById("simulation-text");
const simBadge = document.getElementById("sim-status-badge");
const simReplayBtn = document.getElementById("sim-replay-btn");
const simStopBtn = document.getElementById("sim-stop-btn");

// Override controls
const overrideBadge = document.getElementById("override-badge");
const overrideLaneSelect = document.getElementById("override-lane-select");
const overrideReason = document.getElementById("override-reason");
const activateOverrideBtn = document.getElementById("activate-override");
const clearOverrideBtn = document.getElementById("clear-override");


/* ------------------------------------------------------------------ */
/*  Lane count                                                         */
/* ------------------------------------------------------------------ */

function setLaneCount(count) {
  state.laneCount = count;
  laneCountInput.value = String(count);
  uploadGrid.dataset.count = String(count);
  simulationGrid.dataset.count = String(count);

  document.querySelectorAll(".lane-button").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.count) === count);
  });

  document.querySelectorAll(".upload-card").forEach((card, index) => {
    card.classList.toggle("hidden", index >= count);
  });
}


/* ------------------------------------------------------------------ */
/*  Status helpers                                                     */
/* ------------------------------------------------------------------ */

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.style.color = isError ? "var(--red)" : "var(--muted)";
}

function setSimBadge(mode) {
  simBadge.classList.remove("idle", "running", "stopped");
  if (mode === "running") {
    simBadge.classList.add("running");
    simBadge.textContent = "Running";
  } else if (mode === "stopped") {
    simBadge.classList.add("stopped");
    simBadge.textContent = "Stopped";
  } else {
    simBadge.classList.add("idle");
    simBadge.textContent = "Idle";
  }
}


/* ------------------------------------------------------------------ */
/*  Government Override                                                */
/* ------------------------------------------------------------------ */

function setOverrideBadge(active) {
  overrideBadge.classList.remove("active", "inactive");
  if (active) {
    overrideBadge.classList.add("active");
    overrideBadge.textContent = "ACTIVE";
  } else {
    overrideBadge.classList.add("inactive");
    overrideBadge.textContent = "Inactive";
  }
}

async function activateOverride() {
  const laneId = Number(overrideLaneSelect.value);
  const reason = overrideReason.value.trim() || "Government/Authority override";

  activateOverrideBtn.disabled = true;

  try {
    const response = await fetch("/override", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lane_id: laneId, reason }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Override failed.");

    state.overrideActive = true;
    setOverrideBadge(true);
    clearOverrideBtn.disabled = false;
    activateOverrideBtn.disabled = true;
    setStatus(`🏛️ Override active: ${data.lane_name} has top priority. Run analysis to apply.`);
  } catch (err) {
    setStatus(err.message, true);
    activateOverrideBtn.disabled = false;
  }
}

async function clearOverride() {
  clearOverrideBtn.disabled = true;

  try {
    const response = await fetch("/override/clear", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Clear failed.");

    state.overrideActive = false;
    setOverrideBadge(false);
    activateOverrideBtn.disabled = false;
    clearOverrideBtn.disabled = true;
    setStatus("Override cleared. System returned to automatic mode.");
  } catch (err) {
    setStatus(err.message, true);
    clearOverrideBtn.disabled = false;
  }
}

// Check initial override status
async function checkOverrideStatus() {
  try {
    const response = await fetch("/override/status");
    const data = await response.json();
    if (data.active) {
      state.overrideActive = true;
      setOverrideBadge(true);
      activateOverrideBtn.disabled = true;
      clearOverrideBtn.disabled = false;
    }
  } catch { /* ignore */ }
}


/* ------------------------------------------------------------------ */
/*  Video previews                                                     */
/* ------------------------------------------------------------------ */

function preparePreview(card, input) {
  const preview = card.querySelector(".preview-video");
  const fileName = card.querySelector(".file-name");

  const clear = () => {
    if (preview.dataset.objectUrl) {
      URL.revokeObjectURL(preview.dataset.objectUrl);
      delete preview.dataset.objectUrl;
    }
    preview.pause();
    preview.removeAttribute("src");
    preview.load();
    card.classList.remove("has-preview");
  };

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) {
      clear();
      fileName.textContent = "Choose a traffic video";
      return;
    }

    clear();
    const objectUrl = URL.createObjectURL(file);
    preview.dataset.objectUrl = objectUrl;
    preview.src = objectUrl;
    preview.muted = true;
    preview.defaultMuted = true;
    preview.playsInline = true;
    preview.loop = true;
    preview.autoplay = true;
    preview.preload = "auto";

    preview.addEventListener("loadeddata", () => { preview.currentTime = 0.15; }, { once: true });
    preview.addEventListener("canplay", () => {
      preview.play().catch(() => { preview.pause(); preview.currentTime = 0.15; });
    }, { once: true });
    preview.addEventListener("error", () => {
      card.classList.remove("has-preview");
      fileName.textContent = `${file.name} (preview not supported)`;
    }, { once: true });

    preview.load();
    card.classList.add("has-preview");
    fileName.textContent = file.name;
  });
}


/* ------------------------------------------------------------------ */
/*  Form data builder                                                  */
/* ------------------------------------------------------------------ */

function buildFormData() {
  const formData = new FormData();
  formData.append("lane_count", String(state.laneCount));

  for (let index = 1; index <= state.laneCount; index += 1) {
    const input = document.querySelector(`input[name="lane_${index}"]`);
    const file = input?.files?.[0];
    if (!file) throw new Error(`Lane ${index} is missing a video.`);
    formData.append(`lane_${index}`, file);
  }

  return formData;
}


/* ------------------------------------------------------------------ */
/*  Render lane results (Step 2)                                       */
/* ------------------------------------------------------------------ */

function renderLaneResults(result) {
  laneResults.classList.remove("empty");
  laneResults.innerHTML = result.lanes.map((lane) => {
    const isOverride = lane.is_override;
    const classes = [
      "lane-card",
      lane.lane_id === result.priority_lane ? "priority" : "",
      lane.emergency_detected ? "emergency" : "",
      isOverride ? "override" : "",
    ].filter(Boolean).join(" ");

    let badge = lane.density_level;
    if (isOverride) badge = "🏛️ Override";
    else if (lane.emergency_detected) badge = "🚨 Emergency";

    return `
      <article class="${classes}">
        <div class="lane-head">
          <div>
            <span class="chip">${lane.lane_name}</span>
            <h3>Signal Order #${lane.signal_order}</h3>
          </div>
          <span class="chip">${badge}</span>
        </div>
        <img src="${lane.snapshot_url}" alt="${lane.lane_name} traffic snapshot">
        <div class="meta">
          <div><span>Detector</span><strong>${lane.detector_name || "YOLOv8s"}</strong></div>
          <div><span>Vehicles (tracked)</span><strong>${lane.vehicle_count}</strong></div>
          <div><span>Pedestrians</span><strong>${lane.pedestrian_count || 0}</strong></div>
          <div><span>Average count</span><strong>${lane.average_count}</strong></div>
          <div><span>Peak frame count</span><strong>${lane.peak_count}</strong></div>
          <div><span>Occupancy</span><strong>${(lane.occupancy_ratio * 100).toFixed(1)}%</strong></div>
          <div><span>Green time</span><strong>${lane.green_time}s</strong></div>
          <div><span>Sampled frames</span><strong>${lane.sampled_frames}</strong></div>
          ${lane.emergency_detected ? `<div><span>Emergency</span><strong style="color:var(--red)">${lane.emergency_reason}</strong></div>` : ""}
          ${isOverride ? `<div><span>Override</span><strong style="color:var(--purple)">Government Priority</strong></div>` : ""}
        </div>
      </article>
    `;
  }).join("");

  priorityLane.textContent = result.priority_lane_name;
  cycleTotal.textContent = `${result.cycle_total}s`;
  totalPedestrians.textContent = String(result.total_pedestrians || 0);
  decisionText.textContent = result.decision_text;
}


/* ------------------------------------------------------------------ */
/*  Render simulation grid (Step 3)                                    */
/* ------------------------------------------------------------------ */

function renderSimulation(result) {
  simulationGrid.dataset.count = String(result.lane_count);

  let html = result.lanes.map((lane) => `
    <article class="sim-card state-red" data-lane-id="${lane.lane_id}" data-is-override="${lane.is_override || false}">
      <div class="sim-head">
        <div>
          <span class="chip">${lane.lane_name}</span>
          <h3>${lane.green_time}s green slot</h3>
        </div>
        <span class="chip" data-role="phase">Waiting</span>
      </div>
      <video class="lane-video" src="${lane.video_url}" poster="${lane.snapshot_url}" muted playsinline preload="metadata" loop></video>
      <div class="signal-area">
        <div class="light-stack">
          <span class="light red"></span>
          <span class="light yellow"></span>
          <span class="light green"></span>
        </div>
        <div class="timer">
          <strong class="timer-value" data-role="timer">--</strong>
          <span class="timer-note" data-role="note">Waiting for turn</span>
          <span class="chip">Lane #${lane.lane_id}</span>
        </div>
      </div>
    </article>
  `).join("");

  if (result.pedestrian_phase) {
    html += `
      <article class="ped-card" id="ped-crossing-card">
        <div class="ped-icon">🚶</div>
        <h3>Pedestrian Crossing</h3>
        <p>${result.pedestrian_phase.pedestrian_count} pedestrian(s) — ${result.pedestrian_phase.crossing_time}s crossing</p>
        <div class="ped-timer" data-role="ped-timer">--</div>
      </article>
    `;
  }

  simulationGrid.innerHTML = html;
}


/* ------------------------------------------------------------------ */
/*  Render history                                                     */
/* ------------------------------------------------------------------ */

function renderHistory(items) {
  if (!items.length) {
    historyList.innerHTML = `
      <article class="empty-card">
        <h3>No previous cycles</h3>
        <p>Completed runs will be logged here.</p>
      </article>
    `;
    return;
  }

  historyList.innerHTML = items.map((item) => `
    <article class="history-card">
      <div class="history-head">
        <div>
          <h3>Cycle #${item.id}</h3>
          <p>${formatTimestamp(item.created_at)}</p>
        </div>
        <span class="chip">${item.priority_lane_name}</span>
      </div>
      <div class="meta">
        <div><span>Lane count</span><strong>${item.lane_count}</strong></div>
        <div><span>Total vehicles</span><strong>${item.total_vehicles}</strong></div>
        <div><span>Cycle time</span><strong>${item.cycle_total}s</strong></div>
        <div><span>Mode</span><strong>${item.has_emergency ? "🚨 Emergency" : "Density"}</strong></div>
      </div>
      <p>${item.decision_text}</p>
    </article>
  `).join("");
}


/* ------------------------------------------------------------------ */
/*  Simulation engine — CONTINUOUS LOOP                                */
/* ------------------------------------------------------------------ */

function resetSimulationCards() {
  if (state.intervalId) { clearInterval(state.intervalId); state.intervalId = null; }
  if (state.timeoutId) { clearTimeout(state.timeoutId); state.timeoutId = null; }

  document.querySelectorAll(".sim-card").forEach((card) => {
    card.classList.remove("state-green", "state-yellow", "state-override");
    card.classList.add("state-red");
    const video = card.querySelector(".lane-video");
    if (video) video.pause();
    const phase = card.querySelector('[data-role="phase"]');
    if (phase) phase.textContent = "Red";
    const timer = card.querySelector('[data-role="timer"]');
    if (timer) timer.textContent = "--";
    const note = card.querySelector('[data-role="note"]');
    if (note) note.textContent = "Traffic stopped";
  });

  const pedCard = document.getElementById("ped-crossing-card");
  if (pedCard) {
    pedCard.classList.remove("active");
    const pedTimer = pedCard.querySelector('[data-role="ped-timer"]');
    if (pedTimer) pedTimer.textContent = "--";
  }
}


function updateSignalState(activeLaneId, phase, remaining, isOverride = false) {
  document.querySelectorAll(".sim-card").forEach((card) => {
    const laneId = Number(card.dataset.laneId);
    const video = card.querySelector(".lane-video");
    const phaseTag = card.querySelector('[data-role="phase"]');
    const timer = card.querySelector('[data-role="timer"]');
    const note = card.querySelector('[data-role="note"]');

    card.classList.remove("state-red", "state-yellow", "state-green", "state-override");

    if (laneId !== activeLaneId) {
      card.classList.add("state-red");
      if (video) video.pause();
      if (phaseTag) phaseTag.textContent = "Red";
      if (timer) timer.textContent = "--";
      if (note) note.textContent = "Traffic stopped";
      return;
    }

    if (phase === "green") {
      card.classList.add(isOverride ? "state-override" : "state-green");
      if (phaseTag) phaseTag.textContent = isOverride ? "🏛️ Override" : "Green";
      if (timer) timer.textContent = `${remaining}s`;
      if (note) note.textContent = isOverride ? "Government priority" : "Video playing";
      if (video) video.play().catch(() => {});
      return;
    }

    card.classList.add("state-yellow");
    if (phaseTag) phaseTag.textContent = "Yellow";
    if (timer) timer.textContent = `${remaining}s`;
    if (note) note.textContent = "Transition";
    if (video) video.pause();
  });

  const pedCard = document.getElementById("ped-crossing-card");
  if (pedCard) pedCard.classList.remove("active");
}


function updatePedestrianState(remaining) {
  document.querySelectorAll(".sim-card").forEach((card) => {
    card.classList.remove("state-green", "state-yellow", "state-override");
    card.classList.add("state-red");
    const video = card.querySelector(".lane-video");
    if (video) video.pause();
    const phase = card.querySelector('[data-role="phase"]');
    if (phase) phase.textContent = "Red";
    const timer = card.querySelector('[data-role="timer"]');
    if (timer) timer.textContent = "--";
    const note = card.querySelector('[data-role="note"]');
    if (note) note.textContent = "Pedestrian crossing";
  });

  const pedCard = document.getElementById("ped-crossing-card");
  if (pedCard) {
    pedCard.classList.add("active");
    const pedTimer = pedCard.querySelector('[data-role="ped-timer"]');
    if (pedTimer) pedTimer.textContent = `${remaining}s`;
  }
}


function runSimulation(result) {
  stopSimulation(false);
  resetSimulationCards();

  state.simulationRunning = true;
  state.stopRequested = false;
  state.cycleCount = 0;
  stopButton.disabled = false;
  startButton.disabled = true;
  setSimBadge("running");
  
  if (simReplayBtn) simReplayBtn.style.display = "none";
  if (simStopBtn) simStopBtn.style.display = "inline-block";

  const stages = result.signal_sequence.flatMap((lane) => ([
    { type: "lane", laneId: lane.lane_id, laneName: lane.lane_name, phase: "green", seconds: lane.green_time, isOverride: lane.is_override || false },
    { type: "lane", laneId: lane.lane_id, laneName: lane.lane_name, phase: "yellow", seconds: lane.yellow_time, isOverride: false },
  ]));

  if (result.pedestrian_phase) {
    stages.push({
      type: "pedestrian",
      seconds: result.pedestrian_phase.crossing_time,
      count: result.pedestrian_phase.pedestrian_count,
    });
  }

  const runCycle = () => {
    if (state.stopRequested) { finishSimulation(); return; }

    state.cycleCount += 1;
    let stageIndex = 0;

    const playStage = () => {
      if (state.stopRequested) { finishSimulation(); return; }

      if (stageIndex >= stages.length) {
        simulationText.textContent = `Cycle #${state.cycleCount} complete. Starting cycle #${state.cycleCount + 1}...`;
        state.timeoutId = setTimeout(runCycle, 800);
        return;
      }

      const stage = stages[stageIndex];
      let remaining = stage.seconds;

      if (stage.type === "pedestrian") {
        simulationText.textContent = `🚶 Pedestrian crossing — ${stage.count} person(s), ${stage.seconds}s. (Cycle #${state.cycleCount})`;
        updatePedestrianState(remaining);
      } else {
        const prefix = stage.isOverride ? "🏛️ " : "";
        simulationText.textContent = `${prefix}${stage.laneName} is ${stage.phase.toUpperCase()} for ${stage.seconds}s. (Cycle #${state.cycleCount})`;
        updateSignalState(stage.laneId, stage.phase, remaining, stage.isOverride);
      }

      state.intervalId = setInterval(() => {
        if (state.stopRequested) {
          clearInterval(state.intervalId);
          state.intervalId = null;
          finishSimulation();
          return;
        }
        remaining -= 1;
        if (remaining >= 0) {
          if (stage.type === "pedestrian") updatePedestrianState(remaining);
          else updateSignalState(stage.laneId, stage.phase, remaining, stage.isOverride);
        }
      }, 1000);

      state.timeoutId = setTimeout(() => {
        clearInterval(state.intervalId);
        state.intervalId = null;
        stageIndex += 1;
        playStage();
      }, stage.seconds * 1000);
    };

    playStage();
  };

  runCycle();
}


function stopSimulation(userTriggered = true) {
  state.stopRequested = true;
  state.simulationRunning = false;

  if (state.intervalId) { clearInterval(state.intervalId); state.intervalId = null; }
  if (state.timeoutId) { clearTimeout(state.timeoutId); state.timeoutId = null; }

  if (userTriggered) {
    resetSimulationCards();
    setSimBadge("stopped");
    stopButton.disabled = true;
    startButton.disabled = false;
    if (simStopBtn) simStopBtn.style.display = "none";
    if (simReplayBtn && state.latestResult) simReplayBtn.style.display = "inline-block";
    simulationText.textContent = `Simulation stopped after ${state.cycleCount} cycle(s). Press START or REPLAY to re-analyze and restart.`;
  }
}


function finishSimulation() {
  state.simulationRunning = false;
  resetSimulationCards();
  setSimBadge("stopped");
  stopButton.disabled = true;
  startButton.disabled = false;
  if (simStopBtn) simStopBtn.style.display = "none";
  if (simReplayBtn && state.latestResult) simReplayBtn.style.display = "inline-block";
  simulationText.textContent = `Simulation stopped after ${state.cycleCount} cycle(s).`;
}


/* ------------------------------------------------------------------ */
/*  Timestamp formatter                                                */
/* ------------------------------------------------------------------ */

function formatTimestamp(value) {
  const parsed = new Date(value.replace(" ", "T") + "Z");
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}


/* ------------------------------------------------------------------ */
/*  Event listeners                                                    */
/* ------------------------------------------------------------------ */

document.querySelectorAll(".lane-button").forEach((button) => {
  button.addEventListener("click", () => setLaneCount(Number(button.dataset.count)));
});

document.querySelectorAll(".upload-card").forEach((card) => {
  preparePreview(card, card.querySelector(".video-input"));
});

// START button = submit form → analyze → run simulation
trafficForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  let formData;
  try {
    formData = buildFormData();
  } catch (error) {
    setStatus(error.message, true);
    return;
  }

  if (state.simulationRunning) stopSimulation(false);

  startButton.disabled = true;
  stopButton.disabled = true;
  setSimBadge("idle");
  setStatus("Uploading videos and analyzing with YOLOv8s...");
  resetSimulationCards();

  try {
    const response = await fetch("/analyze", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Analysis failed.");

    state.latestResult = payload;
    renderLaneResults(payload);
    renderSimulation(payload);
    renderHistory(payload.history || []);
    runSimulation(payload);
    setStatus("Analysis complete. Continuous simulation running. Press STOP to halt.");
  } catch (error) {
    setStatus(error.message || "Something went wrong.", true);
    startButton.disabled = false;
  }
});

// STOP button
stopButton.addEventListener("click", () => stopSimulation(true));

// Override buttons
activateOverrideBtn.addEventListener("click", activateOverride);
clearOverrideBtn.addEventListener("click", clearOverride);

if (simStopBtn) {
  simStopBtn.addEventListener("click", () => stopSimulation(true));
}
if (simReplayBtn) {
  simReplayBtn.addEventListener("click", () => {
    if (state.latestResult) {
      runSimulation(state.latestResult);
    }
  });
}


/* ------------------------------------------------------------------ */
/*  Bootstrap                                                          */
/* ------------------------------------------------------------------ */

renderHistory(bootstrap.recentCycles || []);
setLaneCount(4);
setSimBadge("idle");
checkOverrideStatus();
