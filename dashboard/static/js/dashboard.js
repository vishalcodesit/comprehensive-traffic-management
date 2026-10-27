const APPROACHES = ["North", "East", "South", "West"];
const MAX_QUEUE_DISPLAY = 30;

async function poll() {
  try {
    const res = await fetch("/api/state");
    const state = await res.json();
    render(state);
  } catch (e) {
    console.error("poll failed", e);
  } finally {
    setTimeout(poll, 1000);
  }
}

function renderLanes(feed, lanes) {
  const strip = document.getElementById("lanes-" + feed);
  if (!strip || !lanes) return;
  if (!strip.children.length) {
    strip.innerHTML = APPROACHES.map((name) => `
      <div class="lane-card" data-lane="${name}">
        <span class="lane-name">${name} lane</span>
        <span class="lane-count">0</span>
        <span class="lane-unit">vehicles</span>
        <span class="lane-priority">PRIORITY VEHICLE</span>
      </div>`).join("");
  }
  APPROACHES.forEach((name) => {
    const lane = lanes[name] || { count: 0, priority: false };
    const card = strip.querySelector(`[data-lane="${name}"]`);
    card.querySelector(".lane-count").textContent = lane.count;
    card.querySelector(".lane-unit").textContent = lane.count === 1 ? "vehicle" : "vehicles";
    card.classList.toggle("priority", lane.priority);
  });
}

function renderForecast(rows) {
  const body = document.getElementById("forecast-horizons");
  if (!body || !rows || !rows.length) return;
  body.innerHTML = rows.map((r) => `
    <tr>
      <td class="horizon">+${r.hours} h</td>
      <td class="volume">${r.volume.toLocaleString()} <span class="unit">veh/hr</span></td>
      <td><span class="level-pill level-${r.level}">${r.level}</span>
          <span class="confidence">${Math.round(r.confidence * 100)}%</span></td>
    </tr>`).join("");
}

function renderLaneForecast(rows) {
  if (!rows || !rows.length) return;
  const cards = document.querySelectorAll("#busiest-lanes .busiest-card");
  rows.forEach((r, i) => {
    if (!cards[i]) return;
    const busiest = r.busiest || [];
    const lane = busiest.length ? r.lanes[busiest[0]] : null;  // tied lanes share a level
    cards[i].querySelector(".busiest-lane").textContent =
      !busiest.length ? "no traffic yet"
      : busiest.length === APPROACHES.length ? "All lanes even"
      : busiest.join(" / ");
    const levelEl = cards[i].querySelector(".busiest-level");
    levelEl.textContent = lane ? lane.level : "";
    levelEl.className = "busiest-level level-pill " + (lane ? "level-" + lane.level : "");
  });

  document.getElementById("lane-forecast").innerHTML = APPROACHES.map((name) => `
    <tr>
      <td class="horizon">${name}</td>
      ${rows.map((r) => {
        const lane = r.lanes[name];
        const busiest = (r.busiest || []).includes(name) && r.busiest.length < APPROACHES.length ? " busiest" : "";
        return `<td class="lane-cell${busiest}">
          <span class="level-pill level-${lane.level}">${lane.level}</span>
          <span class="unit">${lane.volume.toLocaleString()} veh/hr</span></td>`;
      }).join("")}
    </tr>`).join("");
}

function render(state) {
  renderForecast(state.signal && state.signal.forecast);
  renderLaneForecast(state.signal && state.signal.lane_forecast);

  Object.entries(state.lanes || {}).forEach(([feed, lanes]) => renderLanes(feed, lanes));

  const banner = document.getElementById("emergency-banner");
  const emg = state.emergency || {};
  if (emg.detected) {
    banner.classList.remove("hidden");
    document.getElementById("emergency-zone").textContent = emg.zone || "unknown";
  } else {
    banner.classList.add("hidden");
  }

  const level = (state.signal && state.signal.forecast_level) || "Low";
  const levelEl = document.getElementById("forecast-level");
  levelEl.textContent = level.toUpperCase();
  levelEl.className = "level-badge level-" + level;

  const queues = (state.signal && state.signal.queues) || {};
  const active = state.signal && state.signal.chosen_approach;
  APPROACHES.forEach((name) => {
    const el = document.getElementById("approach-" + name);
    const q = queues[name] || 0;
    const pct = Math.min(100, (q / MAX_QUEUE_DISPLAY) * 100);
    const fill = el.querySelector(".fill");
    if (["East", "West"].includes(name)) fill.style.width = pct + "%";
    else fill.style.height = pct + "%";
    el.querySelector(".count").textContent = Math.round(q);

    el.classList.toggle("active", name === active);
    el.classList.toggle("emergency", emg.detected && emg.zone === name);
  });

  const reasonEl = document.getElementById("decision-reason");
  if (state.models_ready) {
    reasonEl.textContent = (state.signal && state.signal.reason) || "";
  } else {
    reasonEl.textContent = "Trained models not found — run train.py and train_rl.py first.";
  }
}

poll();
