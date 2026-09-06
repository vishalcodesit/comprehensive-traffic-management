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

function render(state) {
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
