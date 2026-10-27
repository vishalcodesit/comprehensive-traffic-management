"""
Live demo dashboard. Runs several things concurrently:

  1. TWO background video-processing threads (YOLOv8 + DeepSORT + emergency
     heuristics), one per feed, started once at launch so detection keeps
     feeding the signal panel whether or not a browser is watching; each
     endpoint streams its thread's latest annotated frame as MJPEG:
       - /video_feed:            a real traffic-intersection clip (see
         src/detection/download_sample_video.py) -- demonstrates that
         detection/tracking genuinely work on real footage.
       - /video_feed_emergency:  the deterministic synthetic clip with a
         strobing-light-bar "ambulance" -- demonstrates the emergency
         detection + signal-preemption feature specifically. A real clip
         can't guarantee an ambulance passes through it on cue; a COCO-
         pretrained YOLOv8 in turn can't reliably see flat-shaded synthetic
         cars, which is exactly why both feeds exist rather than one.
  2. A background "digital twin" thread that steps IntersectionEnv through
     DecisionEngine (real trained DQN + real trained LSTM forecast, cycling
     over held-out test windows) once per tick, so the signal-state panel is
     continuously live rather than computed on request. Each tick's queue
     state is NOT simulated -- it's read directly from the real feed's
     live per-approach vehicle counts (EMA-smoothed), so the signal panel
     shows the DQN reacting to what the camera actually sees right now.
  3. When the emergency-demo feed flags an emergency vehicle, it is injected
     into the digital twin's next observation so the dashboard visibly shows
     the detection actually preempting the signal -- the "integrated" part
     of the Comprehensive Traffic Management System.

Note on honesty: the real clip is pre-recorded, so choosing a green phase
here doesn't cause those cars to move (we don't control real traffic) --
the causal loop that's real is detection -> count -> decision; the loop
that isn't is decision -> real-world queue reduction, which is exactly
what IntersectionEnv's simulated discharge models for the reward/metrics
bookkeeping while the display itself follows the live detected counts.

Run:  python dashboard/app.py   (from the project root, with the venv active)
Then open http://127.0.0.1:5000
"""
import os
import sys
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.detection.video_pipeline import VideoPipeline, DEFAULT_ZONES
from src.detection.synthetic_video import generate as generate_demo_video, lane_zones
from src.detection.download_sample_video import download as download_sample_video
from src.signal_control.intersection_env import IntersectionEnv, APPROACH_NAMES
from src.signal_control.dqn_agent import DQNAgent
from src.integration.decision_engine import DecisionEngine
from src.congestion.predict import CongestionPredictor

app = Flask(__name__)

STATE_LOCK = threading.Lock()
STATE = {
    "real_feed": {
        "n_vehicles": 0,
        "zone_counts": {name: 0.0 for name in APPROACH_NAMES},   # fast EMA: signal decisions
        "lane_traffic": {name: 0.0 for name in APPROACH_NAMES},  # slow EMA: each lane's share of traffic
    },
    "emergency": {"detected": False, "zone": None, "score": 0.0},
    # Live per-lane status for each feed: {lane: {"count": int, "priority": bool}}.
    "lanes": {
        feed: {name: {"count": 0, "priority": False} for name in APPROACH_NAMES}
        for feed in ("real", "emergency_demo")
    },
    "signal": {
        "queues": {name: 0 for name in APPROACH_NAMES},
        "chosen_approach": None,
        "reason": "",
        "forecast_level": "Low",
        # LSTM forecast per horizon: [{"hours", "volume", "level", "confidence"}, ...]
        "forecast": [],
        # Per-lane forecast per horizon: [{"hours", "busiest": [lanes], "lanes": {lane: {"volume", "share", "level"}}}, ...]
        "lane_forecast": [],
        "history": [],  # list of {step, queues:{...}, chosen_approach}
    },
    "models_ready": False,
}
MAX_HISTORY = 60


# --------------------------------------------------------------------------
# Background digital-twin loop: real trained LSTM + real trained DQN.
# --------------------------------------------------------------------------
def _forecast_by_horizon(forecast):
    """Reshapes CongestionPredictor output into one row per horizon for the
    dashboard: predicted volume, level, and the model's probability for it."""
    return [
        {"hours": h, "volume": round(v), "level": lvl, "confidence": round(float(max(p)), 3)}
        for h, v, lvl, p in zip(forecast["horizons_hours"], forecast["predicted_volume"],
                                forecast["predicted_level"], forecast["level_probabilities"])
    ]


def _lane_forecast(forecast, lane_traffic, thresholds):
    """Splits the LSTM's road-level volume forecast across lanes by each
    lane's observed share of traffic. A lane's level is judged as if the
    whole road carried traffic at that lane's density (volume / share of an
    even split), so an evenly loaded road gives every lane the road's level
    and a lane carrying half the traffic is rated as twice as loaded."""
    traffic = np.array([lane_traffic[name] for name in APPROACH_NAMES], dtype=np.float64)
    shares = traffic / traffic.sum() if traffic.sum() > 0 else np.full(len(APPROACH_NAMES), 1 / len(APPROACH_NAMES))
    rows = []
    for h, total in zip(forecast["horizons_hours"], forecast["predicted_volume"]):
        lanes = {}
        for name, share in zip(APPROACH_NAMES, shares):
            density_volume = total * share * len(APPROACH_NAMES)
            level = config.CONGESTION_LEVELS[int(np.searchsorted(thresholds, density_volume, side="left"))]
            lanes[name] = {"volume": round(total * share), "share": round(float(share), 3), "level": level}
        # Every lane tied for the largest share (empty until traffic is seen).
        busiest = ([name for name, share in zip(APPROACH_NAMES, shares) if np.isclose(share, shares.max())]
                   if traffic.sum() > 0 else [])
        rows.append({"hours": h, "busiest": busiest, "lanes": lanes})
    return rows


def signal_loop():
    if not (os.path.exists(config.RL_MODEL_PATH) and os.path.exists(config.CONGESTION_MODEL_PATH)):
        print("[signal_loop] Trained models missing -- signal panel will stay idle. "
              "Run train.py / train_rl.py first.")
        return

    agent = DQNAgent.load(config.RL_MODEL_PATH)
    engine = DecisionEngine(agent)
    predictor = CongestionPredictor()
    processed = np.load(config.PROCESSED_NPZ, allow_pickle=True)
    test_data = processed["X_test"]
    level_thresholds = processed["congestion_thresholds"]  # Low/Medium/High/Severe cut points, veh/hr
    cursor = 0

    # emergency_prob=0 disables the env's own internal random test-emergency
    # spawns (a training-time feature to teach the DQN general emergency
    # response). On the live dashboard the only emergency signal should be
    # the one the video feed actually detects.
    env = IntersectionEnv(emergency_prob=0.0)
    obs, _ = env.reset(seed=int(time.time()))
    with STATE_LOCK:
        STATE["models_ready"] = True

    step = 0
    forecast_level, congestion_bias = "Low", np.zeros(4)
    while True:
        # Read the live, camera-detected vehicle count for each approach and
        # use it directly as this tick's queue state -- this is what makes
        # the signal decision respond to what the camera actually sees,
        # instead of the environment's own simulated arrival process.
        with STATE_LOCK:
            detected_counts = dict(STATE["real_feed"]["zone_counts"])
        env.queues = np.array(
            [min(config.MAX_QUEUE, detected_counts.get(name, 0.0)) for name in APPROACH_NAMES],
            dtype=np.float32,
        )
        display_queues = env.queues.copy()

        if step % 8 == 0:
            window = test_data[cursor % len(test_data)]
            cursor += 1
            forecast = predictor.predict(window)
            forecast_level = forecast["predicted_level"][0]
            horizons = _forecast_by_horizon(forecast)
            with STATE_LOCK:
                lane_traffic = dict(STATE["real_feed"]["lane_traffic"])
            lane_forecast = _lane_forecast(forecast, lane_traffic, level_thresholds)
            with STATE_LOCK:
                STATE["signal"]["forecast"] = horizons
                STATE["signal"]["lane_forecast"] = lane_forecast
            zone_counts = {name: float(q) for name, q in zip(APPROACH_NAMES, env.queues)}
            congestion_bias = engine.congestion_levels_from_forecast(forecast_level, zone_counts)

        # Mirror the video pipeline's current emergency detection into this
        # tick's observation so the controller reacts to it live -- and stops
        # preempting once the vehicle is no longer seen.
        with STATE_LOCK:
            emg = dict(STATE["emergency"])
        env.emergency[:] = 0.0
        if emg["detected"] and emg["zone"] in APPROACH_NAMES:
            idx = APPROACH_NAMES.index(emg["zone"])
            env.emergency[idx] = 1.0

        obs = env._get_obs()
        action, reason = engine.decide(obs, predicted_congestion_levels=congestion_bias)
        # env.step() still handles min-green enforcement, phase/clearance
        # bookkeeping, and reward/throughput accounting internally; its own
        # simulated arrivals get overwritten by real detected counts at the
        # top of the next iteration, so they never reach the display.
        obs, reward, terminated, truncated, info = env.step(action)

        entry = {
            "step": step,
            "queues": {name: float(q) for name, q in zip(APPROACH_NAMES, display_queues)},
            "chosen_approach": APPROACH_NAMES[action],
        }

        with STATE_LOCK:
            STATE["signal"]["queues"] = entry["queues"]
            STATE["signal"]["chosen_approach"] = entry["chosen_approach"]
            STATE["signal"]["reason"] = reason
            STATE["signal"]["forecast_level"] = forecast_level
            STATE["signal"]["history"].append(entry)
            STATE["signal"]["history"] = STATE["signal"]["history"][-MAX_HISTORY:]

        step += 1
        if terminated or truncated:
            obs, _ = env.reset(seed=int(time.time()) + step)
        time.sleep(0.7)


# --------------------------------------------------------------------------
# Video streams: real YOLOv8 + DeepSORT + emergency-heuristic pipeline, run
# by one background thread per source (each with its own tracker instance).
# HTTP clients only read the latest encoded frame, so any number of open
# tabs share the same pipeline instead of each spawning (and double-counting
# into STATE) a pipeline of its own.
# --------------------------------------------------------------------------
class FeedWorker:
    def __init__(self, prepare_source, on_summary, caption, zones=None):
        self.prepare_source = prepare_source  # returns the video path (downloads/generates if needed)
        self.on_summary = on_summary
        self.caption = caption
        self.zones = zones
        self.cond = threading.Condition()
        self.jpeg = None
        self.seq = 0

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        video_path = self.prepare_source()
        pipeline = VideoPipeline(zones=self.zones)
        frame_interval = 1.0 / 20  # don't outrun real time when the GPU is fast
        while True:  # loop the clip forever
            for annotated, summary in pipeline.frames(video_path):
                t0 = time.time()
                self.on_summary(summary)
                cv2.putText(annotated, f"{self.caption} | vehicles: {summary['n_vehicles']}",
                            (16, annotated.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.cond:
                        self.jpeg = buf.tobytes()
                        self.seq += 1
                        self.cond.notify_all()
                time.sleep(max(0.0, frame_interval - (time.time() - t0)))

    def mjpeg(self):
        last_seq = 0
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.seq != last_seq)
                jpeg, last_seq = self.jpeg, self.seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


ZONE_COUNT_EMA_ALPHA = 0.3  # smooths frame-to-frame detector jitter for display + decisions
LANE_SHARE_EMA_ALPHA = 0.005  # ~10 s memory at 20 fps: each lane's typical share, not a passing car


def _lane_status(summary):
    """Per-lane live vehicle count + priority-vehicle flag for the lane panels."""
    return {name: {"count": summary["zone_counts"][name], "priority": summary["zone_emergency"][name]}
            for name in APPROACH_NAMES}


def _on_real_summary(summary):
    with STATE_LOCK:
        STATE["real_feed"]["n_vehicles"] = summary["n_vehicles"]
        STATE["lanes"]["real"] = _lane_status(summary)
        zc = STATE["real_feed"]["zone_counts"]
        lt = STATE["real_feed"]["lane_traffic"]
        for approach, count in summary["zone_counts"].items():
            zc[approach] = (1 - ZONE_COUNT_EMA_ALPHA) * zc.get(approach, 0.0) + ZONE_COUNT_EMA_ALPHA * count
            lt[approach] = (1 - LANE_SHARE_EMA_ALPHA) * lt.get(approach, 0.0) + LANE_SHARE_EMA_ALPHA * count


def _on_emergency_summary(summary):
    zone = next((z for z, v in summary["zone_emergency"].items() if v), None)
    with STATE_LOCK:
        STATE["lanes"]["emergency_demo"] = _lane_status(summary)
        STATE["emergency"]["detected"] = summary["any_emergency"]
        STATE["emergency"]["zone"] = zone


def _as_approaches(zones):
    """Renames a 4-zone split to the intersection's N/E/S/W approaches, so the
    video overlay, lane panels and signal panel all use the same names."""
    return [(APPROACH_NAMES[i], lo, hi) for i, (_, lo, hi) in enumerate(zones)]


def _prepare_real_clip():
    if not os.path.exists(config.SAMPLE_VIDEO_PATH):
        download_sample_video()
    return config.SAMPLE_VIDEO_PATH


def _prepare_emergency_clip():
    video_path = os.path.join(config.SYNTHETIC_DATA_DIR, "demo_traffic.mp4")
    if not os.path.exists(video_path):
        generate_demo_video(path=video_path, seconds=25)
    return video_path


REAL_FEED = FeedWorker(_prepare_real_clip, _on_real_summary, "REAL FEED",
                       zones=_as_approaches(DEFAULT_ZONES))
# One zone per synthetic lane, so emergencies can preempt all four approaches.
EMERGENCY_FEED = FeedWorker(_prepare_emergency_clip, _on_emergency_summary,
                            "EMERGENCY-DETECTION DEMO", zones=_as_approaches(lane_zones()))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Real traffic-intersection footage -- genuine YOLOv8 detections."""
    return Response(REAL_FEED.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_emergency")
def video_feed_emergency():
    """Deterministic synthetic clip with a strobing-light-bar ambulance."""
    return Response(EMERGENCY_FEED.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with STATE_LOCK:
        return jsonify(STATE)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    threading.Thread(target=signal_loop, daemon=True).start()
    REAL_FEED.start()
    EMERGENCY_FEED.start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
