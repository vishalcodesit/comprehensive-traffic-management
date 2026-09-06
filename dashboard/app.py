"""
Live demo dashboard. Runs several things concurrently:

  1. TWO video-processing generators (YOLOv8 + DeepSORT + emergency
     heuristics), streamed to the browser as MJPEG:
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
     continuously live rather than computed on request.
  3. When the emergency-demo feed flags an emergency vehicle, it is injected
     into the digital twin's next observation so the dashboard visibly shows
     the detection actually preempting the signal -- the "integrated" part
     of the Comprehensive Traffic Management System.

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
from src.detection.video_pipeline import VideoPipeline
from src.detection.synthetic_video import generate as generate_demo_video
from src.detection.download_sample_video import download as download_sample_video
from src.signal_control.intersection_env import IntersectionEnv, APPROACH_NAMES
from src.signal_control.dqn_agent import DQNAgent
from src.integration.decision_engine import DecisionEngine
from src.congestion.predict import CongestionPredictor

app = Flask(__name__)

STATE_LOCK = threading.Lock()
STATE = {
    "real_feed": {"n_vehicles": 0},
    "emergency": {"detected": False, "zone": None, "score": 0.0},
    "signal": {
        "queues": {name: 0 for name in APPROACH_NAMES},
        "chosen_approach": None,
        "reason": "",
        "forecast_level": "Low",
        "predicted_volume": None,
        "history": [],  # list of {step, queues:{...}, chosen_approach}
    },
    "models_ready": False,
}
MAX_HISTORY = 60


# --------------------------------------------------------------------------
# Background digital-twin loop: real trained LSTM + real trained DQN.
# --------------------------------------------------------------------------
def signal_loop():
    if not (os.path.exists(config.RL_MODEL_PATH) and os.path.exists(config.CONGESTION_MODEL_PATH)):
        print("[signal_loop] Trained models missing -- signal panel will stay idle. "
              "Run train.py / train_rl.py first.")
        return

    agent = DQNAgent.load(config.RL_MODEL_PATH)
    engine = DecisionEngine(agent)
    predictor = CongestionPredictor()
    test_data = np.load(config.PROCESSED_NPZ, allow_pickle=True)["X_test"]
    cursor = 0

    env = IntersectionEnv()
    obs, _ = env.reset(seed=int(time.time()))
    with STATE_LOCK:
        STATE["models_ready"] = True

    step = 0
    forecast_level, congestion_bias = "Low", np.zeros(4)
    while True:
        if step % 8 == 0:
            window = test_data[cursor % len(test_data)]
            cursor += 1
            forecast = predictor.predict(window)
            forecast_level = forecast["predicted_level"][0]
            with STATE_LOCK:
                zone_counts = {name: float(q) for name, q in zip(APPROACH_NAMES, env.queues)}
            congestion_bias = engine.congestion_levels_from_forecast(forecast_level, zone_counts)

        # If the video pipeline just spotted an emergency vehicle, inject it
        # into this tick's observation so the controller reacts to it live.
        with STATE_LOCK:
            emg = STATE["emergency"]
        if emg["detected"] and emg["zone"] in APPROACH_NAMES:
            idx = APPROACH_NAMES.index(emg["zone"])
            env.emergency[idx] = 1.0
            obs = env._get_obs()  # refresh obs so this tick's decision sees the injection immediately

        action, reason = engine.decide(obs, predicted_congestion_levels=congestion_bias)
        obs, reward, terminated, truncated, info = env.step(action)

        entry = {
            "step": step,
            "queues": {name: float(q) for name, q in zip(APPROACH_NAMES, info["queues"])},
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
# Video streams: real YOLOv8 + DeepSORT + emergency-heuristic pipeline,
# run independently on two sources (each gets its own tracker instance).
# --------------------------------------------------------------------------
def _mjpeg_generator(video_path, on_summary, caption):
    pipeline = VideoPipeline()
    while True:  # loop the clip forever
        for annotated, summary in pipeline.frames(video_path):
            on_summary(summary)
            cv2.putText(annotated, f"{caption} | vehicles: {summary['n_vehicles']}",
                        (16, annotated.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


def _on_real_summary(summary):
    with STATE_LOCK:
        STATE["real_feed"]["n_vehicles"] = summary["n_vehicles"]


def _on_emergency_summary(summary):
    zone = next((z for z, v in summary["zone_emergency"].items() if v), None)
    with STATE_LOCK:
        STATE["emergency"]["detected"] = summary["any_emergency"]
        STATE["emergency"]["zone"] = _zone_to_approach(zone)


def _zone_to_approach(zone_name):
    """Maps video_pipeline's approach_1..4 zones onto the intersection's N/E/S/W naming."""
    if zone_name is None:
        return None
    mapping = {"approach_1": "North", "approach_2": "East", "approach_3": "South", "approach_4": "West"}
    return mapping.get(zone_name, zone_name)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Real traffic-intersection footage -- genuine YOLOv8 detections."""
    video_path = config.SAMPLE_VIDEO_PATH
    if not os.path.exists(video_path):
        download_sample_video()
    return Response(_mjpeg_generator(video_path, _on_real_summary, "REAL FEED"),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_emergency")
def video_feed_emergency():
    """Deterministic synthetic clip with a strobing-light-bar ambulance."""
    video_path = os.path.join(config.SYNTHETIC_DATA_DIR, "demo_traffic.mp4")
    if not os.path.exists(video_path):
        generate_demo_video(path=video_path, seconds=25)
    return Response(_mjpeg_generator(video_path, _on_emergency_summary, "EMERGENCY-DETECTION DEMO"),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with STATE_LOCK:
        return jsonify(STATE)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    threading.Thread(target=signal_loop, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
