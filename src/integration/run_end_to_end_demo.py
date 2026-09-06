"""
Full pipeline demo: the real trained LSTM congestion forecaster and the real
trained DQN signal-control policy, wired together through DecisionEngine,
controlling the IntersectionEnv simulator (used here as the "physical world"
-- ground-truth queues and emergency-vehicle spawns -- exactly as loop
detectors + the video pipeline would supply in a live deployment).

Every few steps we pull a fresh forecast from the LSTM on a real historical
traffic window (cycling through the held-out test split) to demonstrate the
"predict congestion before it occurs" objective actually influencing the
signal-control decision, not just running in isolation.

Run:  python -m src.integration.run_end_to_end_demo
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.signal_control.intersection_env import IntersectionEnv, APPROACH_NAMES
from src.signal_control.dqn_agent import DQNAgent
from src.integration.decision_engine import DecisionEngine
from src.congestion.predict import CongestionPredictor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(n_steps=200, forecast_every=10, seed=42):
    if not os.path.exists(config.RL_MODEL_PATH):
        print(f"Missing trained DQN at {config.RL_MODEL_PATH}. Run: python -m src.signal_control.train_rl")
        return
    if not os.path.exists(config.CONGESTION_MODEL_PATH):
        print(f"Missing trained LSTM at {config.CONGESTION_MODEL_PATH}. Run: python -m src.congestion.train")
        return

    agent = DQNAgent.load(config.RL_MODEL_PATH)
    engine = DecisionEngine(agent)
    predictor = CongestionPredictor()

    test_data = np.load(config.PROCESSED_NPZ, allow_pickle=True)["X_test"]
    cursor = 0

    env = IntersectionEnv()
    obs, _ = env.reset(seed=seed)

    trace = []
    forecast_level = "Low"
    congestion_bias = np.zeros(4)

    for step in range(n_steps):
        if step % forecast_every == 0:
            window = test_data[cursor % len(test_data)]
            cursor += 1
            forecast = predictor.predict(window)
            forecast_level = forecast["predicted_level"][0]  # nearest horizon
            zone_counts = {name: float(q) for name, q in zip(APPROACH_NAMES, env.queues)}
            congestion_bias = engine.congestion_levels_from_forecast(forecast_level, zone_counts)

        action, reason = engine.decide(obs, predicted_congestion_levels=congestion_bias)
        obs, reward, terminated, truncated, info = env.step(action)

        trace.append({
            "step": step,
            "chosen_approach": APPROACH_NAMES[action],
            "reason": reason,
            "queues": info["queues"].tolist(),
            "emergency": info["emergency"].tolist(),
            "forecast_level": forecast_level,
            "reward": reward,
        })
        if info["emergency"].sum() > 0 or "EMERGENCY" in reason:
            print(f"[step {step:3d}] forecast={forecast_level:8s} | {reason}")

        if terminated or truncated:
            break

    summary = env.episode_summary()
    print("\nEnd-to-end demo summary:")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(config.RESULTS_DIR, "end_to_end_demo_trace.json"), "w") as f:
        json.dump({"summary": summary, "trace": trace}, f, indent=2)

    queues = np.array([t["queues"] for t in trace])
    emergencies = np.array([t["emergency"] for t in trace])
    chosen = [APPROACH_NAMES.index(t["chosen_approach"]) for t in trace]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    for i, name in enumerate(APPROACH_NAMES):
        axes[0].plot(queues[:, i], label=f"{name} queue")
        emg_steps = np.where(emergencies[:, i] > 0)[0]
        if len(emg_steps):
            axes[0].scatter(emg_steps, queues[emg_steps, i], color="red", marker="*",
                             s=80, zorder=5, label=f"{name} emergency" if i == 0 else None)
    axes[0].set_ylabel("Queue length (vehicles)")
    axes[0].set_title("End-to-end demo: LSTM forecast + emergency override + DQN signal control")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    axes[1].step(range(len(chosen)), chosen, where="post", color="tab:purple")
    axes[1].set_yticks(range(4)); axes[1].set_yticklabels(APPROACH_NAMES)
    axes[1].set_ylabel("Green phase")
    axes[1].set_xlabel("simulation step")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "end_to_end_demo_trace.png"), dpi=140)
    print("Saved trace plot -> results/end_to_end_demo_trace.png")


if __name__ == "__main__":
    run()
