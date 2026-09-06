"""
Compares the trained DQN adaptive controller against the conventional
fixed-time baseline on identical evaluation episodes (same random seeds ->
same arrival patterns and emergency-vehicle spawns), which is the head-line
result of the signal-control component.

Run:  python -m src.signal_control.evaluate_rl
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.signal_control.intersection_env import IntersectionEnv
from src.signal_control.dqn_agent import DQNAgent
from src.signal_control.baseline_fixed_timer import FixedTimeController

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_policy(policy_fn, reset_fn, n_episodes=30, seed_base=1000):
    metrics = {"avg_queue": [], "throughput": [], "avg_emergency_wait": []}
    for i in range(n_episodes):
        seed = seed_base + i
        env = IntersectionEnv()
        obs, _ = env.reset(seed=seed)
        reset_fn()
        for _ in range(config.RL_MAX_STEPS):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        summary = env.episode_summary()
        metrics["avg_queue"].append(summary["avg_total_wait_per_step"])
        metrics["throughput"].append(summary["throughput"])
        metrics["avg_emergency_wait"].append(summary["avg_emergency_wait_steps"])
    return {k: float(np.mean(v)) for k, v in metrics.items()}, metrics


def run(n_episodes=30):
    env_probe = IntersectionEnv()
    obs_dim = env_probe.observation_space.shape[0]
    n_actions = env_probe.action_space.n

    if os.path.exists(config.RL_MODEL_PATH):
        agent = DQNAgent.load(config.RL_MODEL_PATH)
    else:
        print(f"No trained DQN found at {config.RL_MODEL_PATH}. Run train_rl.py first.")
        return

    dqn_policy = lambda obs: agent.act(obs, epsilon=0.0)
    dqn_mean, dqn_raw = run_policy(dqn_policy, lambda: None, n_episodes=n_episodes)

    fixed = FixedTimeController(n_phases=n_actions, phase_duration=config.MIN_GREEN_STEPS * 2)
    fixed_policy = lambda obs: fixed.act(obs)
    fixed_mean, fixed_raw = run_policy(fixed_policy, fixed.reset, n_episodes=n_episodes)

    results = {"dqn_adaptive": dqn_mean, "fixed_time_baseline": fixed_mean}
    improvement = {
        "queue_reduction_%": 100 * (fixed_mean["avg_queue"] - dqn_mean["avg_queue"]) / fixed_mean["avg_queue"],
        "throughput_gain_%": 100 * (dqn_mean["throughput"] - fixed_mean["throughput"]) / max(fixed_mean["throughput"], 1e-6),
        "emergency_wait_reduction_%": 100 * (fixed_mean["avg_emergency_wait"] - dqn_mean["avg_emergency_wait"]) / max(fixed_mean["avg_emergency_wait"], 1e-6),
    }
    results["improvement_of_dqn_over_baseline"] = improvement

    print(json.dumps(results, indent=2))
    with open(os.path.join(config.RESULTS_DIR, "rl_evaluation_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    labels = ["Avg queue/step", "Throughput", "Avg emergency wait/step"]
    dqn_vals = [dqn_mean["avg_queue"], dqn_mean["throughput"], dqn_mean["avg_emergency_wait"]]
    fixed_vals = [fixed_mean["avg_queue"], fixed_mean["throughput"], fixed_mean["avg_emergency_wait"]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, (label, dv, fv) in enumerate(zip(labels, dqn_vals, fixed_vals)):
        axes[i].bar(["DQN adaptive", "Fixed-time"], [dv, fv], color=["tab:green", "tab:gray"])
        axes[i].set_title(label)
    fig.suptitle(f"Adaptive RL vs. fixed-time signal control ({n_episodes} eval episodes)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS_DIR, "rl_vs_baseline_comparison.png"), dpi=140)
    print("Saved comparison plot to results/rl_vs_baseline_comparison.png")


if __name__ == "__main__":
    run()
