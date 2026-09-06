"""
Trains the DQN adaptive traffic-signal controller against the intersection
simulator, with emergency-vehicle presence baked into both the state and the
reward so the policy learns to preempt for ambulances, not just minimize
average queue length.

Run:  python -m src.signal_control.train_rl
"""
import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.signal_control.intersection_env import IntersectionEnv
from src.signal_control.dqn_agent import DQNAgent

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def run(episodes=config.RL_EPISODES, eps_start=1.0, eps_end=0.05, eps_decay_frac=0.7):
    env = IntersectionEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_dim, n_actions, gamma=config.RL_GAMMA, lr=config.RL_LR,
                      replay_capacity=config.RL_REPLAY_CAPACITY, batch_size=config.RL_BATCH_SIZE)

    decay_episodes = max(1, int(episodes * eps_decay_frac))
    history = {"episode_reward": [], "avg_queue": [], "avg_emergency_wait": [], "epsilon": []}

    t0 = time.time()
    for ep in range(1, episodes + 1):
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * (ep / decay_episodes))
        obs, _ = env.reset(seed=None)
        ep_reward = 0.0
        losses = []

        for _ in range(config.RL_MAX_STEPS):
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, reward, next_obs, float(done))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            ep_reward += reward
            if done:
                break

        if ep % config.RL_TARGET_UPDATE_EVERY == 0:
            agent.update_target()

        summary = env.episode_summary()
        history["episode_reward"].append(ep_reward)
        history["avg_queue"].append(summary["avg_total_wait_per_step"])
        history["avg_emergency_wait"].append(summary["avg_emergency_wait_steps"])
        history["epsilon"].append(epsilon)

        if ep % 20 == 0 or ep == episodes:
            avg_r = np.mean(history["episode_reward"][-20:])
            avg_q = np.mean(history["avg_queue"][-20:])
            print(f"Episode {ep:4d}/{episodes}  eps={epsilon:.3f}  "
                  f"avg_reward(20)={avg_r:.1f}  avg_queue(20)={avg_q:.2f}  "
                  f"buffer={len(agent.buffer)}")

    elapsed = time.time() - t0
    print(f"RL training complete in {elapsed:.1f}s.")

    os.makedirs(os.path.dirname(config.RL_MODEL_PATH), exist_ok=True)
    agent.save(config.RL_MODEL_PATH)
    print(f"Saved DQN policy -> {config.RL_MODEL_PATH}")

    with open(os.path.join(config.RESULTS_DIR, "rl_train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if HAS_PLT:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        window = 15
        smoothed_r = np.convolve(history["episode_reward"], np.ones(window) / window, mode="valid")
        ax[0].plot(smoothed_r); ax[0].set_title(f"Episode reward ({window}-ep moving avg)")
        ax[0].set_xlabel("episode")
        ax[1].plot(history["avg_queue"]); ax[1].set_title("Avg total queue / step")
        ax[1].set_xlabel("episode")
        ax[2].plot(history["avg_emergency_wait"], color="tab:red")
        ax[2].set_title("Avg emergency-vehicle wait / step")
        ax[2].set_xlabel("episode")
        fig.tight_layout()
        fig.savefig(os.path.join(config.RESULTS_DIR, "rl_training_curves.png"), dpi=140)
        print("Saved RL training curves plot.")

    return agent


if __name__ == "__main__":
    run()
