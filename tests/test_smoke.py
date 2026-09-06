"""
Fast smoke tests that don't require trained model checkpoints, a GPU, or
downloaded datasets -- safe to run in CI. Anything that needs a trained
model (evaluate.py, evaluate_rl.py, the dashboard) is exercised manually
per README.md instead.

Run:  pytest tests/test_smoke.py -v
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def test_intersection_env_runs():
    from src.signal_control.intersection_env import IntersectionEnv

    env = IntersectionEnv(max_steps=50)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    assert np.isfinite(total_reward)
    summary = env.episode_summary()
    assert "avg_total_wait_per_step" in summary


def test_dqn_agent_train_step():
    from src.signal_control.intersection_env import IntersectionEnv
    from src.signal_control.dqn_agent import DQNAgent

    env = IntersectionEnv(max_steps=20)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                      replay_capacity=500, batch_size=8)
    obs, _ = env.reset(seed=1)
    for _ in range(20):
        action = agent.act(obs, epsilon=1.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, float(terminated or truncated))
        obs = next_obs
    loss = agent.train_step()
    assert loss is None or np.isfinite(loss)


def test_fixed_time_controller():
    from src.signal_control.baseline_fixed_timer import FixedTimeController

    ctrl = FixedTimeController(n_phases=4, phase_duration=3)
    actions = [ctrl.act() for _ in range(12)]
    assert actions == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_decision_engine_emergency_override():
    from src.signal_control.dqn_agent import DQNAgent
    from src.integration.decision_engine import DecisionEngine

    agent = DQNAgent(obs_dim=17, n_actions=4, replay_capacity=10, batch_size=4)
    engine = DecisionEngine(agent)

    obs = np.zeros(17, dtype=np.float32)
    obs[0:4] = [0.1, 0.9, 0.1, 0.1]  # queues, approach 1 (East) has the longest
    obs[8:12] = [0, 1, 0, 0]         # emergency flagged on East
    action, reason = engine.decide(obs)
    assert action == 1
    assert "EMERGENCY" in reason


def test_synthetic_video_generation(tmp_path):
    from src.detection.synthetic_video import generate

    out = generate(path=str(tmp_path / "demo.mp4"), seconds=2)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_congestion_preprocess_helpers():
    import pandas as pd
    from src.congestion.preprocess import engineer_features, add_congestion_labels

    rng = pd.date_range("2020-01-01", periods=200, freq="h")
    df = pd.DataFrame({
        "date_time": rng,
        "traffic_volume": np.random.randint(500, 6000, size=len(rng)).astype(float),
        "temp": np.random.uniform(260, 300, size=len(rng)),
        "rain_1h": np.zeros(len(rng)),
        "snow_1h": np.zeros(len(rng)),
        "clouds_all": np.random.uniform(0, 100, size=len(rng)),
        "weather_main": np.random.choice(["Clear", "Clouds", "Rain"], size=len(rng)),
        "holiday": ["None"] * len(rng),
    })
    df, weather_cols = engineer_features(df)
    assert "hour_sin" in df.columns and len(weather_cols) > 0

    df, thresholds = add_congestion_labels(df)
    assert df["congestion_level"].between(0, 3).all()
    assert len(thresholds) == 3
