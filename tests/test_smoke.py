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


def test_rejected_switch_still_advances_phase_clock():
    """Repeatedly requesting a switch before min-green (as the emergency
    override does) must eventually be honoured, not deadlock."""
    from src.signal_control.intersection_env import IntersectionEnv

    env = IntersectionEnv(emergency_prob=0.0)
    env.reset(seed=0)
    env.step(1)  # switch to East: phase clock restarts
    for _ in range(env.min_green + 1):
        env.step(3)
    assert env.current_phase == 3


def test_lane_zones_cover_all_four_lanes():
    from src.detection.synthetic_video import lane_zones, LANES, WIDTH

    zones = lane_zones()
    for i, x in enumerate(LANES):
        frac = x / WIDTH
        hits = [name for name, lo, hi in zones if lo <= frac < hi]
        assert hits == [f"approach_{i + 1}"]


def test_flicker_ignores_steady_and_noisy_roofs():
    """A red car's roof drifting while staying lit, or a few noisy red
    pixels, must not count as a strobing light bar; a real strobe must."""
    from src.detection.emergency_classifier import EmergencyVehicleDetector

    det = EmergencyVehicleDetector()
    red_car = np.array([0.42, 0.35, 0.49, 0.36, 0.47, 0.33, 0.50, 0.37, 0.46, 0.34, 0.48, 0.35])
    noise = np.array([0.08, 0.02, 0.10, 0.01, 0.14, 0.00, 0.09, 0.02, 0.12, 0.01, 0.10, 0.03])
    strobe = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5])
    assert det._swings(red_car) == 0
    assert det._swings(noise) == 0
    assert det._swings(strobe) >= det.full_flicker_swings


def test_lane_overlay_marks_priority_lane():
    from src.detection.video_pipeline import draw_lane_overlay, DEFAULT_ZONES

    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    names = [name for name, _, _ in DEFAULT_ZONES]
    counts = {n: i for i, n in enumerate(names)}
    priority = {n: n == names[2] for n in names}
    draw_lane_overlay(frame, DEFAULT_ZONES, counts, priority)

    pure_red = (frame[..., 2] == 255) & (frame[..., 0] == 0) & (frame[..., 1] == 0)
    xs = np.where(pure_red)[1]
    assert xs.size and 0.5 * 854 <= xs.min() and xs.max() < 0.75 * 854  # only the priority lane's label
