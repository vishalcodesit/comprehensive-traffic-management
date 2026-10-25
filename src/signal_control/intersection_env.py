"""
A compact, self-contained 4-approach intersection simulator exposed as a
Gymnasium environment. This intentionally does not depend on SUMO (which
needs a separate, heavyweight native install) -- it models the two things
that actually drive the reward: per-approach queueing dynamics and
emergency-vehicle presence, which is exactly what src/detection feeds it
in the integrated pipeline.

Action space: Discrete(4) -- which single approach (N=0, E=1, S=2, W=3)
gets the green light. A minimum green duration is enforced; requesting a
switch before it elapses is ignored (and mildly penalized) rather than
crashing, matching real signal-controller safety interlocks.

Observation: for each of the 4 approaches -- normalized queue length,
normalized recent arrival rate, emergency-vehicle-waiting flag -- plus a
one-hot of the current green phase and normalized time-in-phase.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

APPROACH_NAMES = ["North", "East", "South", "West"]


class IntersectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_queue=config.MAX_QUEUE, min_green=config.MIN_GREEN_STEPS,
                 max_steps=config.RL_MAX_STEPS, discharge_rate=3,
                 emergency_prob=0.01, clearance_steps=1):
        super().__init__()
        self.n_approaches = config.NUM_APPROACHES
        self.max_queue = max_queue
        self.min_green = min_green
        self.max_steps = max_steps
        self.discharge_rate = discharge_rate
        self.emergency_prob = emergency_prob
        self.clearance_steps = clearance_steps

        self.action_space = spaces.Discrete(self.n_approaches)
        obs_dim = self.n_approaches * 3 + self.n_approaches + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng()

    def _sample_arrival_rates(self):
        # Each episode gets a fresh mix of light/moderate/heavy approaches so
        # the policy has to generalize across traffic regimes, not memorize one.
        base = self._rng.uniform(0.15, 0.9, size=self.n_approaches)
        return base

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.queues = np.zeros(self.n_approaches, dtype=np.float32)
        self.arrival_rates = self._sample_arrival_rates()
        self.emergency = np.zeros(self.n_approaches, dtype=np.float32)
        self.emergency_wait = np.zeros(self.n_approaches, dtype=np.float32)
        self.current_phase = 0
        self.time_in_phase = 0
        self.clearance_remaining = 0
        self.step_count = 0
        self.cumulative_wait = 0.0
        self.cumulative_throughput = 0
        self.cumulative_emergency_wait = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        q_norm = np.clip(self.queues / self.max_queue, 0, 1)
        rate_norm = self.arrival_rates
        emg = self.emergency
        phase_onehot = np.zeros(self.n_approaches, dtype=np.float32)
        phase_onehot[self.current_phase] = 1.0
        time_norm = min(self.time_in_phase / (self.min_green * 3), 1.0)
        obs = np.concatenate([q_norm, rate_norm, emg, phase_onehot, [time_norm]]).astype(np.float32)
        return obs

    def step(self, action):
        action = int(action)
        invalid_switch = False

        if action != self.current_phase:
            if self.time_in_phase < self.min_green:
                # Ignored: keep current phase. The phase clock must still
                # advance, otherwise a controller that keeps requesting the
                # switch (e.g. the emergency override) deadlocks forever.
                invalid_switch = True
                self.time_in_phase += 1
            else:
                self.current_phase = action
                self.time_in_phase = 0
                self.clearance_remaining = self.clearance_steps
        else:
            self.time_in_phase += 1

        # Discharge vehicles from the green approach (unless in clearance/yellow).
        served = 0
        if self.clearance_remaining > 0:
            self.clearance_remaining -= 1
        else:
            served = min(self.discharge_rate, self.queues[self.current_phase])
            self.queues[self.current_phase] -= served
            if self.emergency[self.current_phase] > 0 and served > 0:
                self.emergency[self.current_phase] = 0.0
                self.emergency_wait[self.current_phase] = 0.0
        self.cumulative_throughput += served

        # New arrivals (Poisson) at every approach.
        arrivals = self._rng.poisson(self.arrival_rates)
        self.queues = np.clip(self.queues + arrivals, 0, self.max_queue)

        # Randomly spawn an emergency vehicle on a non-empty, non-flagged approach.
        if self._rng.random() < self.emergency_prob:
            candidates = [i for i in range(self.n_approaches) if self.emergency[i] == 0]
            if candidates:
                self.emergency[self._rng.choice(candidates)] = 1.0

        self.emergency_wait += self.emergency  # accrues only while flag is set

        # --- Reward ---
        total_queue = float(np.sum(self.queues))
        overflow_penalty = 5.0 * float(np.sum(self.queues >= self.max_queue))
        emergency_penalty = 4.0 * float(np.sum(self.emergency))  # penalize every step an ambulance waits
        switch_penalty = 0.5 if invalid_switch else 0.0

        reward = -(0.1 * total_queue) - emergency_penalty - overflow_penalty - switch_penalty
        self.cumulative_wait += total_queue
        self.cumulative_emergency_wait += float(np.sum(self.emergency))

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "queues": self.queues.copy(),
            "emergency": self.emergency.copy(),
            "served": served,
            "invalid_switch": invalid_switch,
            "avg_queue": total_queue / self.n_approaches,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def episode_summary(self):
        return {
            "avg_total_wait_per_step": self.cumulative_wait / max(self.step_count, 1),
            "throughput": self.cumulative_throughput,
            "avg_emergency_wait_steps": self.cumulative_emergency_wait / max(self.step_count, 1),
        }
