"""
The "comprehensive" part of the project: fuses the three trained/heuristic
ML components into one signal-control decision per tick.

  1. Priority Vehicle Detection (src/detection) -> a hard emergency flag per
     approach. If any approach has a waiting emergency vehicle, it gets the
     green unconditionally -- this is a safety interlock, not something we
     leave to a learned policy's discretion (matches the "few approaches
     incorporate real-time... automatic prioritization" gap called out in
     docs/report.md).

  2. Traffic Congestion Prediction (src/congestion) -> a forecast congestion
     level (Low..Severe) per approach for the near future. This *soft*-biases
     the DQN's Q-values towards approaches that are about to get busy, so the
     controller starts clearing a queue proactively before it actually forms,
     rather than reacting only after it's built up.

  3. Adaptive Signal Control (src/signal_control) -> the learned DQN policy
     supplies the base Q-values that steps 1-2 override/adjust.

`IntersectionEnv`'s observation layout is reused directly as the interface
contract (see intersection_env.py): [queue(4), arrival_rate(4), emergency(4),
phase_onehot(4), time_in_phase(1)].
"""
import numpy as np


class DecisionEngine:
    def __init__(self, dqn_agent, congestion_bias_weight=1.5):
        self.agent = dqn_agent
        self.congestion_bias_weight = congestion_bias_weight
        self.n_approaches = 4

    def decide(self, obs: np.ndarray, predicted_congestion_levels=None):
        """
        obs: 17-dim vector matching IntersectionEnv's observation space.
        predicted_congestion_levels: optional length-4 array, values in
            {0,1,2,3} = {Low,Medium,High,Severe}, one per approach, typically
            produced by broadcasting the LSTM's forecast across approaches
            and weighting by each approach's current share of traffic.
        Returns: (action:int, reason:str)
        """
        emergency = obs[8:12]
        if emergency.sum() > 0:
            queues = obs[0:4]
            candidates = np.where(emergency > 0)[0]
            action = int(candidates[np.argmax(queues[candidates])])
            return action, f"EMERGENCY override -> approach {action}"

        q_values = self.agent.q_values(obs)

        if predicted_congestion_levels is not None:
            bias = self.congestion_bias_weight * (np.asarray(predicted_congestion_levels) / 3.0)
            adjusted = q_values + bias
            action = int(np.argmax(adjusted))
            return action, (f"DQN+congestion-bias -> approach {action} "
                             f"(Q={q_values.round(2)}, bias={bias.round(2)})")

        action = int(np.argmax(q_values))
        return action, f"DQN policy -> approach {action} (Q={q_values.round(2)})"

    def congestion_levels_from_forecast(self, predicted_level_name, zone_counts):
        """Broadcasts a single global LSTM forecast (e.g. 'High') across the 4
        approaches, weighted by each approach's current share of detected
        vehicles, so a globally-busy forecast still respects which specific
        approach is actually loaded right now."""
        from config import CONGESTION_LEVELS
        base_level = CONGESTION_LEVELS.index(predicted_level_name)
        counts = np.array(list(zone_counts.values()), dtype=np.float32)
        if counts.sum() == 0:
            return np.full(self.n_approaches, base_level, dtype=np.float32)
        share = counts / counts.sum()
        # Scale so the busiest approach gets the full forecast level and
        # quieter approaches get proportionally less -- avoids over-biasing
        # an approach the cameras show as empty just because the corridor
        # as a whole is forecast to be congested.
        return np.clip(base_level * share * self.n_approaches, 0, 3)
