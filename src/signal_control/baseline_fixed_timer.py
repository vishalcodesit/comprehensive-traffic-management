"""Fixed-time round-robin signal controller -- the conventional baseline that
adaptive RL control is compared against (see docs/report.md Results)."""


class FixedTimeController:
    def __init__(self, n_phases=4, phase_duration=8):
        self.n_phases = n_phases
        self.phase_duration = phase_duration
        self.reset()

    def reset(self):
        self._t = 0

    def act(self, obs=None):
        phase = (self._t // self.phase_duration) % self.n_phases
        self._t += 1
        return int(phase)
