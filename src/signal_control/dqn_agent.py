"""Deep Q-Network agent (MLP Q-function + replay buffer + target network)."""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = map(np.array, zip(*batch))
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, gamma=0.95, lr=5e-4,
                 replay_capacity=20000, batch_size=64, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(replay_capacity)

    def act(self, obs, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            q = self.q_net(x)
            return int(q.argmax(dim=-1).item())

    def q_values(self, obs):
        with torch.no_grad():
            x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            return self.q_net(x).cpu().numpy()[0]

    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        s = torch.from_numpy(s.astype(np.float32)).to(self.device)
        a = torch.from_numpy(a.astype(np.int64)).unsqueeze(1).to(self.device)
        r = torch.from_numpy(r.astype(np.float32)).to(self.device)
        s2 = torch.from_numpy(s2.astype(np.float32)).to(self.device)
        done = torch.from_numpy(done.astype(np.float32)).to(self.device)

        q_pred = self.q_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s2).max(dim=1)[0]
            q_target = r + self.gamma * q_next * (1 - done)

        loss = F.smooth_l1_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save({"q_state": self.q_net.state_dict(),
                    "obs_dim": self.q_net.net[0].in_features,
                    "n_actions": self.n_actions}, path)

    @classmethod
    def load(cls, path, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], device=device)
        agent.q_net.load_state_dict(ckpt["q_state"])
        agent.q_net.eval()
        return agent
