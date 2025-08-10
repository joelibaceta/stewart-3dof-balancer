# stewart/rl/storage.py
import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, n_steps, obs_shape, action_dim, device):
        self.n_steps = n_steps
        self.obs = np.zeros((n_steps, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        self.logprobs = np.zeros((n_steps,), dtype=np.float32)
        self.values = np.zeros((n_steps,), dtype=np.float32)
        self.rewards = np.zeros((n_steps,), dtype=np.float32)
        self.dones = np.zeros((n_steps,), dtype=np.float32)
        self.ptr = 0
        self.full = False
        self.device = device

    def add(self, obs, action, logprob, value, reward, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr] = value
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def compute_returns_adv(self, last_value, gamma=0.99, lam=0.95):
        n = self.n_steps
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * last_value * nonterminal - self.values[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = last_gae
            last_value = self.values[t]
        returns = adv + self.values
        # normalizar ventajas
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.advantages = adv
        self.returns = returns

    def get(self, batch_size):
        idxs = np.arange(self.n_steps)
        np.random.shuffle(idxs)
        for start in range(0, self.n_steps, batch_size):
            b = idxs[start:start+batch_size]
            yield (
                torch.from_numpy(self.obs[b]).to(self.device).float().div_(255.0),  # (B,H,W,3) -> NCHW lo hace la policy
                torch.from_numpy(self.actions[b]).to(self.device),
                torch.from_numpy(self.logprobs[b]).to(self.device),
                torch.from_numpy(self.advantages[b]).to(self.device),
                torch.from_numpy(self.returns[b]).to(self.device),
            )

    def reset(self):
        self.ptr = 0
        self.full = False