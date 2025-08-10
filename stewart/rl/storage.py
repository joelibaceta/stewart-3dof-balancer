# stewart/rl/storage.py
import numpy as np
import torch

class RolloutBuffer:
    """
    Buffer para PPO:
      obs:      (T, H, W, C)  uint8  [NHWC]
      actions:  (T, A)        float32
      logprobs: (T,)          float32
      values:   (T,)          float32
      rewards:  (T,)          float32
      dones:    (T,)          bool

    compute_gae(last_value, gamma, lam) calcula:
      advantages: (T,) float32
      returns:    (T,) float32
    """

    def __init__(self, n_steps, obs_shape, action_dim, device):
        self.n = int(n_steps)
        self.device = device

        # Esperado por tu pipeline: obs en NHWC uint8
        self.obs      = np.zeros((self.n, *obs_shape), dtype=np.uint8)
        self.actions  = np.zeros((self.n, action_dim), dtype=np.float32)
        self.logprobs = np.zeros((self.n,), dtype=np.float32)
        self.values   = np.zeros((self.n,), dtype=np.float32)
        self.rewards  = np.zeros((self.n,), dtype=np.float32)
        self.dones    = np.zeros((self.n,), dtype=np.bool_)

        # Se llenan en compute_gae / compute_returns_adv
        self.advantages = np.zeros((self.n,), dtype=np.float32)
        self.returns    = np.zeros((self.n,), dtype=np.float32)

        self.ptr = 0  # cuántos pasos válidos hay escritos [0..n]

    # ---------- escritura ----------
    def reset(self):
        """Reinicia el puntero para un nuevo rollout (no realoca memoria)."""
        self.ptr = 0

    def add(self, obs, action, logprob, value, reward, done):
        """
        Firma alineada a tu train_ppo.py:
        add(obs, action_np, logp.item(), value.item(), reward, done)
        """
        i = self.ptr
        if i >= self.n:
            raise RuntimeError(f"RolloutBuffer lleno: ptr={self.ptr}, n={self.n}")
        self.obs[i]      = obs
        self.actions[i]  = action
        self.logprobs[i] = float(logprob)
        self.values[i]   = float(value)
        self.rewards[i]  = float(reward)
        self.dones[i]    = bool(done)
        self.ptr += 1

    # ---------- ventajas / retornos ----------
    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        """
        last_value: V(s_T) del estado siguiente al último paso guardado.
        Soporta buffers parciales usando self.ptr como longitud efectiva.
        """
        L = self.ptr
        if L == 0:
            raise RuntimeError("compute_gae llamado sin datos (ptr=0).")

        adv = 0.0
        for t in reversed(range(L)):
            next_value = float(last_value) if t == L - 1 else self.values[t + 1]
            nonterminal = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            adv = delta + gamma * lam * nonterminal * adv
            self.advantages[t] = adv

        self.returns[:L] = self.advantages[:L] + self.values[:L]

    # Alias que usa tu script:
    def compute_returns_adv(self, last_value, gamma=0.99, lam=0.95):
        self.compute_gae(last_value, gamma, lam)

    # ---------- iteración en mini-batches ----------
    def get(self, batch_size):
        """
        Itera en mini-batches barajados sobre los primeros self.ptr pasos.
        Devuelve tensores ya en self.device:
          obs_b   : (B, H, W, C) uint8
          act_b   : (B, A)       float32
          oldlog_b: (B,)         float32
          adv_b   : (B,)         float32
          ret_b   : (B,)         float32
        """
        L = self.ptr
        if L == 0:
            raise RuntimeError("get() llamado sin datos (ptr=0).")

        idxs = np.random.permutation(L)
        for start in range(0, L, batch_size):
            j = idxs[start:start + batch_size]
            yield (
                torch.from_numpy(self.obs[j]).to(self.device),
                torch.from_numpy(self.actions[j]).to(self.device),
                torch.from_numpy(self.logprobs[j]).to(self.device),
                torch.from_numpy(self.advantages[j]).to(self.device),
                torch.from_numpy(self.returns[j]).to(self.device),
            )

    # ---------- utilidades ----------
    def is_full(self):
        return self.ptr >= self.n

    def size(self):
        return self.ptr

    def __len__(self):
        return self.ptr