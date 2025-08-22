import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from stewart.models.vision_cnn import VisionCNN

class ActorCriticCNN(nn.Module):
    EPS = 1e-6

    def __init__(self, action_dim=3, feature_dim=256, logstd_init=-0.5, obs_shape=(12,84,84)):
        super().__init__()
        C,H,W = obs_shape
        # TODO: tu CNN de visión; aquí un placeholder:
        self.encoder = VisionCNN(in_channels=C, out_dim=feature_dim)

        # Actor
        self.mu = nn.Linear(feature_dim, action_dim)
        self.logstd = nn.Parameter(torch.ones(action_dim) * float(logstd_init))

        # Crítico
        self.v = nn.Linear(feature_dim, 1)

    def forward(self, obs):
        """
        obs: (B,C,H,W) en [0,1] (float32)
        Returns:
          mu: (B,A)
          logstd: (A,)  # broadcast por batch
          value: (B,1)
        """
        feat = self.encoder(obs)               # (B,feature_dim)
        mu = self.mu(feat)                     # (B,A)
        value = self.v(feat)                   # (B,1)
        logstd = self.logstd.clamp(-3.0, 0.5)  # evita std extremas
        return mu, logstd, value

    @staticmethod
    def _atanh(x, eps=1e-6):
        # clip para evitar NaNs en atanh
        x = x.clamp(-1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def act(self, obs):
        """
        Obs: (B,C,H,W) float en [0,1].
        Devuelve:
          action: (B,A) en [-1,1]
          logprob: (B,1)  log-prob corregido por tanh
          value: (B,1)
        """
        mu, logstd, value = self.forward(obs)
        std = logstd.exp()
        base = Normal(mu, std)
        u = base.rsample()              # reparam
        a = torch.tanh(u)

        # log|det J| = sum log(1 - tanh(u)^2) con forma estable
        # log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))
        log_det = torch.sum(2.0 * (torch.log(torch.tensor(2.0)) - u - F.softplus(-2.0*u)), dim=-1, keepdim=True)
        logprob = base.log_prob(u).sum(dim=-1, keepdim=True) - log_det
        return a, logprob, value

    def evaluate_actions(self, obs, actions):
        """
        Evalúa log-prob y valor para acciones YA squashed en [-1,1].
        """
        actions = actions.clamp(-1 + self.EPS, 1 - self.EPS)
        mu, logstd, value = self.forward(obs)
        std = logstd.exp()
        base = Normal(mu, std)

        # Inversa estable de tanh
        u = self._atanh(actions, eps=self.EPS)

        # Mismo jacobiano que en act()
        log_det = torch.sum(2.0 * (torch.log(torch.tensor(2.0)) - u - F.softplus(-2.0*u)), dim=-1, keepdim=True)
        logprob = base.log_prob(u).sum(dim=-1, keepdim=True) - log_det

        # Entropía: usar la de la base como aproximación (estándar en PPO-tanh)
        entropy = base.entropy().sum(dim=-1, keepdim=True)
        return logprob, entropy, value