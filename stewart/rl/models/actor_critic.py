import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from .vision import SmallCNN

class ActorCriticCNN(nn.Module):
    """
    Actor-Critic con acciones continuas (gaussiana diagonal).
    Mantiene tu interfaz original (action_dim, feature_dim, logstd_init).
    """
    def __init__(self, action_dim=3, feature_dim=256, logstd_init=-0.5):
        super().__init__()
        # SmallCNN ahora es independiente de HxW
        self.backbone = SmallCNN(in_channels=3, feat_dim=feature_dim)

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        # log-std global por dimensión de acción
        self.logstd = nn.Parameter(torch.full((action_dim,), float(logstd_init)))

        self.apply(self._ortho_init)

    @staticmethod
    def _ortho_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        feat = self.backbone(obs)                   # (B, feature_dim)
        mean = self.policy(feat)                    # (B, A)
        value = self.value_head(feat).squeeze(-1)   # (B,)
        std = self.logstd.exp().expand_as(mean)     # (B, A)
        dist = Independent(Normal(mean, std), 1)    # log_prob -> (B,)
        return dist, value

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        dist, value = self.forward(obs)
        action = dist.base_dist.loc if deterministic else dist.sample()
        logprob = dist.log_prob(action)             # (B,)
        return action, logprob, value

    def evaluate_actions(self, obs, actions):
        dist, value = self.forward(obs)
        logprob = dist.log_prob(actions)            # (B,)
        entropy = dist.entropy().mean()             # escalar
        return logprob, entropy, value