# stewart/rl/models/actor_critic.py
import torch
import torch.nn as nn
from .vision import SmallVision
from torch.distributions import Normal

class ActorCriticCNN(nn.Module):
    def __init__(self, action_dim=3, feature_dim=256, logstd_init=-0.5):
        super().__init__()
        self.backbone = SmallVision(in_channels=3, out_dim=feature_dim)
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        # log-std global por acción (continua)
        self.logstd = nn.Parameter(torch.ones(action_dim) * logstd_init)

    def forward(self, obs):
        feat = self.backbone(obs)
        mean = self.policy(feat)
        value = self.value_head(feat).squeeze(-1)
        std = self.logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist, value

    @torch.no_grad()
    def act(self, obs):
        dist, value = self.forward(obs)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        return action, logprob, value