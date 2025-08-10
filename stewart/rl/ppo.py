import torch
import torch.nn.functional as F
from .models.actor_critic import ActorCritic
from .storage import RolloutBuffer

class PPOTrainer:
    def __init__(self, obs_shape=(3,84,84), action_dim=3, device="cuda" if torch.cuda.is_available() else "cpu",
                 lr=3e-4, n_steps=2048, batch_size=256, epochs=4,
                 gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        self.device = device
        self.model = ActorCritic(obs_shape=obs_shape, action_dim=action_dim).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.n_steps, self.batch_size, self.epochs = n_steps, batch_size, epochs
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_coef, self.vf_coef, self.ent_coef = clip_coef, vf_coef, ent_coef
        self.max_grad_norm = max_grad_norm
        self.buffer = RolloutBuffer(n_steps, obs_shape, action_dim, device)

    @torch.no_grad()
    def _obs_to_tensor(self, obs):
        # obs: (H,W,3) uint8
        t = torch.as_tensor(obs, device=self.device, dtype=torch.float32) / 255.0  # (H,W,3) float
        t = t.permute(2,0,1).unsqueeze(0).contiguous()  # (1,3,H,W)
        return t

    def collect_rollout(self, env):
        self.buffer.ptr = 0
        obs, _ = env.reset()
        for _ in range(self.n_steps):
            obs_t = self._obs_to_tensor(obs)
            action, logprob, value = self.model.act(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            self.buffer.add(obs, action_np, logprob.squeeze(0), float(reward), done, value.squeeze(0))
            obs = next_obs
            if done: obs, _ = env.reset()

        # bootstrap
        obs_t = self._obs_to_tensor(obs)
        _, _, last_value = self.model.forward(obs_t)
        self.buffer.compute_gae(last_value.squeeze(0), self.gamma, self.gae_lambda)

    def update(self):
        losses = {}
        for _ in range(self.epochs):
            for obs_b, act_b, oldlog_b, adv_b, ret_b in self.buffer.get(self.batch_size):
                # normalizar advantages
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                # preparar tensores
                obs_b = obs_b.float() / 255.0
                obs_b = obs_b.permute(0,3,1,2).contiguous()  # NHWC->NCHW

                newlog, entropy, values = self.model.evaluate_actions(obs_b, act_b)
                ratio = (newlog - oldlog_b).exp()

                # clipped policy loss
                unclipped = -adv_b * ratio
                clipped   = -adv_b * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                pg_loss = torch.max(unclipped, clipped).mean()

                # value loss (clip opcional)
                v_loss = F.mse_loss(values, ret_b)

                # entropy bonus
                ent = entropy.mean()

                loss = pg_loss + self.vf_coef*v_loss - self.ent_coef*ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

                losses = {"loss":loss.item(), "pg":pg_loss.item(), "vf":v_loss.item(), "ent":ent.item()}
        return losses