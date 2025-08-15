import torch
import torch.nn.functional as F
from .models.actor_critic import ActorCriticCNN
from .storage import RolloutBuffer

class PPO:
    def __init__(self, 
                 model=None,                      # <-- NUEVO (opcional)
                 obs_shape=(3,84,84), action_dim=3,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 lr=3e-4, n_steps=2048, batch_size=256, epochs=4,
                 gamma=0.99, gae_lambda=0.95,
                 clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        self.device = device
        self.model = model or ActorCriticCNN(obs_shape=obs_shape, action_dim=action_dim)
        self.model = self.model.to(device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.n_steps, self.batch_size, self.epochs = n_steps, batch_size, epochs
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_coef, self.vf_coef, self.ent_coef = clip_coef, vf_coef, ent_coef
        self.max_grad_norm = max_grad_norm

        # lo puedes mantener si quieres usar collect_rollout(), 
        # pero tu script usa un buffer externo
        self.buffer = RolloutBuffer(n_steps, (obs_shape[1], obs_shape[2], obs_shape[0]) if obs_shape[0] in (1,3) else obs_shape, action_dim, device)

    @torch.no_grad()
    def _obs_to_tensor(self, obs):
        t = torch.as_tensor(obs, device=self.device, dtype=torch.float32) / 255.0
        t = t.permute(2,0,1).unsqueeze(0).contiguous()
        return t

    def collect_rollout(self, env):
        """(Opcional) usa el buffer interno. Tu script no lo usa."""
        self.buffer.ptr = 0
        obs, _ = env.reset()
        for _ in range(self.n_steps):
            obs_t = self._obs_to_tensor(obs)
            action, logprob, value = self.model.act(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # OJO: firma del add aquí también debe respetar (obs, action, logprob, value, reward, done)
            self.buffer.add(obs, action_np, logprob.squeeze(0), value.squeeze(0), float(reward), done)
            obs = next_obs
            if done: obs, _ = env.reset()

        # bootstrap
        obs_t = self._obs_to_tensor(obs)
        _, _, last_value = self.model.forward(obs_t)
        self.buffer.compute_gae(last_value.squeeze(0), self.gamma, self.gae_lambda)

    def update(self, buffer=None, epochs=None, batch_size=None):
        """
        Acepta buffer externo y devuelve lista de [policy_loss, value_loss, entropy] por minibatch,
        tal como espera tu train_ppo.py.
        """
        buf = buffer if buffer is not None else self.buffer
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size

        out = []
        for _ in range(epochs):
            for obs_b, act_b, oldlog_b, adv_b, ret_b in buf.get(batch_size):
                # normalizar advantages
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                # preparar tensores
                obs_b = obs_b.float() / 255.0
                
                newlog, entropy, values = self.model.evaluate_actions(obs_b, act_b)
                ratio = (newlog - oldlog_b).exp()

                # clipped policy loss
                unclipped = -adv_b * ratio
                clipped   = -adv_b * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                pg_loss = torch.max(unclipped, clipped).mean()

                # value loss
                v_loss = F.mse_loss(values, ret_b)

                # entropy bonus
                ent = entropy.mean() if hasattr(entropy, "mean") else entropy

                loss = pg_loss + self.vf_coef*v_loss - self.ent_coef*ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

                out.append([pg_loss.item(), v_loss.item(), ent.item()])
        return out