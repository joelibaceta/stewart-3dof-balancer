import torch
import torch.nn.functional as F
from stewart.models.actor_critic_cnn import ActorCriticCNN
from stewart.buffers.rollout_buffer import RolloutBuffer
import numpy as np

class PPO:
    """
    Implementación del algoritmo Proximal Policy Optimization (PPO)
    Utiliza una arquitectura Actor-Critic con convoluciones (CNN).
    """

    def __init__(
        self,
        model=None,
        obs_shape=(3, 84, 84),
        action_dim=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=3e-4,
        n_steps=1024,
        batch_size=256,
        epochs=6,
        gamma=0.99,
        gae_lambda=0.98,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.001,
        max_grad_norm=0.5,
    ):
        """
        Inicializa el agente PPO.

        Parámetros clave:
        - model: Red neuronal Actor-Critic (opcional).
        - obs_shape: Dimensión de la observación (C,H,W).
        - action_dim: Dimensión del espacio de acción.
        - lr: Tasa de aprendizaje (alfa).
        - n_steps: Cantidad de pasos por rollout.
        - batch_size: Tamaño del minibatch para actualización.
        - epochs: Cantidad de épocas por actualización.
        - gamma: Factor de descuento.
        - gae_lambda: Parámetro para Generalized Advantage Estimation.
        - clip_coef: Coeficiente de clip para la pérdida PPO.
        - vf_coef: Coeficiente para la pérdida del value function.
        - ent_coef: Coeficiente de bonificación por entropía.
        - max_grad_norm: Máximo valor para norm de gradiente (clipping).
        """
        self.device = device
        self.model = model or ActorCriticCNN(obs_shape=obs_shape, action_dim=action_dim)
        self.model = self.model.to(device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.n_steps, self.batch_size, self.epochs = n_steps, batch_size, epochs
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_coef, self.vf_coef, self.ent_coef = clip_coef, vf_coef, ent_coef
        self.max_grad_norm = max_grad_norm
        self.last_mean_advantage = 0.0

        self.ent_coef_base = ent_coef
        self.entropy_anneal_pct = 0.1  # o recíbelo como argumento también si quieres que sea configurable
        self.total_steps = int(1e6)    # también configurable
        self.entropy_anneal_steps = int(self.total_steps * self.entropy_anneal_pct)
        self.current_step = 0

        self.buffer = RolloutBuffer(
            n_steps,
            (
                (obs_shape[1], obs_shape[2], obs_shape[0])
                if obs_shape[0] in (1, 3)
                else obs_shape
            ),
            action_dim,
            device,
        )

    @torch.no_grad()
    def _obs_to_tensor(self, obs):
        """
        Convierte una observación NumPy (HWC) en un tensor normalizado (CHW) para la red.
        """
        t = torch.as_tensor(obs, device=self.device, dtype=torch.float32) / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,C,H,W)
        return t


    def update(self, buffer=None, epochs=None, batch_size=None):
        """
        Actualiza los parámetros del modelo PPO a partir de un buffer (interno o externo).

        Devuelve:
        Lista con [policy_loss, value_loss, entropy] por minibatch.
        """
        progress = min(self.current_step / self.entropy_anneal_steps, 1.0)
        current_ent_coef = self.ent_coef_base * (1.0 - progress)

        buf = buffer if buffer is not None else self.buffer
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size

        out = []
        for _ in range(epochs):
            for obs_b, act_b, oldlog_b, adv_b, ret_b in buf.get(batch_size):
                # Normalización de ventajas
                adv_b = ((adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)).detach()

                # Normalización de observaciones (de 0 a 1)
                if obs_b.dtype == torch.uint8:
                    obs_b = obs_b.float() / 255.0
                

                # Evaluación de acciones actuales (nuevas logits, entropía, valor)
                newlog, entropy, values = self.model.evaluate_actions(obs_b, act_b)
                ratio = (newlog - oldlog_b).exp()  # nuevo_logprob / viejo_logprob

                # PPO loss con clipping
                unclipped = -adv_b * ratio
                clipped = -adv_b * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(unclipped, clipped).mean()

                # Pérdida del value function (MSE)
                v_loss = F.mse_loss(values, ret_b)

                # Bonificación por entropía (fomenta exploración)
                ent = entropy.mean() if hasattr(entropy, "mean") else entropy

                # Pérdida total
                loss = pg_loss + self.vf_coef * v_loss - current_ent_coef * ent

                # Backpropagation
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5 
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.opt.step()

                out.append([pg_loss.item(), v_loss.item(), ent.item(), grad_norm])

        return out
