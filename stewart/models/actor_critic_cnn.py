import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, TransformedDistribution, TanhTransform
from stewart.models.vision_cnn import VisionCNN
import numpy as np  # usado en la fórmula de entropía

class ActorCriticCNN(nn.Module):
    """
    Actor-Critic con acciones continuas modeladas como una Gaussiana diagonal transformada por tanh.
    Utiliza una CNN para extraer características de imágenes como observaciones.

    Salidas:
    - Política (actor): distribuciones de probabilidad sobre acciones continuas [-1, 1]
    - Valor (critic): estimación escalar del estado-valor V(s)
    """

    EPS = 1e-8

    def __init__(
        self,
        action_dim: int = 3,
        feature_dim: int = 256,
        logstd_init: float = -0.5,
        obs_shape=None
    ):
        """
        Parámetros:
        - action_dim: número de dimensiones de la acción continua
        - feature_dim: dimensión del vector latente extraído por la CNN
        - logstd_init: valor inicial del logaritmo de la desviación estándar de la política
        - obs_shape: forma de la observación, usada para deducir número de canales (puede ser (C,H,W) o (H,W,C))
        """
        super().__init__()

        # Deducción de canales de entrada a partir de la forma de la observación
        in_channels = 12  # por defecto RGB stack de 4 frames
   
        # Red convolucional para extraer features del input visual
        self.backbone = VisionCNN(in_channels=in_channels, feat_dim=feature_dim)

        # Actor: red totalmente conectada que predice la media de la acción
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, action_dim)
        )

        # Crítico: red que predice el valor del estado
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),  
            nn.Linear(64, 1)
        )

        # log-std es un parámetro aprendido, compartido entre todos los estados (no depende de la observación)
        self.logstd = nn.Parameter(torch.full((action_dim,), float(logstd_init)))

        # Inicialización ortogonal para mayor estabilidad
        self.apply(self._ortho_init)
        

    @staticmethod
    def _ortho_init(m):
        """
        Inicialización ortogonal para capas lineales y convolucionales.
        """
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, obs, return_pretanh_mean: bool = False):
        """
        Computa la distribución de acciones (policy) y el valor del estado (critic) para una observación dada.

        Parámetros:
        - obs: batch de observaciones
        - return_pretanh_mean: si es True, también retorna la media antes de aplicar tanh (útil para `act()` determinista)

        Devuelve:
        - dist: distribución de acciones (transformada por tanh)
        - value: valor del estado V(s)
        - mean (opcional): media antes de tanh
        """
        feat = self.backbone(obs)
        mean = self.policy(feat)

        # Control del rango del log_std (evita explosiones numéricas)
        logstd = self.logstd.clamp(-5.0, 2.0)
        std = logstd.exp().expand_as(mean)

        # Distribución base: Gaussiana diagonal
        base = Normal(mean, std)

        # Distribución transformada con tanh para restringir salida a [-1, 1]
        squash = TransformedDistribution(base, [TanhTransform(cache_size=1)])
        dist = Independent(squash, 1)  # acciones como un vector multivariado

        value = self.value_head(feat).squeeze(-1)

        if return_pretanh_mean:
            return dist, value, mean
        return dist, value

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        """
        Escoge una acción dada una observación.

        Parámetros:
        - obs: observación actual
        - deterministic: si True, usa la media (sin muestreo); si False, muestrea de la política

        Devuelve:
        - action: acción final en [-1, 1]
        - logprob: log-probabilidad de la acción tomada (con Jacobiano de tanh incluido)
        - value: valor estimado del estado
        """
        dist, value, mean = self.forward(obs, return_pretanh_mean=True)

        if deterministic:
            action = torch.tanh(mean)
            sampled = mean  # sin ruido
        else:
            sampled = dist.rsample()          
            action = torch.tanh(sampled)   

        logprob = dist.log_prob(sampled)    
        logprob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1) 

        return action, logprob, value

    def evaluate_actions(self, obs, actions):
        """
        Evalúa un batch de acciones bajo la política actual.

        Parámetros:
        - obs: batch de observaciones
        - actions: batch de acciones tomadas

        Devuelve:
        - logprob: log-probabilidad de las acciones dadas
        - entropy: entropía aproximada de la política
        - value: valor estimado de los estados
        """
        dist, value = self.forward(obs)

        # Evita problemas numéricos en los bordes del tanh
        actions = actions.clamp(-1 + 1e-6, 1 - 1e-6)
        logprob = dist.log_prob(actions)

        # Entropía aproximada de la política (usando la base Normal, ignorando tanh)
        logstd = self.logstd.clamp(-5.0, 2.0)
        entropy = 0.5 + 0.5 * np.log(2 * np.pi) + logstd
        entropy = entropy.sum(-1)
        entropy = entropy.mean()

        return logprob, entropy, value