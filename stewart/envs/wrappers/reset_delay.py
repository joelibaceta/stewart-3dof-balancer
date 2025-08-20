# utils/wrappers.py
import time
import gymnasium as gym
import numpy as np

class ResetDelayWrapper(gym.Wrapper):
    """
    Tras reset(), ejecuta 'steps' pasos con una acción neutra y
    devuelve SOLO la última obs/info. Útil para llenar el frame stack
    y dejar que el entorno se estabilice.
    """
    def __init__(self, env, steps: int = 4, action_neutral=None):
        super().__init__(env)
        self.steps = int(steps)
        self.action_neutral = action_neutral  # e.g. np.zeros(3, dtype=np.float32)

    def _zero_action(self):
        if self.action_neutral is not None:
            return np.asarray(self.action_neutral, dtype=self.action_space.dtype)
        # Box(-1,1)^3 -> acción neutra = 0
        if isinstance(self.action_space, gym.spaces.Box):
            return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        # fallback
        return self.action_space.sample() * 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        a0 = self._zero_action()
        # “quemamos” steps; si termina antes, reseteamos y seguimos
        for _ in range(self.steps):
            obs, _, terminated, truncated, _ = self.env.step(a0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["burnin_steps"] = self.steps
        return obs, info