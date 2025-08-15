# ponlo cerca de tu clase env (mismo archivo o un util)
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ToCHW(gym.ObservationWrapper):
    """
    Convierte (T,H,W,C) uint8 -> (T*C, H, W) uint8.
    Asumimos que SIEMPRE hay frame stack antes de este wrapper.
    """
    def __init__(self, env):
        super().__init__(env)
        T, H, W, C = env.observation_space.shape  # p.ej. (4,84,84,3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(T * C, H, W), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray):
        # obs: (T,H,W,C) -> (T*C,H,W) sin ifs; si no viene así, que falle temprano
        T, H, W, C = obs.shape
        return obs.transpose(0, 3, 1, 2).reshape(T * C, H, W)