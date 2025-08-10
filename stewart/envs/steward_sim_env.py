# stewart/envs/stewart_balance_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stewart.sim.core import StewartSimCore
import pybullet as p

class StewartBalanceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    def __init__(self, img_size=84, use_gui=False):
        super().__init__()
        self.core = StewartSimCore(use_gui=use_gui, img_size=img_size)
        self.max_angle = np.array([3.14,3.14,3.14], dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(0,255, shape=(img_size,img_size,3), dtype=np.uint8)
        self.max_steps, self.step_count = 1000, 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.core.reset(random_ball=True, seed=seed)
        return self.core.get_rgb(), {}

    def step(self, action):
        self.step_count += 1
        a = np.clip(action, -1, 1).astype(np.float32) * self.max_angle
        self.core.step(a)
        obs = self.core.get_rgb()
        # recompensa: -dist^2 al circuncentro (en mundo, aproximación)
        center = np.array(self.core._circumcenter_world(), np.float32)
        ball = self.core.get_dense_state()["ball_world"]
        dist2 = float(np.sum((ball - center)**2))
        reward = -dist2
        terminated = (ball[2] < 0.02)  # cayó muy abajo
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def render(self):
        return self.core.get_rgb()

    def close(self):
        self.core.close()