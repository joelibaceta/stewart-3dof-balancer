# stewart/envs/stewart_balance_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stewart.sim.core import StewartSimCore

def _barycentric(P, A, B, C):
    (x,y),(x1,y1),(x2,y2),(x3,y3) = P,A,B,C
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-12:
        return -1.0, -1.0, -1.0  # degenerado
    u = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    v = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    w = 1.0 - u - v
    return u, v, w

class StewartBalanceEnv(gym.Env):
    """
    Obs:  RGB uint8 (H,W,3)
    Act:  Box(-1,1)^3 -> se escala por action_scale (rad)
    Reward (geom, por defecto):
      + k_in si la bola está dentro del triángulo (en coords locales del top)
      + k_center * (1 - dist_al_centro / radio_max)  (0..1)
      - k_out si está fuera
    Reward (image, opcional):
      mismo concepto pero desde la máscara de segmentación de PyBullet.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        img_size=84,
        use_gui=False,
        reward_mode: str = "geom",     # "geom" | "image"
        action_scale: float = 0.35,    # rad ~ 20°
        k_in: float = 0.1,
        k_out: float = 0.2,
        k_center: float = 1.0,
        max_steps: int = 1000,
        spawn_margin_frac: float = 0.2
    ):
        super().__init__()
        self.core = StewartSimCore(use_gui=use_gui, img_size=img_size)
        self.img_size = int(img_size)
        self.reward_mode = reward_mode
        self.action_scale = float(action_scale)
        self.k_in, self.k_out, self.k_center = float(k_in), float(k_out), float(k_center)
        self.max_steps = int(max_steps)
        self.spawn_margin_frac = float(spawn_margin_frac)

        # espacios
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 255, shape=(self.img_size, self.img_size, 3), dtype=np.uint8)

        self.step_count = 0

    # -------------- Gym API --------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.core.reset(random_ball=True, seed=seed, margin_frac=None, margin_m= max(0.02, 2*self.core.ball_r))
        obs = self.core.get_rgb()
        return obs, {}

    def step(self, action):
        self.step_count += 1

        # acc -> rad
        a = np.clip(action, -1, 1).astype(np.float32) * self.action_scale
        self.core.step(a)

        # observación
        obs = self.core.get_rgb()

        # reward
        if self.reward_mode == "image":
            reward, info_extra = self._reward_image()
        else:
            reward, info_extra = self._reward_geom()

        # terminaciones
        # (geométrica simple: si la bola cae demasiado bajo en Z, terminó)
        ball_world = self.core.get_dense_state()["ball_world"]
        terminated = bool(ball_world[2] < 0.02)  # ajusta según tu robot
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "ball_world": ball_world,
            **info_extra
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self.core.get_rgb()

    def close(self):
        self.core.close()

    # -------------- Reward geométrico (en frame local del top) --------------
    def _reward_geom(self):
        # vértices del top en local XY
        A, B, C = self.core.get_top_triangle_xy()
        # centro (circuncentro) en local
        cx, cy = self.core.circumcenter_local_xy()
        # bola en local
        bx, by = self.core.get_ball_local_xy()

        # inside por barycéntricas
        u, v, w = _barycentric((bx, by), A, B, C)
        inside = (u >= 0.0) and (v >= 0.0) and (w >= 0.0)

        # normalización radial (tomamos el máximo de dist desde centro a vértices)
        cxcy = np.array([cx, cy], dtype=np.float64)
        rmax = max(
            np.linalg.norm(np.array(A) - cxcy),
            np.linalg.norm(np.array(B) - cxcy),
            np.linalg.norm(np.array(C) - cxcy),
        ) + 1e-8
        r = np.linalg.norm(np.array([bx, by]) - cxcy)
        center_score = 1.0 - min(r / rmax, 1.0)  # 1 en centro, 0 en borde o más

        reward = (self.k_in if inside else -self.k_out) + self.k_center * center_score

        return reward, {
            "inside_top": bool(inside),
            "center_score": float(center_score),
            "center_local": np.array([cx, cy], np.float32),
            "ball_local": np.array([bx, by], np.float32),
        }

    # -------------- Reward por imagen (máscara de segmentación) --------------
    def _reward_image(self):
        import numpy as np
        rgb, seg = self.core.get_rgb_and_seg()

        # seg codifica (obj_id << 24) + link_index, background suele ser -1
        obj_id = seg // (1 << 24)
        link_idx = seg %  (1 << 24)

        robot_id = self.core.robot_id
        top_link = self.core.top
        ball_id  = self.core.ball_id

        top_mask  = (obj_id == robot_id) & (link_idx == top_link)
        ball_mask = (obj_id == ball_id)

        # centroide del top
        if np.any(top_mask):
            ys, xs = np.where(top_mask)
            cx = xs.mean()
            cy = ys.mean()
            rmax = np.sqrt(((xs - cx)**2 + (ys - cy)**2).max()) + 1e-8
        else:
            h, w = seg.shape
            cx, cy = w/2, h/2
            rmax = max(w, h)/2.0

        # centroide de la bola
        if np.any(ball_mask):
            bys, bxs = np.where(ball_mask)
            bx = bxs.mean()
            by = bys.mean()
            inside = bool(top_mask[int(round(by)), int(round(bx))]) if 0 <= int(round(by)) < seg.shape[0] and 0 <= int(round(bx)) < seg.shape[1] else False
            dist = np.sqrt((bx - cx)**2 + (by - cy)**2)
            center_score = 1.0 - min(dist / rmax, 1.0)
        else:
            inside = False
            center_score = 0.0

        reward = (self.k_in if inside else -self.k_out) + self.k_center * center_score

        return reward, {
            "inside_top": bool(inside),
            "center_score": float(center_score),
            "img_center_px": np.array([cx, cy], np.float32),
        }