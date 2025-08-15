# stewart/envs/steward_sim_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stewart.sim.core import StewartSimCore
from gymnasium.wrappers import FrameStackObservation
from stewart.envs.to_chw import ToCHW

def _barycentric(P, A, B, C):
    (x,y),(x1,y1),(x2,y2),(x3,y3) = P,A,B,C
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-12:
        return -1.0, -1.0, -1.0
    u = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    v = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    w = 1.0 - u - v
    return u, v, w

class StewartBalanceEnv(gym.Env):
    """
    Obs: RGB uint8 (H,W,3)
    Act: Box(-1,1)^3 -> se escala por action_scale (rad)
    Reward (geom por defecto):
      + k_in si la bola está dentro del triángulo (local top)
      + k_center * (1 - dist_centro / radio_max)
      - k_out si está fuera
    Termination:
      - z_local de la bola por debajo de z_drop_local
      - fuera del triángulo por out_patience steps (si terminate_on_leave=True)
      - o por max_steps (truncation)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        img_size=84,
        use_gui=False,
        reward_mode: str = "image",
        action_scale: float = 0.35,
        k_in: float = 0.1,
        k_out: float = 0.2,
        k_center: float = 1.0,
        max_steps: int = 1000,
        spawn_margin_frac: float = 0.2,
        # ---- cámara/GUI -> Core ----
        cam_mode: str = "fixed",
        cam_eye_world=(0.0, 0.0, 0.28),
        cam_center_world=(0.0, 0.0, 0.10),
        cam_up_world=(0.0, 1.0, 0.0),
        sync_gui_camera: bool = False,
        gui_target: str = "top",
        gui_fixed_eye=(0.00, 0.00, 0.28),
        gui_fixed_center=(0.00, 0.00, 0.10),
        debug_frames: bool = False,
        # ---- terminación local ----
        terminate_on_leave: bool = True,
        out_patience: int = 8,
        z_drop_local: float = -0.01,   # ~1 cm por debajo del plano del top
        render_camera: bool = False,
    ):
        super().__init__()

        self.core = StewartSimCore(
            use_gui=use_gui,
            img_size=img_size,
            sync_gui_camera=sync_gui_camera,
            gui_target=gui_target,
            gui_fixed_eye=gui_fixed_eye,
            gui_fixed_center=gui_fixed_center,
            debug_frames=debug_frames,
            cam_mode=cam_mode,
            cam_eye_world=cam_eye_world,
            cam_center_world=cam_center_world,
            cam_up_world=cam_up_world,
        )

        self.img_size = int(img_size)
        self.reward_mode = reward_mode.lower().strip()
        self.action_scale = float(action_scale)
        self.k_in, self.k_out, self.k_center = float(k_in), float(k_out), float(k_center)
        self.max_steps = int(max_steps)
        self.spawn_margin_frac = float(spawn_margin_frac)

        self.render_camera = bool(render_camera)
        self._rmax_local = 1.0

        # terminación
        self.terminate_on_leave = bool(terminate_on_leave)
        self.out_patience = int(out_patience)
        self.z_drop_local = float(z_drop_local)
        self._outside_count = 0

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 255, shape=(self.img_size, self.img_size, 3), dtype=np.uint8)

        self.step_count = 0

    @classmethod
    def make(cls, img_size=84, use_gui=False, reward_mode="image"):
        env = cls(
            img_size=img_size,
            use_gui=use_gui,
            reward_mode=reward_mode,
            cam_mode="fixed",
            cam_eye_world=(0.0, 0.0, 0.32),
            cam_center_world=(0.0, 0.0, 0.10),
            cam_up_world=(0.0, 1.0, 0.0),
            render_camera=True,
        )
        env = FrameStackObservation(env, stack_size=4)
        env = ToCHW(env)
        return env

    # -------------- Gym API --------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._outside_count = 0

        # --- init p/ acciones/joints ---
        init = None
        if options:
            if "init_joint_rad" in options and options["init_joint_rad"] is not None:
                init = options["init_joint_rad"]
            elif "init_joint_deg" in options and options["init_joint_deg"] is not None:
                v = options["init_joint_deg"]
                init = float(np.deg2rad(v)) if isinstance(v, (int, float)) \
                    else np.deg2rad(np.asarray(v, dtype=np.float32))
        if init is None:
            init = getattr(self, "init_joint_rad", None)

        # core reset (spawnea bola) + primera obs
        self.core.reset(random_ball=True, seed=seed, init_joint_rad=init)
        obs = self.core.get_rgb()

        # --------- estado para reward ----------
        # rmax fijo (geom) a partir del triángulo del top
        A, B, C = self.core.get_top_triangle_xy()
        cx, cy = self.core.circumcenter_local_xy()
        cxcy = np.array([cx, cy], dtype=np.float64)
        self._rmax_local = max(
            np.linalg.norm(np.array(A) - cxcy),
            np.linalg.norm(np.array(B) - cxcy),
            np.linalg.norm(np.array(C) - cxcy),
        ) + 1e-8

        # velocidad de bola y suavidad de acción: estado previo
        bx, by = self.core.get_ball_local_xy()
        self._prev_ball_xy = np.array([bx, by], dtype=np.float64)
        self._prev_action  = np.zeros(3, dtype=np.float32)
        # dt del paso del env (~ 4 substeps a 1 kHz)
        self._dt_env = getattr(self, "_dt_env", 4e-3)

        self._last_ball_speed = 0.0
        self._last_delta_a    = 0.0

        return obs, {}

    def step(self, action):
        self.step_count += 1

        # acción escalada (en rad)
        a = np.clip(action, -1, 1).astype(np.float32) * self.action_scale

        # sim step
        self.core.step(a)

        # obs
        obs = self.core.get_rgb()

        # ------ métricas para reward (suavidad/velocidad) ------
        bx, by = self.core.get_ball_local_xy()
        ball_xy = np.array([bx, by], dtype=np.float64)
        self._last_ball_speed = float(
            np.linalg.norm(ball_xy - self._prev_ball_xy) / (self._dt_env if self._dt_env > 0 else 1e-3)
        )
        self._last_delta_a = float(np.linalg.norm(a - self._prev_action))
        # actualizar estado previo
        self._prev_ball_xy = ball_xy
        self._prev_action  = a

        # ------ reward ------
        reward, info_extra = self._reward_image()
        inside = bool(info_extra.get("inside_top", False))

        # --- terminaciones (local top) ---
        _, _, z_loc = self.core.get_ball_local_xyz()
        fell_below = (z_loc < self.z_drop_local)

        if self.terminate_on_leave:
            self._outside_count = 0 if inside else (self._outside_count + 1)
            left_for_too_long = (self._outside_count >= self.out_patience)
        else:
            left_for_too_long = False

        terminated = bool(fell_below or left_for_too_long)
        truncated  = bool(self.step_count >= self.max_steps)

        info = {
            "ball_world": self.core.get_dense_state()["ball_world"],
            "z_local": float(z_loc),
            "outside_streak": int(self._outside_count),
            "joint_angles_rad": self.core.get_joint_angles_rad(),
            **info_extra,
        }

        if self.render_camera:
            import cv2
            bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            cv2.imshow("Camera View", bgr)  # (puedes escalar si quieres)
            cv2.waitKey(1)

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self.core.get_rgb()

    def close(self):
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.core.close()

    # # -------------- Reward geométrico --------------
    # def _reward_geom(self):
    #     A, B, C = self.core.get_top_triangle_xy()
    #     cx, cy = self.core.circumcenter_local_xy()
    #     bx, by = self.core.get_ball_local_xy()

    #     u, v, w = _barycentric((bx, by), A, B, C)
    #     inside = (u >= 0.0) and (v >= 0.0) and (w >= 0.0)

    #     cxcy = np.array([cx, cy], dtype=np.float64)
    #     rmax = max(
    #         np.linalg.norm(np.array(A) - cxcy),
    #         np.linalg.norm(np.array(B) - cxcy),
    #         np.linalg.norm(np.array(C) - cxcy),
    #     ) + 1e-8
    #     r = np.linalg.norm(np.array([bx, by]) - cxcy)
    #     center_score = 1.0 - min(r / rmax, 1.0)

    #     # ------------------------------
    #     # Reward principal
    #     reward = (self.k_in if inside else -self.k_out) + self.k_center * center_score

    #     # Bonus por sobrevivir
    #     survive_bonus = 0.001  # o el valor que quieras
    #     reward += survive_bonus

    #     # ------------------------------
    #     return reward, {
    #         "inside_top": bool(inside),
    #         "center_score": float(center_score),
    #         "center_local": np.array([cx, cy], np.float32),
    #         "ball_local": np.array([bx, by], np.float32),
    #     }
    
    def _decode_seg(self, seg: np.ndarray):
        # int32 (no uint32) para que el -1 del fondo se mantenga en -1
        raw = seg.astype(np.int32)
        obj_id   = raw >> 24
        link_plus = raw & ((1 << 24) - 1)  # [0 .. (1<<24)-1]
        link_idx = link_plus - 1           # ahora sí: -1=base, 0..N-1=links reales
        return obj_id, link_idx
    
    def _distance_to_triangle_edges(self, P, A, B, C):
        def point_line_distance(P, Q, R):  # Distancia de P a línea QR
            PQ = np.array(P) - np.array(Q)
            QR = np.array(R) - np.array(Q)
            proj_len = np.dot(PQ, QR) / (np.linalg.norm(QR)**2 + 1e-8)
            proj_len = np.clip(proj_len, 0.0, 1.0)
            closest = np.array(Q) + proj_len * QR
            return np.linalg.norm(np.array(P) - closest)

        return min(
            point_line_distance(P, A, B),
            point_line_distance(P, B, C),
            point_line_distance(P, C, A),
        )

    # -------------- Reward por imagen --------------
    def _reward_image(self):
        rgb, seg = self.core.get_rgb_and_seg()
        obj_id, link_idx = self._decode_seg(seg)

        robot_id = self.core.robot_id
        top_link = self.core.top

        # --- Máscara del top (segmentación de la plataforma) ---
        top_mask = (obj_id == robot_id) & (link_idx == top_link)
        if not np.any(top_mask) and (robot_id in np.unique(obj_id)):
            top_mask = (obj_id == robot_id)

        # --- Centro visual del top (en imagen) ---
        if np.any(top_mask):
            ys, xs = np.where(top_mask)
            cx_img = xs.mean()
            cy_img = ys.mean()
            rmax_px = np.sqrt(((xs - cx_img)**2 + (ys - cy_img)**2).max()) + 1e-8
        else:
            h, w = seg.shape
            cx_img, cy_img = w / 2, h / 2
            rmax_px = max(w, h) / 2.0

        # --- Posición de la bola en coordenadas locales ---
        bx, by = self.core.get_ball_local_xy()
        cx, cy = self.core.circumcenter_local_xy()

        r = np.linalg.norm(np.array([bx, by]) - np.array([cx, cy]))
        rmax_geom = max(
            np.linalg.norm(np.array(A) - np.array([cx, cy]))
            for A in self.core.get_top_triangle_xy()
        ) + 1e-8
        center_score = 1.0 - min(r / rmax_geom, 1.0)

        # --- Verificar si el centro de la bola + radio ya están fuera ---
        # Considera que la bola puede estar parcialmente fuera.
        ball_radius = getattr(self.core, "ball_radius", 0.013)  # 26mm / 2, default si no está definido
        A, B, C = self.core.get_top_triangle_xy()
        u, v, w = _barycentric((bx, by), A, B, C)
        margin = 0.5 * ball_radius  # margen para decir "está saliendo"
        inside = (u >= -margin) and (v >= -margin) and (w >= -margin)

        reward = (self.k_in if inside else -self.k_out) + self.k_center * center_score
        reward += 0.001  # sobrevivencia

        return reward, {
            "inside_top": bool(inside),
            "center_score": float(center_score),
            "center_local": np.array([cx, cy], np.float32),
            "ball_local": np.array([bx, by], np.float32),
        }