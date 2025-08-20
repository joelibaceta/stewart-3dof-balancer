# stewart/envs/steward_sim_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stewart.sim.core import StewartSimCore
from gymnasium.wrappers import FrameStackObservation
from stewart.envs.to_chw import ToCHW
from stewart.envs.reset_delay import ResetDelayWrapper

from stewart.utils.geometry import barycentric, distance_to_triangle_edges
from stewart.utils.segmentation import decode_seg



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
        action_scale: float = 0.1,
        k_in: float = 0.05,
        k_out: float = 0.1,
        k_center: float = 1.0,
        k_delta_a: float = 0.01,
        max_steps: int = 1000,
        spawn_margin_frac: float = 0.3,
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
        z_drop_local: float = -0.01,   
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

        self.k_delta_a = float(k_delta_a)

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
        #env = ResetDelayWrapper(env, steps=2)
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
        reward, info_extra = self.reward()
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
    

    def reward(self):
        ball_pos = np.array(self.core.get_ball_local_xy(), dtype=np.float64)
        center_pos = np.array(self.core.circumcenter_local_xy(), dtype=np.float64)
        dist_center = np.linalg.norm(ball_pos - center_pos)

        # Recompensa por cercanía al centro (shaping)
        rmax_geom = max(np.linalg.norm(np.array(v) - center_pos) for v in self.core.get_top_triangle_xy()) + 1e-8
        
        # Esta función premia la cercanía de forma suave y continua
        center_reward = self.k_center * (1.0 - min(dist_center / rmax_geom, 1.0))

        # Recompensa por estar dentro del triángulo (o penalización)
        inside = self._is_ball_inside_top(ball_pos)
        if inside:
            inside_reward = self.k_in
        else:
            inside_reward = -self.k_out

        # Penalización por cambios bruscos
        delta_a_penalty = -self.k_delta_a * self._last_delta_a

        # Recompensa por sobrevivir (constante)
        survival_reward = 0.001

        # Recompensa total
        total_reward = center_reward + inside_reward + delta_a_penalty + survival_reward


        return total_reward, {
            "inside_top": bool(inside),
            "center_score": float(center_reward),
            "center_local": center_pos.astype(np.float32),
            "ball_local": ball_pos.astype(np.float32),
            "delta_a": float(self._last_delta_a),
            "dist_center": float(dist_center),
        }
    
    def _get_top_visual_data(self):
        rgb, seg = self.core.get_rgb_and_seg()
        obj_id, link_idx = decode_seg(seg)

        robot_id = self.core.robot_id
        top_link = self.core.top

        # Máscara del top en imagen
        top_mask = (obj_id == robot_id) & (link_idx == top_link)
        if not np.any(top_mask) and (robot_id in np.unique(obj_id)):
            top_mask = (obj_id == robot_id)

        # Centro visual del top (si existe máscara)
        if np.any(top_mask):
            ys, xs = np.where(top_mask)
            cx = xs.mean()
            cy = ys.mean()
            rmax = np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max()) + 1e-8
        else:
            h, w = seg.shape
            cx, cy = w / 2, h / 2
            rmax = max(w, h) / 2.0

        return top_mask, (cx, cy), rmax

    def _is_ball_inside_top(self, ball_pos):
        A, B, C = self.core.get_top_triangle_xy()
        u, v, w = barycentric(ball_pos, A, B, C)

        ball_radius = getattr(self.core, "ball_radius", 0.013)
        margin = 0.5 * ball_radius
        return (u >= -margin) and (v >= -margin) and (w >= -margin)