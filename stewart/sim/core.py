# stewart/sim/core.py
import pybullet as p, pybullet_data, numpy as np
from importlib import resources
from .utils import (
    link_index_by_name, joint_index_by_name, axes_from_quat,
    camera_view_proj, get_rgb_and_seg,
    joint_limits,                    
    offset_triangle_metric,          
    sample_uniform_triangle          
)
class StewartSimCore:
    def __init__(self,
        use_gui=False,
        img_size=84,
        sync_gui_camera=False,
        gui_target="top",
        gui_fixed_eye=(0.00, 0.00, 0.28),
        gui_fixed_center=(0.00, 0.00, 0.10),
        debug_frames=False,
        cam_mode="fixed",
        cam_eye_world=(0.00, 0.00, 0.28),
        cam_center_world=(0.00, 0.00, 0.10),
        cam_up_world=(0.00, 1.00, 0.00),
        rotate180=True
    ):
        self.use_gui = use_gui
        self.sync_gui_camera = sync_gui_camera
        self.gui_target = gui_target
        self.gui_fixed_eye = np.array(gui_fixed_eye, dtype=np.float32)
        self.gui_fixed_center = np.array(gui_fixed_center, dtype=np.float32)
        self.debug_frames = bool(debug_frames)

        self.cam_mode = cam_mode
        self.cam_eye_world = np.array(cam_eye_world, dtype=np.float32)
        self.cam_center_world = np.array(cam_center_world, dtype=np.float32)
        self.cam_up_world = np.array(cam_up_world, dtype=np.float32)

        self.img_w = self.img_h = img_size
        self.ball_id = None
        self._cons_created = False
        self.rng = np.random.default_rng()

        self.rotate180 = bool(rotate180)

        # bola (metros / kg)
        self.ball_r = 0.008
        self.ball_m = 0.055

        # rango aleatorio por defecto (±grados)
        self.init_rand_deg = 25.0

        # PyBullet
        p.connect(p.GUI if use_gui else p.DIRECT, options="--opengl2")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        if self.use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # assets
        assets_root = resources.files("stewart").joinpath("assets")
        p.setAdditionalSearchPath(str(assets_root))
        robot_urdf = str(assets_root / "robot.urdf")
        self.robot_id = p.loadURDF(robot_urdf, [0,0,0.1], useFixedBase=True)

        self._init_indices()
        self._solver()
        self._create_constraints()

        self.reset(random_ball=True)

    # ---------- setup ----------
    def _init_indices(self):
        self.top  = link_index_by_name(self.robot_id, "top")
        self.tip2 = link_index_by_name(self.robot_id, "arm2_tip")
        self.tip3 = link_index_by_name(self.robot_id, "arm3_tip")
        self.j1   = joint_index_by_name(self.robot_id, "joint_dot1")
        self.j2   = joint_index_by_name(self.robot_id, "joint_dot2")
        self.j3   = joint_index_by_name(self.robot_id, "joint_dot3")

        for j in range(p.getNumJoints(self.robot_id)):
            if j not in (self.j1, self.j2, self.j3):
                p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

    def set_seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _solver(self):
        p.setPhysicsEngineParameter(numSolverIterations=300, erp=0.85,
                                    contactERP=0.85, frictionERP=0.85, globalCFM=1e-7)
        p.setTimeStep(1/1000)
        p.changeDynamics(self.robot_id, self.top, lateralFriction=0.12,
                         rollingFriction=0.0, spinningFriction=0.0)

    def _create_constraints(self):
        if self._cons_created: return

        def wpose(l): return p.getLinkState(self.robot_id, l, True)[4]
        def w2l(link, pt):
            ls = p.getLinkState(self.robot_id, link, True); pos, orn = ls[4], ls[5]
            inv_pos, inv_orn = p.invertTransform(pos, orn)
            local,_ = p.multiplyTransforms(inv_pos, inv_orn, pt, [0,0,0,1]); return local

        t2 = wpose(self.tip2)
        p2a = w2l(self.top,  t2); p2b = w2l(self.tip2, t2)
        c2 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip2,
                                p.JOINT_POINT2POINT, [0,0,0], p2a, p2b)
        p.changeConstraint(c2, maxForce=20000, erp=0.85)
        g2 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip2,
                                p.JOINT_GEAR, [0,0,1], [0,0,0], [0,0,0])
        p.changeConstraint(g2, gearRatio=1.0, maxForce=300)

        t3 = wpose(self.tip3)
        p3a = w2l(self.top,  t3); p3b = w2l(self.tip3, t3)
        c3 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip3,
                                p.JOINT_POINT2POINT, [0,0,0], p3a, p3b)
        p.changeConstraint(c3, maxForce=20000, erp=0.85)
        g3 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip3,
                                p.JOINT_GEAR, [0,0,1], [0,0,0], [0,0,0])
        p.changeConstraint(g3, gearRatio=1.0, maxForce=300)

        self._cons_created = True

    # ---------- cámara ----------
    def _view_proj(self):
        if self.cam_mode == "fixed":
            eye = self.cam_eye_world; center = self.cam_center_world; up = self.cam_up_world
            view = p.computeViewMatrix(eye.tolist(), center.tolist(), up.tolist())
            proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0)
            return view, proj, center, eye, up
        else:
            return camera_view_proj(self.robot_id, self.top, self.img_w, self.img_h)

    # ---------- utilidades ----------
    def get_ball_local_xyz(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        top_pos, top_orn = self._pose(self.top)
        inv_pos, inv_orn = p.invertTransform(top_pos, top_orn)
        (x, y, z), _ = p.multiplyTransforms(inv_pos, inv_orn, ball_pos, [0,0,0,1])
        return float(x), float(y), float(z)

    def get_joint_angles_rad(self):
        q1 = p.getJointState(self.robot_id, self.j1)[0]
        q2 = p.getJointState(self.robot_id, self.j2)[0]
        q3 = p.getJointState(self.robot_id, self.j3)[0]
        return np.array([q1, q2, q3], dtype=np.float32)

    def set_joint_angles_rad(self, q):
        q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
        p.resetJointState(self.robot_id, self.j1, q1)
        p.resetJointState(self.robot_id, self.j2, q2)
        p.resetJointState(self.robot_id, self.j3, q3)


    # ---------- bola ----------
    def _spawn_ball(self, pos=None):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_r)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_r, rgbaColor=[0.9,0.9,0.9,1])
        if pos is None:
            aabb_min, aabb_max = p.getAABB(self.robot_id, self.top)
            center = [(aabb_min[i] + aabb_max[i]) * 0.5 for i in range(3)]
            _,orn = self._pose(self.top); Z = axes_from_quat(orn)[2]
            lift = self.ball_r + 0.01
            pos = [center[0]+Z[0]*lift, center[1]+Z[1]*lift, center[2]+Z[2]*lift]

        self.ball_id = p.createMultiBody(self.ball_m, col, vis, basePosition=pos)

        # Menos rebote, más fricción y algo de damping
        p.changeDynamics(self.ball_id, -1,
            mass=0.018,  # ~18 gramos, puedes ajustar según comportamiento
            restitution=0.05,            # rebote alto, típico del vidrio
            lateralFriction=0.1,         # superficie lisa
            rollingFriction=0.0001,      # mínimo para que ruede libremente
            spinningFriction=0.0001,     # casi sin resistencia al giro
            linearDamping=0.001,         # muy poca resistencia al movimiento lineal
            angularDamping=0.001         # muy poca resistencia al giro
        )

    def _get_top_vertices_xy(self):
        def link_world(name):
            for i in range(p.getNumJoints(self.robot_id)):
                if p.getJointInfo(self.robot_id, i)[12].decode() == name:
                    return p.getLinkState(self.robot_id, i, True)[4]
        a_w = link_world("arm1_tip")
        b_w = link_world("arm2_tip")
        c_w = link_world("arm3_tip")

        top_pos, top_orn = self._pose(self.top)
        inv_pos, inv_orn = p.invertTransform(top_pos, top_orn)

        def to_local_xy_w(wpt):
            (x, y, _), _ = p.multiplyTransforms(inv_pos, inv_orn, wpt, [0,0,0,1])
            return np.array([x, y], dtype=np.float64)

        A = to_local_xy_w(a_w); B = to_local_xy_w(b_w); C = to_local_xy_w(c_w)
        return A, B, C, top_pos, top_orn

    def sample_point_on_top(self, margin_frac: float | None = 0.2, margin_m: float | None = None):
        if margin_m is None:
            margin_m = self.ball_r + 0.015

        A, B, C, top_pos, top_orn = self._get_top_vertices_xy()

        if margin_m is not None and margin_m > 0.0:
            A2, B2, C2 = offset_triangle_metric(A, B, C, float(margin_m))
        else:
            # fallback fraccional
            from .utils import triangle_incenter, shrink_triangle_towards
            I = triangle_incenter(A, B, C)
            t = float(np.clip(margin_frac or 0.0, 0.0, 0.49))
            A2, B2, C2 = shrink_triangle_towards(A, B, C, I, t)

        x_local, y_local = sample_uniform_triangle(self.rng, A2, B2, C2)

        X, Y, Z = axes_from_quat(top_orn)
        lift = self.ball_r + 0.01
        return [
            top_pos[0] + X[0]*x_local + Y[0]*y_local + Z[0]*lift,
            top_pos[1] + X[1]*x_local + Y[1]*y_local + Z[1]*lift,
            top_pos[2] + X[2]*x_local + Y[2]*y_local + Z[2]*lift,
        ]

    # ---------- API pública ----------
    def reset(self, random_ball=True, seed=None, margin_frac=0.2, margin_m=None,
            init_joint_rad=None, init_rand_deg=None,
            spawn_mode="random",          # "random" | "center"
            spawn_margin_m=None,          # override métrico seguro (recomendado)
            settle_steps=1):              # pasos Bullet antes del 1er frame
        if seed is not None:
            self.set_seed(seed)

        # --- joints iniciales (igual que antes) ---
        if init_joint_rad is None:
            deg = self.init_rand_deg if init_rand_deg is None else float(init_rand_deg)
            r = np.deg2rad(deg)
            q = self.rng.uniform(-r, +r, size=3).astype(np.float32)
        else:
            qv = np.asarray(init_joint_rad, dtype=np.float32)
            q = np.repeat(qv, 3)[:3] if qv.shape == () else qv.astype(np.float32)

        limits = joint_limits(self.robot_id, (self.j1, self.j2, self.j3))
        q = np.array([np.clip(q[i], limits[i][0], limits[i][1]) for i in range(3)], dtype=np.float32)

        p.resetBasePositionAndOrientation(self.robot_id, [0,0,0.1], [0,0,0,1])
        self.set_joint_angles_rad(q)

        # --- recrear bola ---
        if self.ball_id is not None:
            try: p.removeBody(self.ball_id)
            except: pass
            self.ball_id = None

        if not random_ball or spawn_mode == "center":
            # Spawn exactamente en el centro del top, elevado por la normal
            aabb_min, aabb_max = p.getAABB(self.robot_id, self.top)
            center = [(aabb_min[i] + aabb_max[i]) * 0.5 for i in range(3)]
            _, top_orn = self._pose(self.top); Z = axes_from_quat(top_orn)[2]
            lift = self.ball_r + 0.01
            pos = [center[0]+Z[0]*lift, center[1]+Z[1]*lift, center[2]+Z[2]*lift]
            self._spawn_ball(pos=pos)
        else:
            # Aleatorio PERO con margen MÉTRICO seguro
            safe_m = (spawn_margin_m if spawn_margin_m is not None
                    else (self.ball_r + 0.020 if margin_m is None else float(margin_m)))
            pos = self.sample_point_on_top(margin_m=safe_m)
            self._spawn_ball(pos=pos)

        # Asienta y quita velocidad residual
        for _ in range(max(0, int(settle_steps))):
            p.stepSimulation()
        p.resetBaseVelocity(self.ball_id, [0,0,0], [0,0,0])

        return q

    def step(self, target_angles_rad):
        a1, a2, a3 = target_angles_rad
        p.setJointMotorControl2(self.robot_id, self.j1, p.POSITION_CONTROL, targetPosition=float(a1), force=80)
        p.setJointMotorControl2(self.robot_id, self.j2, p.POSITION_CONTROL, targetPosition=float(a2), force=80)
        p.setJointMotorControl2(self.robot_id, self.j3, p.POSITION_CONTROL, targetPosition=float(a3), force=80)
        for _ in range(4):
            p.stepSimulation()
        self._maybe_debug_frames()

    # ---------- GUI camera sync ----------
    def _sync_gui_camera(self, center, eye):
        if not self.use_gui or not self.sync_gui_camera:
            return

        if self.cam_mode == "fixed":
            center_v = np.array(center, dtype=np.float32)
            eye_v = np.array(eye, dtype=np.float32)
        else:
            if self.gui_target == "fixed":
                center_v = self.gui_fixed_center; eye_v = self.gui_fixed_eye
            else:
                if self.gui_target == "base":
                    cpos, corn = p.getBasePositionAndOrientation(self.robot_id)
                else:
                    cpos, corn = self._pose(self.top)
                _, _, Z = axes_from_quat(corn)
                center_v = np.array(cpos, dtype=np.float32)
                eye_v = center_v + np.array(Z, dtype=np.float32) * 0.25

        f = center_v - eye_v
        dist = float(np.linalg.norm(f) + 1e-9)
        yaw = float(np.degrees(np.arctan2(f[1], f[0])))
        pitch = float(-np.degrees(np.arctan2(f[2], np.linalg.norm(f[:2]))))
        p.resetDebugVisualizerCamera(dist, yaw, pitch, center_v.tolist())

    def get_rgb(self):
        view, proj, center, eye, _up = self._view_proj()
        self._sync_gui_camera(center, eye)
        rgb, _ = get_rgb_and_seg(self.img_w, self.img_h, view, proj)
        return rgb

    def get_rgb_and_seg(self):
        view, proj, center, eye, _up = self._view_proj()
        self._sync_gui_camera(center, eye)
        return get_rgb_and_seg(self.img_w, self.img_h, view, proj)

    def get_dense_state(self):
        ball_pos,_ = p.getBasePositionAndOrientation(self.ball_id)
        return {"ball_world": np.array(ball_pos, dtype=np.float32)}

    # ---------- wrappers para el ENV ----------
    def get_top_triangle_xy(self):
        A, B, C, _, _ = self._get_top_vertices_xy()
        return (float(A[0]), float(A[1])), (float(B[0]), float(B[1])), (float(C[0]), float(C[1]))

    def get_ball_local_xy(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        top_pos, top_orn = self._pose(self.top)
        inv_pos, inv_orn = p.invertTransform(top_pos, top_orn)
        (x, y, _), _ = p.multiplyTransforms(inv_pos, inv_orn, ball_pos, [0,0,0,1])
        return float(x), float(y)

    def circumcenter_local_xy(self):
        A, B, C, _, _ = self._get_top_vertices_xy()
        ax, ay = A; bx, by = B; cx, cy = C
        d = 2.0 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
        if abs(d) < 1e-12:
            ux, uy = (ax+bx+cx)/3.0, (ay+by+cy)/3.0
        else:
            ax2 = ax*ax + ay*ay; bx2 = bx*bx + by*by; cx2 = cx*cx + cy*cy
            ux = (ax2*(by-cy) + bx2*(cy-ay) + cx2*(ay-by)) / d
            uy = (ax2*(cx-bx) + bx2*(ax-cx) + cx2*(bx-ax)) / d
        return float(ux), float(uy)

    # ---------- debug helpers ----------
    def _draw_axes_at(self, pos, orn, scale=0.05, life=0.2):
        X, Y, Z = axes_from_quat(orn)
        p.addUserDebugLine(pos, (pos[0]+X[0]*scale, pos[1]+X[1]*scale, pos[2]+X[2]*scale),
                           [255, 0, 0], lifeTime=life, lineWidth=2)
        p.addUserDebugLine(pos, (pos[0]+Y[0]*scale, pos[1]+Y[1]*scale, pos[2]+Y[2]*scale),
                           [0, 255, 0], lifeTime=life, lineWidth=2)
        p.addUserDebugLine(pos, (pos[0]+Z[0]*scale, pos[1]+Z[1]*scale, pos[2]+Z[2]*scale),
                           [0, 0, 255], lifeTime=life, lineWidth=2)

    def _maybe_debug_frames(self):
        if not self.debug_frames: return
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        self._draw_axes_at(base_pos, base_orn, scale=0.05, life=0.15)
        top_pos, top_orn = self._pose(self.top)
        self._draw_axes_at(top_pos, top_orn, scale=0.05, life=0.15)

    # ---------- helpers privados ----------
    def _pose(self, link):
        ls = p.getLinkState(self.robot_id, link, True)
        return ls[4], ls[5]

    def close(self):
        try: p.disconnect()
        except: pass