# stewart/sim/core.py
import pybullet as p, pybullet_data, numpy as np
from importlib import resources
from .utils import (
    link_index_by_name, joint_index_by_name, axes_from_quat,
    to_local_xy, triangle_incenter, shrink_triangle_towards,
    sample_uniform_triangle, camera_view_proj, get_rgb_and_seg
)

class StewartSimCore:
    def __init__(self, use_gui=False, img_size=84, sync_gui_camera=False):
        self.use_gui = use_gui
        self.sync_gui_camera = sync_gui_camera

        self.img_w = self.img_h = img_size
        self.ball_id = None
        self._cons_created = False
        self.rng = np.random.default_rng()

        # bola (metros / kg)
        self.ball_r = 0.008     # 0.8 cm de radio
        self.ball_m = 0.0055    # ~5.5 g

        # PyBullet
        p.connect(p.GUI if use_gui else p.DIRECT, options="--opengl2")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

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
        p.changeDynamics(self.ball_id, -1, lateralFriction=0.03,
                         rollingFriction=0.0001, spinningFriction=0.0001, restitution=0.15)

    def _get_top_vertices_xy(self):
        """Devuelve (A,B,C, top_pos, top_orn), donde A,B,C son vértices del top en XY local."""
        # busca los tips en mundo
        def link_world(name):
            for i in range(p.getNumJoints(self.robot_id)):
                if p.getJointInfo(self.robot_id, i)[12].decode() == name:
                    return p.getLinkState(self.robot_id, i, True)[4]
        a_w = link_world("arm1_tip")
        b_w = link_world("arm2_tip")
        c_w = link_world("arm3_tip")

        # al frame LOCAL del top (plano XY)
        top_pos, top_orn = self._pose(self.top)
        inv_pos, inv_orn = p.invertTransform(top_pos, top_orn)

        def to_local_xy_w(wpt):
            (x, y, _), _ = p.multiplyTransforms(inv_pos, inv_orn, wpt, [0,0,0,1])
            return np.array([x, y], dtype=np.float64)

        A = to_local_xy_w(a_w)
        B = to_local_xy_w(b_w)
        C = to_local_xy_w(c_w)
        return A, B, C, top_pos, top_orn


    def sample_point_on_top(self, margin_frac: float | None = 0.2, margin_m: float | None = None):
        """
        Devuelve posición en MUNDO para spawnear la bola.
        Usa margen MÉTRICO (metros) si está disponible; si no, cae a margen fraccional.
        """
        # --- margen por defecto en METROS ---
        if margin_m is None:
            margin_m = self.ball_r + 0.015  # bola + 1.5 cm extra

        # vértices en XY local + pose del top
        A, B, C, top_pos, top_orn = self._get_top_vertices_xy()

        if margin_m is not None and margin_m > 0.0:
            # ---------- OFFSET MÉTRICO DEL TRIÁNGULO ----------
            centroid = (A + B + C) / 3.0

            def edge_offset(P, Q, m):
                e = Q - P
                n = np.array([-e[1], e[0]], dtype=np.float64)
                n /= (np.linalg.norm(n) + 1e-12)
                if np.dot(n, centroid - P) < 0:  # asegura que apunta hacia adentro
                    n = -n
                d = float(np.dot(n, P) + m)      # n·x = d
                return n, d

            n1, d1 = edge_offset(B, C, margin_m)
            n2, d2 = edge_offset(C, A, margin_m)
            n3, d3 = edge_offset(A, B, margin_m)

            def intersect(nu, du, nv, dv):
                M = np.stack([nu, nv], axis=0)
                b = np.array([du, dv], dtype=np.float64)
                det = np.linalg.det(M)
                if abs(det) < 1e-12:
                    return centroid.copy()
                return np.linalg.solve(M, b)

            A2 = intersect(n1, d1, n2, d2)
            B2 = intersect(n2, d2, n3, d3)
            C2 = intersect(n3, d3, n1, d1)

            # Fallback si el margen colapsa el triángulo
            area = 0.5 * abs(
                A2[0]*(B2[1]-C2[1]) + B2[0]*(C2[1]-A2[1]) + C2[0]*(A2[1]-B2[1])
            )
            if area < 1e-10:
                # shrink fraccional suave hacia el incentro
                a = np.linalg.norm(B - C); b = np.linalg.norm(C - A); c = np.linalg.norm(A - B)
                I = (a*A + b*B + c*C) / (a + b + c + 1e-12)
                t = 0.3
                A2 = (1.0 - t) * A + t * I
                B2 = (1.0 - t) * B + t * I
                C2 = (1.0 - t) * C + t * I
        else:
            # ---------- MARGEN FRACCIONAL (legacy) ----------
            a = np.linalg.norm(B - C); b = np.linalg.norm(C - A); c = np.linalg.norm(A - B)
            I = (a*A + b*B + c*C) / (a + b + c + 1e-12)
            t = float(np.clip(margin_frac or 0.0, 0.0, 0.49))
            A2 = (1.0 - t) * A + t * I
            B2 = (1.0 - t) * B + t * I
            C2 = (1.0 - t) * C + t * I

        # --- muestreo uniforme en triángulo reducido ---
        r1 = self.rng.random(); r2 = self.rng.random()
        u = 1.0 - np.sqrt(r1)
        v = np.sqrt(r1) * (1.0 - r2)
        w = np.sqrt(r1) * r2
        P_local = u*A2 + v*B2 + w*C2
        x_local, y_local = float(P_local[0]), float(P_local[1])

        # --- volver a MUNDO + lift ---
        X, Y, Z = axes_from_quat(top_orn)
        lift = self.ball_r + 0.01
        return [
            top_pos[0] + X[0]*x_local + Y[0]*y_local + Z[0]*lift,
            top_pos[1] + X[1]*x_local + Y[1]*y_local + Z[1]*lift,
            top_pos[2] + X[2]*x_local + Y[2]*y_local + Z[2]*lift,
        ]

    # ---------- API pública ----------
    def reset(self, random_ball=True, seed=None, margin_frac=0.2):
        if seed is not None: self.set_seed(seed)

        p.resetBasePositionAndOrientation(self.robot_id, [0,0,0.1], [0,0,0,1])
        for j in (self.j1, self.j2, self.j3):
            p.resetJointState(self.robot_id, j, 0.0)

        if self.ball_id is not None:
            try: p.removeBody(self.ball_id)
            except: pass
            self.ball_id = None

        if random_ball:
            self._spawn_ball(pos=self.sample_point_on_top(margin_frac=None, margin_m=self.ball_r + 0.015))
        else:
            self._spawn_ball()

        for _ in range(10): p.stepSimulation()

    def step(self, target_angles_rad):
        a1, a2, a3 = target_angles_rad
        p.setJointMotorControl2(self.robot_id, self.j1, p.POSITION_CONTROL, targetPosition=float(a1), force=80)
        p.setJointMotorControl2(self.robot_id, self.j2, p.POSITION_CONTROL, targetPosition=float(a2), force=80)
        p.setJointMotorControl2(self.robot_id, self.j3, p.POSITION_CONTROL, targetPosition=float(a3), force=80)
        for _ in range(4): p.stepSimulation()

    def _sync_gui_camera(self, center, eye):
        if not self.use_gui or not self.sync_gui_camera:
            return
        f = np.array(center) - np.array(eye)
        dist = float(np.linalg.norm(f))
        yaw = float(np.degrees(np.arctan2(f[1], f[0])))
        pitch = float(-np.degrees(np.arctan2(f[2], np.linalg.norm(f[:2]))))
        p.resetDebugVisualizerCamera(cameraDistance=dist,
                                     cameraYaw=yaw,
                                     cameraPitch=pitch,
                                     cameraTargetPosition=center)

    def get_rgb(self):
        view, proj, center, eye, _up = camera_view_proj(self.robot_id, self.top, self.img_w, self.img_h)
        self._sync_gui_camera(center, eye)
        rgb, _ = get_rgb_and_seg(self.img_w, self.img_h, view, proj)
        return rgb

    def get_rgb_and_seg(self):
        view, proj, center, eye, _up = camera_view_proj(self.robot_id, self.top, self.img_w, self.img_h)
        self._sync_gui_camera(center, eye)
        return get_rgb_and_seg(self.img_w, self.img_h, view, proj)

    def get_dense_state(self):
        ball_pos,_ = p.getBasePositionAndOrientation(self.ball_id)
        return {"ball_world": np.array(ball_pos, dtype=np.float32)}

    # ---------- helpers privados ----------
    def _pose(self, link):
        ls = p.getLinkState(self.robot_id, link, True)
        return ls[4], ls[5]

    def close(self):
        try: p.disconnect()
        except: pass