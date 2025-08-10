import gymnasium as gym
import numpy as np
import pybullet as p, pybullet_data
import cv2 as cv
import time, math

class BalancerEnv(gym.Env):
    """
    Obs = stack de 4 frames (grayscale) 84x84 (por defecto).
    Action = Δθ para [joint_dot1, joint_dot2, joint_dot3] en rad (Box[-0.03, 0.03]).
    Recompensa = -||ball_xy - center_xy|| (en el frame local del 'top') - 1e-3*||Δθ||^2.
    Done si la canica cae (z muy baja o sale del AABB extendido) o max_steps.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    # ======================== CONFIG ========================
    IMG = 84
    FRAME_STACK = 4
    ACT_DELTA = 0.03
    MAX_STEPS = 2000
    DT = 1.0/240.0
    SUBSTEPS = 6                    # steps de física por step RL
    JOINT_LIMIT = 1.57              # +/- 90°
    BALL_RADIUS = 0.01
    BALL_MASS   = 0.02
    TOP_FRICTION = dict(lateralFriction=0.12, rollingFriction=0.0, spinningFriction=0.0, restitution=0.0)
    BALL_FRICTION = dict(lateralFriction=0.04, rollingFriction=0.0002, spinningFriction=0.0002, restitution=0.15)
    # ========================================================

    def __init__(self, render_mode=None, img_size: int = IMG, frame_stack: int = FRAME_STACK, headless=True):
        super().__init__()
        self.render_mode = render_mode
        self.img_size = img_size
        self.frame_stack = frame_stack

        # Acción: Δθ para 3 servos (dot1/2/3)
        high = np.full((3,), self.ACT_DELTA, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

        # Observación: stack de frames (grayscale)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(self.img_size, self.img_size, self.frame_stack), dtype=np.uint8
        )

        # Conexión headless
        if headless:
            p.connect(p.DIRECT, options="--opengl2")
        else:
            p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.DT)
        p.setPhysicsEngineParameter(
            numSolverIterations=300, erp=0.85, contactERP=0.85, frictionERP=0.85, globalCFM=1e-7
        )

        # Para targets absolutos (θ) de cada dot; se actualiza con Δθ
        self.theta = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        # IDs y caches
        self.robot_id = None
        self.top = None
        self.tip2 = None
        self.tip3 = None
        self.bottom = None
        self.j_dot = None
        self.ball_id = None

        self.frames = None  # buffer para frame stacking

        self._setup_world()

    # ----------------------- Helpers originales adaptados -----------------------
    @staticmethod
    def _link_index_by_name(body_id, name):
        if name == "base_link":
            return -1
        for i in range(p.getNumJoints(body_id)):
            if p.getJointInfo(body_id, i)[12].decode() == name:
                return i
        raise RuntimeError(f"Link '{name}' no encontrado")

    @staticmethod
    def _joint_index_by_name(body_id, name):
        for i in range(p.getNumJoints(body_id)):
            if p.getJointInfo(body_id, i)[1].decode() == name:
                return i
        raise RuntimeError(f"Joint '{name}' no encontrado")

    @staticmethod
    def _get_link_world_pose(body, link):
        ls = p.getLinkState(body, link, computeForwardKinematics=True)
        return ls[4], ls[5]

    @staticmethod
    def _world_to_local(body, link, world_point):
        ls = p.getLinkState(body, link, computeForwardKinematics=True)
        link_pos, link_orn = ls[4], ls[5]
        inv_pos, inv_orn = p.invertTransform(link_pos, link_orn)
        local_point, _ = p.multiplyTransforms(inv_pos, inv_orn, world_point, [0,0,0,1])
        return local_point

    @staticmethod
    def _quat_to_axes(q):
        m = p.getMatrixFromQuaternion(q)
        X = [m[0], m[3], m[6]]
        Y = [m[1], m[4], m[7]]
        Z = [m[2], m[5], m[8]]
        return X, Y, Z

    # circuncentro del triángulo de tips, en mundo
    def _triangle_circumcenter_world(self):
        a_w = p.getLinkState(self.robot_id, self._link_index_by_name(self.robot_id,"arm1_tip"), True)[4]
        b_w = p.getLinkState(self.robot_id, self._link_index_by_name(self.robot_id,"arm2_tip"), True)[4]
        c_w = p.getLinkState(self.robot_id, self._link_index_by_name(self.robot_id,"arm3_tip"), True)[4]

        ax, ay, _ = self._world_to_local(self.robot_id, self.top, a_w)
        bx, by, _ = self._world_to_local(self.robot_id, self.top, b_w)
        cx, cy, _ = self._world_to_local(self.robot_id, self.top, c_w)

        d = 2.0 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
        if abs(d) < 1e-9:
            ux, uy = (ax+bx+cx)/3.0, (ay+by+cy)/3.0
        else:
            ax2ay2 = ax*ax + ay*ay
            bx2by2 = bx*bx + by*by
            cx2cy2 = cx*cx + cy*cy
            ux = (ax2ay2*(by-cy) + bx2by2*(cy-ay) + cx2cy2*(ay-by)) / d
            uy = (ax2ay2*(cx-bx) + bx2by2*(ax-cx) + cx2cy2*(bx-ax)) / d

        top_pos, top_orn = self._get_link_world_pose(self.robot_id, self.top)
        X, Y, _ = self._quat_to_axes(top_orn)
        center_world = [
            top_pos[0] + X[0]*ux + Y[0]*uy,
            top_pos[1] + X[1]*ux + Y[1]*uy,
            top_pos[2] + X[2]*ux + Y[2]*uy,
        ]
        return center_world

    # imagen de la cámara sobre el top (centrada en el circuncentro)
    def _get_top_cam_rgb(self, img_w=None, img_h=None, height_above=0.15, fov=50,
                         near=0.01, far=2.0, mirror=True, renderer=p.ER_TINY_RENDERER):
        img_w = img_w or self.img_size
        img_h = img_h or self.img_size

        center = self._triangle_circumcenter_world()

        top_pos, top_orn = self._get_link_world_pose(self.robot_id, self.top)
        _, Y, Z = self._quat_to_axes(top_orn)

        eye = [center[0] + Z[0]*height_above,
               center[1] + Z[1]*height_above,
               center[2] + Z[2]*height_above]
        target = center
        up = Y

        view = p.computeViewMatrix(eye, target, up)
        proj = p.computeProjectionMatrixFOV(fov=fov, aspect=float(img_w)/img_h, nearVal=near, farVal=far)

        w, h, rgba, depth, seg = p.getCameraImage(
            width=img_w, height=img_h, viewMatrix=view, projectionMatrix=proj, renderer=renderer
        )
        rgba = np.asarray(rgba, np.uint8).reshape(h, w, 4)
        rgb = rgba[:, :, :3].copy()
        if mirror:
            rgb = cv.rotate(rgb, cv.ROTATE_180)
        return rgb

    # distancia 2D (en el frame del top) entre bola y circuncentro
    def _ball_dist_to_center(self):
        # bola en mundo
        bpos, _ = p.getBasePositionAndOrientation(self.ball_id)
        # al frame local del top
        bx, by, _ = self._world_to_local(self.robot_id, self.top, bpos)
        # circuncentro en local
        center_w = self._triangle_circumcenter_world()
        cx, cy, _ = self._world_to_local(self.robot_id, self.top, center_w)
        return math.hypot(bx - cx, by - cy)

    def _ball_fell(self):
        # se considera “caída” si z < umbral o sale mucho del AABB del top
        bpos, _ = p.getBasePositionAndOrientation(self.ball_id)
        if bpos[2] < 0.02:
            return True
        aabb_min, aabb_max = p.getAABB(self.robot_id, self.top)
        margin = 0.06
        if not (aabb_min[0]-margin <= bpos[0] <= aabb_max[0]+margin and
                aabb_min[1]-margin <= bpos[1] <= aabb_max[1]+margin):
            return True
        return False

    # ----------------------- creación / reset del mundo -----------------------
    def _setup_world(self):
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("robot.urdf", [0,0,0.1], useFixedBase=True)

        self.top    = self._link_index_by_name(self.robot_id, "top")
        self.tip2   = self._link_index_by_name(self.robot_id, "arm2_tip")
        self.tip3   = self._link_index_by_name(self.robot_id, "arm3_tip")
        self.bottom = self._link_index_by_name(self.robot_id, "base_link")

        # Constraints (P2P + GEAR) como en tu script
        # -- tip2
        tip2_world = p.getLinkState(self.robot_id, self.tip2, True)[4]
        parent_p2 = self._world_to_local(self.robot_id, self.top,  tip2_world)
        child_p2  = self._world_to_local(self.robot_id, self.tip2, tip2_world)
        cid2 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip2,
                                  p.JOINT_POINT2POINT, [0,0,0], parent_p2, child_p2)
        p.changeConstraint(cid2, maxForce=20000, erp=0.85)
        gid2 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip2,
                                  p.JOINT_GEAR, [0,0,1], [0,0,0], [0,0,0])
        p.changeConstraint(gid2, gearRatio=1.0, maxForce=300)
        # -- tip3
        tip3_world = p.getLinkState(self.robot_id, self.tip3, True)[4]
        parent_p3 = self._world_to_local(self.robot_id, self.top,  tip3_world)
        child_p3  = self._world_to_local(self.robot_id, self.tip3, tip3_world)
        cid3 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip3,
                                  p.JOINT_POINT2POINT, [0,0,0], parent_p3, child_p3)
        p.changeConstraint(cid3, maxForce=20000, erp=0.85)
        gid3 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip3,
                                  p.JOINT_GEAR, [0,0,1], [0,0,0], [0,0,0])
        p.changeConstraint(gid3, gearRatio=1.0, maxForce=300)

        # motores: solo dots
        j_dot1 = self._joint_index_by_name(self.robot_id, "joint_dot1")
        j_dot2 = self._joint_index_by_name(self.robot_id, "joint_dot2")
        j_dot3 = self._joint_index_by_name(self.robot_id, "joint_dot3")
        self.j_dot = [j_dot1, j_dot2, j_dot3]

        active = set(self.j_dot)
        for jid in range(p.getNumJoints(self.robot_id)):
            if jid not in active:
                p.setJointMotorControl2(self.robot_id, jid, p.VELOCITY_CONTROL, force=0)

        # damping
        p.changeDynamics(self.robot_id, -1, linearDamping=0.02, angularDamping=0.02)
        for jid in range(p.getNumJoints(self.robot_id)):
            p.changeDynamics(self.robot_id, jid, linearDamping=0.02, angularDamping=0.02)

        # fricción top
        p.changeDynamics(self.robot_id, self.top, **self.TOP_FRICTION)

        # Deja asentar
        for _ in range(5):
            p.stepSimulation()

        # bola
        self._spawn_ball()

        # estado motores
        self.theta[:] = 0.0
        for i, jid in enumerate(self.j_dot):
            p.setJointMotorControl2(self.robot_id, jid, p.POSITION_CONTROL, targetPosition=float(self.theta[i]), force=80)

        self.step_count = 0

    def _spawn_ball(self):
        if self.ball_id is not None:
            p.removeBody(self.ball_id)
        ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.BALL_RADIUS)
        ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.BALL_RADIUS, rgbaColor=[0.9,0.9,0.9,1.0])

        # centro del top (AABB) + levantar en su normal
        aabb_min, aabb_max = p.getAABB(self.robot_id, self.top)
        center_world = [(aabb_min[i] + aabb_max[i]) * 0.5 for i in range(3)]
        top_pos, top_orn = self._get_link_world_pose(self.robot_id, self.top)
        _, _, Z = self._quat_to_axes(top_orn)
        lift = self.BALL_RADIUS + 0.01
        spawn_pos = [center_world[0]+Z[0]*lift, center_world[1]+Z[1]*lift, center_world[2]+Z[2]*lift]

        self.ball_id = p.createMultiBody(self.BALL_MASS, ball_col, ball_vis, basePosition=spawn_pos)
        p.changeDynamics(self.ball_id, -1, **self.BALL_FRICTION)

    # ----------------------- Gym API -----------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_world()

        # pequeño ruido inicial de la bola
        bpos, born = p.getBasePositionAndOrientation(self.ball_id)
        jitter = np.random.uniform(-0.005, 0.005, size=2)
        bpos = [bpos[0] + jitter[0], bpos[1] + jitter[1], bpos[2]]
        p.resetBasePositionAndOrientation(self.ball_id, bpos, born)

        # buffer de frames
        rgb = self._get_top_cam_rgb(img_w=self.img_size, img_h=self.img_size, height_above=0.15, fov=50)
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        self.frames = [gray.copy() for _ in range(self.frame_stack)]

        self.step_count = 0
        return self._obs(), {}

    def _obs(self):
        return np.stack(self.frames, axis=-1)

    def step(self, action):
        # acción como Δθ
        action = np.clip(np.asarray(action, dtype=np.float32), -self.ACT_DELTA, self.ACT_DELTA)
        self.theta = np.clip(self.theta + action, -self.JOINT_LIMIT, self.JOINT_LIMIT)

        # aplica targets
        for i, jid in enumerate(self.j_dot):
            p.setJointMotorControl2(self.robot_id, jid, p.POSITION_CONTROL,
                                    targetPosition=float(self.theta[i]), force=80)

        # integra física
        for _ in range(self.SUBSTEPS):
            p.stepSimulation()

        # reward
        dist = self._ball_dist_to_center()
        reward = -dist - 1e-3 * float(np.dot(action, action))

        # done?
        self.step_count += 1
        terminated = self._ball_fell()
        truncated = self.step_count >= self.MAX_STEPS

        # nueva observación (frame stack)
        rgb = self._get_top_cam_rgb(img_w=self.img_size, img_h=self.img_size, height_above=0.15, fov=50)
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        self.frames.pop(0); self.frames.append(gray)

        info = {"dist_to_center": dist}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        # devuelve un frame RGB grande si lo pides
        rgb = self._get_top_cam_rgb(img_w=256, img_h=256, height_above=0.15, fov=50)
        return rgb

    def close(self):
        if p.isConnected():
            p.disconnect()