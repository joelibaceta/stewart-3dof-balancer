import gymnasium as gym
from gymnasium import spaces
import numpy as np, pybullet as p, pybullet_data, time

class StewartBalanceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, img_size=84, use_gui=False):
        super().__init__()
        self.img_h = self.img_w = img_size
        self.use_gui = use_gui

        # --- Bullet ---
        if use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT, options="--opengl2")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("robot.urdf", [0,0,0.1], useFixedBase=True)

        # --- Helpers mínimos ---
        def link_index_by_name(body_id, name):
            if name == "base_link": return -1
            for i in range(p.getNumJoints(body_id)):
                if p.getJointInfo(body_id, i)[12].decode() == name:
                    return i
            raise RuntimeError(f"Link '{name}' no encontrado")
        self.top   = link_index_by_name(self.robot_id, "top")
        self.tip2  = link_index_by_name(self.robot_id, "arm2_tip")
        self.tip3  = link_index_by_name(self.robot_id, "arm3_tip")
        self.j_dot1 = self._joint_index("joint_dot1")
        self.j_dot2 = self._joint_index("joint_dot2")
        self.j_dot3 = self._joint_index("joint_dot3")

        # fisica / solver
        p.setPhysicsEngineParameter(numSolverIterations=300, erp=0.85, contactERP=0.85, frictionERP=0.85, globalCFM=1e-7)
        p.setTimeStep(1.0/1000.0)

        # top fricción (liso)
        p.changeDynamics(self.robot_id, self.top, lateralFriction=0.12, rollingFriction=0.0, spinningFriction=0.0, restitution=0.0)

        # desactivar motores en todo menos en 3 servos
        activos = {self.j_dot1, self.j_dot2, self.j_dot3}
        for jid in range(p.getNumJoints(self.robot_id)):
            if jid not in activos:
                p.setJointMotorControl2(self.robot_id, jid, p.VELOCITY_CONTROL, force=0)

        # Constraints (como en tu sim)
        self._create_top_constraints()

        # --- Acción/Observación ---
        self.max_angle = np.array([3.14, 3.14, 3.14], dtype=np.float32)  # rad
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.img_h, self.img_w, 3), dtype=np.uint8)

        # bolita
        self.ball_radius = 0.01
        self.ball_mass   = 0.02
        self.ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_radius)
        self.ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_radius,
                                            rgbaColor=[0.9,0.9,0.9,1.0])

        self.max_steps = 3000
        self.substeps  = 4
        self.step_count = 0

    # ------- util -------
    def _joint_index(self, name):
        for i in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id, i)[1].decode() == name:
                return i
        raise RuntimeError(f"Joint '{name}' no encontrado")

    def _quat_to_axes(self, q):
        m = p.getMatrixFromQuaternion(q)
        X = [m[0], m[3], m[6]]
        Y = [m[1], m[4], m[7]]
        Z = [m[2], m[5], m[8]]
        return X, Y, Z

    def _triangle_circumcenter_world(self):
        def L(name): return p.getLinkState(self.robot_id, self._link(name), True)[4]
        a_w = L("arm1_tip"); b_w = L("arm2_tip"); c_w = L("arm3_tip")
        ax,ay,_ = self._world_to_local_top(a_w)
        bx,by,_ = self._world_to_local_top(b_w)
        cx,cy,_ = self._world_to_local_top(c_w)
        d = 2.0*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))
        if abs(d)<1e-9:
            ux,uy=(ax+bx+cx)/3.0,(ay+by+cy)/3.0
        else:
            ax2ay2=ax*ax+ay*ay; bx2by2=bx*bx+by*by; cx2cy2=cx*cx+cy*cy
            ux=(ax2ay2*(by-cy)+bx2by2*(cy-ay)+cx2cy2*(ay-by))/d
            uy=(ax2ay2*(cx-bx)+bx2by2*(ax-cx)+cx2cy2*(bx-ax))/d
        top_pos, top_orn = self._get_link_pose(self.top)
        X,Y,_ = self._quat_to_axes(top_orn)
        return [top_pos[0]+X[0]*ux+Y[0]*uy,
                top_pos[1]+X[1]*ux+Y[1]*uy,
                top_pos[2]+X[2]*ux+Y[2]*uy]

    def _link(self, name):
        for i in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id, i)[12].decode()==name:
                return i
        raise RuntimeError

    def _get_link_pose(self, link):
        ls = p.getLinkState(self.robot_id, link, True)
        return ls[4], ls[5]

    def _world_to_local_top(self, world_point):
        top_pos, top_orn = self._get_link_pose(self.top)
        inv_pos, inv_orn = p.invertTransform(top_pos, top_orn)
        local, _ = p.multiplyTransforms(inv_pos, inv_orn, world_point, [0,0,0,1])
        return local

    def _create_top_constraints(self):
        # igual que tu script: dos P2P + dos GEAR
        def wpose(lid): return p.getLinkState(self.robot_id, lid, True)[4]
        tip2_w = wpose(self.tip2)
        tip3_w = wpose(self.tip3)

        def world_to_local(body, link, pt):
            ls = p.getLinkState(body, link, True); pos,orn = ls[4], ls[5]
            inv_pos,inv_orn = p.invertTransform(pos,orn)
            local,_ = p.multiplyTransforms(inv_pos,inv_orn, pt, [0,0,0,1])
            return local

        parent_p2 = world_to_local(self.robot_id, self.top,  tip2_w)
        child_p2  = world_to_local(self.robot_id, self.tip2, tip2_w)
        cid2 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip2,
                                  p.JOINT_POINT2POINT, [0,0,0], parent_p2, child_p2)
        p.changeConstraint(cid2, maxForce=20000, erp=0.85)
        gid2 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip2,
                                  p.JOINT_GEAR, [0,0,1], [0,0,0], [0,0,0])
        p.changeConstraint(gid2, gearRatio=1.0, maxForce=300)

        parent_p3 = world_to_local(self.robot_id, self.top,  tip3_w)
        child_p3  = world_to_local(self.robot_id, self.tip3, tip3_w)
        cid3 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip3,
                                  p.JOINT_POINT2POINT, [0,0,0], parent_p3, child_p3)
        p.changeConstraint(cid3, maxForce=20000, erp=0.85)
        gid3 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip3,
                                  p.JOINT_GEAR, [0,0,1], [0,0,0], [0,0,0])
        p.changeConstraint(gid3, gearRatio=1.0, maxForce=300)

    # ------- Gym API -------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # reset mundo
        p.resetBasePositionAndOrientation(self.robot_id, [0,0,0.1], [0,0,0,1])

        # randomiza ligeramente los 3 servos
        base_angles = 0.15*np.random.uniform(-1,1, size=3)
        for jid, a in zip([self.j_dot1,self.j_dot2,self.j_dot3], base_angles):
            p.resetJointState(self.robot_id, jid, a)
            p.setJointMotorControl2(self.robot_id, jid, p.POSITION_CONTROL, targetPosition=a, force=80)

        # re-crear constraints (por si reset borró estados internos)
        p.removeAllUserParameters()  # defensivo
        # (recrear engranajes/p2p)
        self._create_top_constraints()

        # crear/colocar bolita en el centro (ligero ruido)
        if hasattr(self, "ball_id"):
            p.removeBody(self.ball_id)
        center = self._triangle_circumcenter_world()
        top_pos, top_orn = self._get_link_pose(self.top)
        _,_,Z = self._quat_to_axes(top_orn)
        lift = self.ball_radius + 0.01
        spawn = [center[0]+Z[0]*lift, center[1]+Z[1]*lift, center[2]+Z[2]*lift]
        self.ball_id = p.createMultiBody(self.ball_mass, self.ball_col, self.ball_vis, basePosition=spawn)
        p.changeDynamics(self.ball_id, -1, lateralFriction=0.04, rollingFriction=0.0002, spinningFriction=0.0002, restitution=0.15)

        for _ in range(10): p.stepSimulation()

        return self._obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        targets = (action * self.max_angle).tolist()

        p.setJointMotorControl2(self.robot_id, self.j_dot1, p.POSITION_CONTROL, targetPosition=targets[0], force=80)
        p.setJointMotorControl2(self.robot_id, self.j_dot2, p.POSITION_CONTROL, targetPosition=targets[1], force=80)
        p.setJointMotorControl2(self.robot_id, self.j_dot3, p.POSITION_CONTROL, targetPosition=targets[2], force=80)

        for _ in range(self.substeps):
            p.stepSimulation()

        # recompensa = -distancia^2 al centro (en el plano del top)
        reward = -self._dist2_to_center_plane()
        terminated = self._ball_lost()  # cayó/escapó
        truncated  = self.step_count >= self.max_steps
        return self._obs(), float(reward), bool(terminated), bool(truncated), {}

    # ------- observación (imagen) -------
    def _obs(self):
        center = self._triangle_circumcenter_world()
        top_pos, top_orn = self._get_link_pose(self.top)
        _, Y, Z = self._quat_to_axes(top_orn)
        eye = [center[0] + Z[0]*0.15, center[1] + Z[1]*0.15, center[2] + Z[2]*0.15]
        view = p.computeViewMatrix(eye, center, Y)
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0)
        w, h, rgba, *_ = p.getCameraImage(self.img_w, self.img_h, viewMatrix=view, projectionMatrix=proj,
                                          renderer=p.ER_TINY_RENDERER)
        img = np.asarray(rgba, np.uint8).reshape(h, w, 4)[:, :, :3]
        # espejo 180º para que coincida con tu “bottom”
        img = np.ascontiguousarray(np.rot90(img, 2))
        return img

    # ------- métricas / dones -------
    def _dist2_to_center_plane(self):
        center = self._triangle_circumcenter_world()
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        top_pos, top_orn = self._get_link_pose(self.top)
        X,Y,_ = self._quat_to_axes(top_orn)
        rel = [ball_pos[i]-top_pos[i] for i in range(3)]
        x = rel[0]*X[0] + rel[1]*X[1] + rel[2]*X[2]
        y = rel[0]*Y[0] + rel[1]*Y[1] + rel[2]*Y[2]
        cx,cy,_ = self._world_to_local_top(center)
        dx, dy = (x - cx), (y - cy)
        return dx*dx + dy*dy

    def _ball_lost(self):
        aabb_min, aabb_max = p.getAABB(self.robot_id, self.top)
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        # fuera si se aleja mucho horizontalmente o cae por debajo
        return (ball_pos[2] < aabb_min[2] - 0.05) or (np.sqrt(self._dist2_to_center_plane()) > 0.25)

    def close(self):
        try: p.disconnect()
        except: pass