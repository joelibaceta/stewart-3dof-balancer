# stewart/sim/core.py
import pybullet as p, pybullet_data, numpy as np
from importlib import resources
class StewartSimCore:
    def __init__(self, use_gui=False, img_size=84):
        self.img_w = self.img_h = img_size
        p.connect(p.GUI if use_gui else p.DIRECT, options="--opengl2")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        p.loadURDF("plane.urdf")
        self.rng = np.random.default_rng()
        assets_root = resources.files("stewart").joinpath("assets")
        assets_root_path = str(assets_root)
        p.setAdditionalSearchPath(assets_root_path)
        robot_urdf = str(assets_root / "robot.urdf")
        self.robot_id = p.loadURDF(robot_urdf, [0,0,0.1], useFixedBase=True)

        self._init_indices()
        self._solver()
        self._create_constraints()
        self._spawn_ball()

    def _init_indices(self):
        def link_by(name):
            if name=="base_link": return -1
            for i in range(p.getNumJoints(self.robot_id)):
                if p.getJointInfo(self.robot_id,i)[12].decode()==name: return i
            raise RuntimeError(name)
        def joint_by(name):
            for i in range(p.getNumJoints(self.robot_id)):
                if p.getJointInfo(self.robot_id,i)[1].decode()==name: return i
            raise RuntimeError(name)
        self.top   = link_by("top")
        self.tip2  = link_by("arm2_tip")
        self.tip3  = link_by("arm3_tip")
        self.j1    = joint_by("joint_dot1")
        self.j2    = joint_by("joint_dot2")
        self.j3    = joint_by("joint_dot3")

        # desactiva motores del resto
        for j in range(p.getNumJoints(self.robot_id)):
            if j not in (self.j1,self.j2,self.j3):
                p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

    def set_seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _solver(self):
        p.setPhysicsEngineParameter(numSolverIterations=300, erp=0.85, contactERP=0.85, frictionERP=0.85, globalCFM=1e-7)
        p.setTimeStep(1/1000)
        p.changeDynamics(self.robot_id, self.top, lateralFriction=0.12, rollingFriction=0.0, spinningFriction=0.0)

    def _create_constraints(self):
        def wpose(l): return p.getLinkState(self.robot_id, l, True)[4]
        def world_to_local(link, pt):
            ls = p.getLinkState(self.robot_id, link, True); pos,orn = ls[4], ls[5]
            inv_pos,inv_orn = p.invertTransform(pos,orn)
            local,_ = p.multiplyTransforms(inv_pos,inv_orn, pt, [0,0,0,1]); return local

        t2 = wpose(self.tip2)
        p2a = world_to_local(self.top,  t2); p2b = world_to_local(self.tip2, t2)
        c2 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip2, p.JOINT_POINT2POINT,[0,0,0],p2a,p2b)
        p.changeConstraint(c2, maxForce=20000, erp=0.85)
        g2 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip2, p.JOINT_GEAR,[0,0,1],[0,0,0],[0,0,0])
        p.changeConstraint(g2, gearRatio=1.0, maxForce=300)

        t3 = wpose(self.tip3)
        p3a = world_to_local(self.top,  t3); p3b = world_to_local(self.tip3, t3)
        c3 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip3, p.JOINT_POINT2POINT,[0,0,0],p3a,p3b)
        p.changeConstraint(c3, maxForce=20000, erp=0.85)
        g3 = p.createConstraint(self.robot_id, self.top, self.robot_id, self.tip3, p.JOINT_GEAR,[0,0,1],[0,0,0],[0,0,0])
        p.changeConstraint(g3, gearRatio=1.0, maxForce=300)

    def _spawn_ball(self, pos=None):
        self.ball_r, self.ball_m = 0.01, 0.02
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.ball_r)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.ball_r, rgbaColor=[0.9,0.9,0.9,1])
        if pos is None:
            aabb_min,aabb_max = p.getAABB(self.robot_id, self.top)
            center = [(aabb_min[i]+aabb_max[i])*0.5 for i in range(3)]
            _,orn = self._pose(self.top); Z = self._axes(orn)[2]
            lift = self.ball_r+0.01
            pos = [center[0]+Z[0]*lift, center[1]+Z[1]*lift, center[2]+Z[2]*lift]
        self.ball_id = p.createMultiBody(self.ball_m, col, vis, basePosition=pos)
        p.changeDynamics(self.ball_id,-1,lateralFriction=0.04,rollingFriction=0.0002,spinningFriction=0.0002,restitution=0.15)

    # ---------- API pública mínima ----------
    def reset(self, random_ball=True, seed=None):
        if seed is not None:
            self.set_seed(seed)
        p.resetBasePositionAndOrientation(self.robot_id,[0,0,0.1],[0,0,0,1])
        for j in (self.j1,self.j2,self.j3):
            p.resetJointState(self.robot_id,j,0.0)
        self._create_constraints()
        p.removeBody(self.ball_id)
        if random_ball:
            self._spawn_ball(pos=self.sample_point_on_top())
        else:
            self._spawn_ball()
        for _ in range(10): p.stepSimulation()

    def step(self, target_angles_rad):
        a1,a2,a3 = target_angles_rad
        p.setJointMotorControl2(self.robot_id, self.j1, p.POSITION_CONTROL, targetPosition=float(a1), force=80)
        p.setJointMotorControl2(self.robot_id, self.j2, p.POSITION_CONTROL, targetPosition=float(a2), force=80)
        p.setJointMotorControl2(self.robot_id, self.j3, p.POSITION_CONTROL, targetPosition=float(a3), force=80)
        for _ in range(4): p.stepSimulation()

    def get_rgb(self):
        center = self._circumcenter_world()
        _,orn = self._pose(self.top); _,Y,Z = self._axes(orn)
        eye = [center[0]+Z[0]*0.15, center[1]+Z[1]*0.15, center[2]+Z[2]*0.15]
        view = p.computeViewMatrix(eye, center, Y)
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.01, 2.0)
        w,h,rgba,*_ = p.getCameraImage(self.img_w,self.img_h,view,proj,renderer=p.ER_TINY_RENDERER)
        img = np.asarray(rgba, np.uint8).reshape(h,w,4)[:,:,:3]
        return np.ascontiguousarray(np.rot90(img,2))

    def get_dense_state(self):
        ball_pos,_ = p.getBasePositionAndOrientation(self.ball_id)
        return {"ball_world": np.array(ball_pos, dtype=np.float32)}

    # ---------- helpers privados ----------
    def _pose(self, link): 
        ls = p.getLinkState(self.robot_id, link, True); return ls[4], ls[5]
    def _axes(self, q):
        m = p.getMatrixFromQuaternion(q); X=[m[0],m[3],m[6]]; Y=[m[1],m[4],m[7]]; Z=[m[2],m[5],m[8]]; return X,Y,Z
    def _circumcenter_world(self):
        def L(name):
            for i in range(p.getNumJoints(self.robot_id)):
                if p.getJointInfo(self.robot_id,i)[12].decode()==name:
                    return p.getLinkState(self.robot_id,i,True)[4]
        a_w,b_w,c_w = L("arm1_tip"),L("arm2_tip"),L("arm3_tip")
        top_pos,top_orn = self._pose(self.top); X,Y,_ = self._axes(top_orn)
        def to_local(w):
            inv_pos,inv_orn = p.invertTransform(top_pos, top_orn)
            (x,y,_),_ = p.multiplyTransforms(inv_pos, inv_orn, w, [0,0,0,1]); return x,y
        ax,ay = to_local(a_w); bx,by = to_local(b_w); cx,cy = to_local(c_w)
        d = 2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))
        if abs(d)<1e-9:
            ux,uy = (ax+bx+cx)/3,(ay+by+cy)/3
        else:
            ax2=ax*ax+ay*ay; bx2=bx*bx+by*by; cx2=cx*cx+cy*cy
            ux=(ax2*(by-cy)+bx2*(cy-ay)+cx2*(ay-by))/d
            uy=(ax2*(cx-bx)+bx2*(ax-cx)+cx2*(bx-ax))/d
        return [top_pos[0]+X[0]*ux+Y[0]*uy, top_pos[1]+X[1]*ux+Y[1]*uy, top_pos[2]+X[2]*ux+Y[2]*uy]

    def sample_point_on_top(self):
        """Devuelve una posición en MUNDO para spawnear la bola:
        punto uniforme dentro del triángulo (en local del top) + pequeño lift.
        """
        # 1) tips en mundo
        a_w = p.getLinkState(self.robot_id, self._find("arm1_tip"), True)[4]
        b_w = p.getLinkState(self.robot_id, self._find("arm2_tip"), True)[4]
        c_w = p.getLinkState(self.robot_id, self._find("arm3_tip"), True)[4]

        # 2) al frame LOCAL del top (plano XY)
        top_pos, top_orn = self._pose(self.top)
        inv_pos, inv_orn = p.invertTransform(top_pos, top_orn)

        def to_local_xy(wpt):
            (x,y,z),_ = p.multiplyTransforms(inv_pos, inv_orn, wpt, [0,0,0,1])
            return x, y

        ax, ay = to_local_xy(a_w)
        bx, by = to_local_xy(b_w)
        cx, cy = to_local_xy(c_w)

        # 3) baricéntricas uniformes dentro del triángulo
        r1 = self.rng.random()
        r2 = self.rng.random()
        u = 1.0 - np.sqrt(r1)
        v = np.sqrt(r1) * (1.0 - r2)
        w = np.sqrt(r1) * r2
        x_local = u*ax + v*bx + w*cx
        y_local = u*ay + v*by + w*cy

        # 4) volver a MUNDO y levantar siguiendo la normal del top
        X, Y, Z = self._axes(top_orn)
        lift = self.ball_r + 0.01
        return [
            top_pos[0] + X[0]*x_local + Y[0]*y_local + Z[0]*lift,
            top_pos[1] + X[1]*x_local + Y[1]*y_local + Z[1]*lift,
            top_pos[2] + X[2]*x_local + Y[2]*y_local + Z[2]*lift,
        ]

    def _find(self, name):
        for i in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id,i)[12].decode()==name: return i
        raise RuntimeError
    def close(self): 
        try: p.disconnect()
        except: pass