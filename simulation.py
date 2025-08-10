import pybullet as p, pybullet_data, time, math
import cv2 as cv
import numpy as np

 
# --- Init ---
p.connect(p.DIRECT, options="--opengl2")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Carga TU URDF (top ahora tiene <collision> en el URDF)
robot_id = p.loadURDF("robot.urdf", [0, 0, 0.1], useFixedBase=True)

# --- Helpers ---
def link_index_by_name(body_id, name):
    if name == "base_link":
        return -1
    for i in range(p.getNumJoints(body_id)):
        if p.getJointInfo(body_id, i)[12].decode() == name:
            return i
    raise RuntimeError(f"Link '{name}' no encontrado")

def joint_index_by_name(body_id, name):
    for i in range(p.getNumJoints(body_id)):
        if p.getJointInfo(body_id, i)[1].decode() == name:
            return i
    raise RuntimeError(f"Joint '{name}' no encontrado")

def world_to_local(body, link, world_point):
    ls = p.getLinkState(body, link, computeForwardKinematics=True)
    link_pos, link_orn = ls[4], ls[5]
    inv_pos, inv_orn = p.invertTransform(link_pos, link_orn)
    local_point, _ = p.multiplyTransforms(inv_pos, inv_orn, world_point, [0,0,0,1])
    return local_point

def get_link_world_pose(body, link):
    ls = p.getLinkState(body, link, computeForwardKinematics=True)
    return ls[4], ls[5]  # pos, orn

def get_body_or_link_pose(body_id, link_index):
    if link_index == -1:
        return p.getBasePositionAndOrientation(body_id)
    ls = p.getLinkState(body_id, link_index, computeForwardKinematics=True)
    return ls[4], ls[5]  # pos, orn

def quat_to_axes(q):
    m = p.getMatrixFromQuaternion(q)
    X = [m[0], m[3], m[6]]
    Y = [m[1], m[4], m[7]]
    Z = [m[2], m[5], m[8]]
    return X, Y, Z

# Si aún no lo tienes:
def get_body_or_link_pose(body_id, link_id):
    if link_id == -1:
        return p.getBasePositionAndOrientation(body_id)
    ls = p.getLinkState(body_id, link_id, computeForwardKinematics=True)
    return ls[4], ls[5]  # pos, orn

def get_top_cam_rgb(img_w=160, img_h=160, height_above=0.25,
                    fov=60, near=0.01, far=2.0, mirror=False,
                    renderer=p.ER_TINY_RENDERER):
    # Centro geométrico del top (independiente del origin del mesh)
    aabb_min, aabb_max = p.getAABB(robot_id, top)
    center = [(aabb_min[i] + aabb_max[i]) * 0.5 for i in range(3)]

    # Usa la orientación del top: Z = normal, Y = "up" de la cámara
    top_pos, top_orn = get_link_world_pose(robot_id, top)
    _, Y, Z = quat_to_axes(top_orn)

    # Ojo encima del top a lo largo de su normal
    eye = [center[0] + Z[0]*height_above,
           center[1] + Z[1]*height_above,
           center[2] + Z[2]*height_above]
    target = center
    up = Y  # evita roll raro y mantiene encuadre estable

    view = p.computeViewMatrix(eye, target, up)
    proj = p.computeProjectionMatrixFOV(
        fov=fov, aspect=float(img_w)/img_h, nearVal=near, farVal=far
    )

    w, h, rgba, depth, seg = p.getCameraImage(
        width=img_w, height=img_h,
        viewMatrix=view, projectionMatrix=proj,
        renderer=renderer
    )

    rgba = np.asarray(rgba, np.uint8).reshape(h, w, 4)
    rgb = rgba[:, :, :3].copy()
    if mirror:
        rgb = cv.rotate(rgb, cv.ROTATE_180)
    return rgb, depth, seg


# def get_bottom_cam_rgb(img_w=128, img_h=128, dist_below = 0.06, fov=60, near=0.01, far=1.5):
#     # Obtener la imagen RGB de la cámara inferior
#     bpos, born = get_body_or_link_pose(robot_id, bottom)
#     bx, by, bz = quat_to_axes(born)

#     eye = [bpos[0] - bz[0]*dist_below,
#            bpos[1] - bz[1]*dist_below,
#            bpos[2] - bz[2]*dist_below]

#     target = [bpos[0] + bz[0]*0.20,
#               bpos[1] + bz[1]*0.20,
#               bpos[2] + bz[2]*0.20]

#     view  = p.computeViewMatrix(eye, target, by)
#     proj  = p.computeProjectionMatrixFOV(fov=fov, aspect = float(img_w)/img_h, nearVal=near, farVal=far)

#     _w, _h, rgba, depth, seg = p.getCameraImage(
#         width=img_w, height=img_h, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER
#     )

#     rgb = np.array(rgba, dtype=np.uint8)[..., :3].copy()
#     return rgb[:,:,:3], depth, seg

cv.namedWindow("Bottom Camera", cv.WINDOW_NORMAL)
cv.resizeWindow("Bottom Camera", 320, 320)
frame_skip = 4
step = 0

# --- Indices de links usados ---
top  = link_index_by_name(robot_id, "top") 
bottom = link_index_by_name(robot_id, "base_link")
tip2 = link_index_by_name(robot_id, "arm2_tip")
tip3 = link_index_by_name(robot_id, "arm3_tip")

# --- Solver más rígido ---
p.setPhysicsEngineParameter(
    numSolverIterations=300,
    erp=0.85, contactERP=0.85, frictionERP=0.85,
    globalCFM=1e-7
)
p.setTimeStep(1.0/1000.0)

# --- Deja que la FK asiente ---
for _ in range(5):
    p.stepSimulation(); time.sleep(1/240)

# --- Constraints vértices 2 y 3: P2P + GEAR (bloquea torsión, quedan 2 GDL) ---
# Vértice 2
tip2_world = p.getLinkState(robot_id, tip2, True)[4]
parent_p2 = world_to_local(robot_id, top,  tip2_world)
child_p2  = world_to_local(robot_id, tip2, tip2_world)
cid2 = p.createConstraint(robot_id, top, robot_id, tip2, p.JOINT_POINT2POINT, [0,0,0], parent_p2, child_p2)
p.changeConstraint(cid2, maxForce=20000, erp=0.85)
gid2 = p.createConstraint(robot_id, top, robot_id, tip2, p.JOINT_GEAR, [0,0,1], [0,0,0], [0,0,0])
p.changeConstraint(gid2, gearRatio=1.0, maxForce=300)

# Vértice 3
tip3_world = p.getLinkState(robot_id, tip3, True)[4]
parent_p3 = world_to_local(robot_id, top,  tip3_world)
child_p3  = world_to_local(robot_id, tip3, tip3_world)
cid3 = p.createConstraint(robot_id, top, robot_id, tip3, p.JOINT_POINT2POINT, [0,0,0], parent_p3, child_p3)
p.changeConstraint(cid3, maxForce=20000, erp=0.85)
gid3 = p.createConstraint(robot_id, top, robot_id, tip3, p.JOINT_GEAR, [0,0,1], [0,0,0], [0,0,0])
p.changeConstraint(gid3, gearRatio=1.0, maxForce=300)

# --- Solo actuamos los servos en los dots; todo lo demás pasivo ---
j_dot1 = joint_index_by_name(robot_id, "joint_dot1")
j_dot2 = joint_index_by_name(robot_id, "joint_dot2")
j_dot3 = joint_index_by_name(robot_id, "joint_dot3")

activos = {j_dot1, j_dot2, j_dot3}
for jid in range(p.getNumJoints(robot_id)):
    if jid not in activos:
        p.setJointMotorControl2(robot_id, jid, p.VELOCITY_CONTROL, force=0)

# Un toque de damping para estabilidad
p.changeDynamics(robot_id, -1, linearDamping=0.02, angularDamping=0.02)
for jid in range(p.getNumJoints(robot_id)):
    p.changeDynamics(robot_id, jid, linearDamping=0.02, angularDamping=0.02)

# --- Fricciones del top (muy liso) ---
p.changeDynamics(robot_id, top,
                 lateralFriction=0.12,
                 rollingFriction=0.0,
                 spinningFriction=0.0,
                 restitution=0.0)

# --- Sliders para los 3 servos de base ---
lim = 3.14
s_dot1 = p.addUserDebugParameter("dot1 [rad]", -lim, lim, 0.0)
s_dot2 = p.addUserDebugParameter("dot2 [rad]", -lim, lim, 0.0)
s_dot3 = p.addUserDebugParameter("dot3 [rad]", -lim, lim, 0.0)

# --- CANICA LIBRE (centrada con AABB del top) ---
ball_radius = 0.01
ball_mass   = 0.02
ball_col = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0.9,0.9,0.9,1.0])

# Centro del top en mundo via AABB (robusto aunque el mesh tenga origin desplazado)
aabb_min, aabb_max = p.getAABB(robot_id, top)
center_world = [(aabb_min[i] + aabb_max[i]) * 0.5 for i in range(3)]

# Normal Z del top
top_pos, top_orn = get_link_world_pose(robot_id, top)
_, _, Z = quat_to_axes(top_orn)

# Spawnear justo por encima del centro real del top, en su normal
lift = ball_radius + 0.01
spawn_pos = [center_world[0] + Z[0]*lift,
             center_world[1] + Z[1]*lift,
             center_world[2] + Z[2]*lift]

ball_id = p.createMultiBody(ball_mass, ball_col, ball_vis, basePosition=spawn_pos)
p.changeDynamics(ball_id, -1,
                 lateralFriction=0.04,
                 rollingFriction=0.0002,
                 spinningFriction=0.0002,
                 restitution=0.15)

# --- Cámara ---
p.resetDebugVisualizerCamera(0.60, 45, -30, [0,0,0.12])

# --- Loop principal: sliders en los dots ---
while p.isConnected():
    step += 1
    if step % frame_skip == 0:
        rgb, _, _ = get_top_cam_rgb(img_w=160, img_h=160, renderer=p.ER_TINY_RENDERER)  # o OPENGL si tienes EGL
        h, w, _ = rgb.shape
        cv.drawMarker(rgb, (w//2, h//2), (0,255,0), markerType=cv.MARKER_CROSS, markerSize=12, thickness=1)
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        cv.imshow("Bottom Camera", bgr)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    #a1 = p.readUserDebugParameter(s_dot1)
    #a2 = p.readUserDebugParameter(s_dot2)
    #a3 = p.readUserDebugParameter(s_dot3)

    #p.setJointMotorControl2(robot_id, j_dot1, p.POSITION_CONTROL, targetPosition=a1, force=80)
    #p.setJointMotorControl2(robot_id, j_dot2, p.POSITION_CONTROL, targetPosition=a2, force=80)
    #p.setJointMotorControl2(robot_id, j_dot3, p.POSITION_CONTROL, targetPosition=a3, force=80)

    p.stepSimulation()
    time.sleep(1/240)
