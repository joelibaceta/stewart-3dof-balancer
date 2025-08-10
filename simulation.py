import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("robot.urdf", [0, 0, 0.1], useFixedBase=True)

# Obtener posiciones de los 3 tips
tip_links = [3, 7, 11]
tip_positions = []

# Esperar una simulación para que las posiciones estén disponibles
for _ in range(5):
    p.stepSimulation()
    time.sleep(1. / 240.)

for link in tip_links:
    pos, _ = p.getLinkState(robot_id, link)[:2]
    tip_positions.append(pos)

# Usar el primer tip como referencia
ref_pos = tip_positions[0]
# Posición deseada del centro del top_plate
top_plate_pos = [ref_pos[0] - 0.114, ref_pos[1], ref_pos[2] + 0.0001]

# Orientación corregida del STL (rotar -90° en X)
top_orientation = p.getQuaternionFromEuler([-1.57, 0, 0])

# Crear top_plate
top_col = p.createCollisionShape(p.GEOM_MESH, fileName="meshes/triangle_filled.stl", meshScale=[0.001]*3)
top_vis = p.createVisualShape(p.GEOM_MESH, fileName="meshes/triangle_filled.stl", meshScale=[0.001]*3, rgbaColor=[1, 0, 0, 1])
top_plate_id = p.createMultiBody(
    baseMass=0.05,
    baseCollisionShapeIndex=top_col,
    baseVisualShapeIndex=top_vis,
    basePosition=top_plate_pos,
    baseOrientation=top_orientation
)


# Crear constraints como en URDF
for i in range(3):
    tip_pos = tip_positions[i]
    offset = [tip_pos[j] - top_plate_pos[j] for j in range(3)]
    p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=tip_links[i],
        childBodyUniqueId=top_plate_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,  # O usa POINT2POINT si quieres libertad
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=offset
    )


# Cámara
p.resetDebugVisualizerCamera(
    cameraDistance=0.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.1]
)

# Loop
while p.isConnected():
    p.stepSimulation()
    time.sleep(1. / 240.)
