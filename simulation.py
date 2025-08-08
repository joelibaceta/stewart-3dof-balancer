import pybullet as p
import pybullet_data
import time

# Conectarse al simulador en modo GUI
p.connect(p.GUI)

# Usar assets predeterminados de PyBullet (como el plano)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Cargar el plano base
plane_id = p.loadURDF("plane.urdf")

# Ajustar gravedad
p.setGravity(0, 0, -9.81)

# Cargar tu modelo URDF (ajusta el nombre si es distinto)
robot_id = p.loadURDF("robot.urdf", [0, 0, 0.1], useFixedBase=True)

top_col = p.createCollisionShape(p.GEOM_MESH, fileName="meshes/triangle_filled.stl", meshScale=[0.001]*3)
top_vis = p.createVisualShape(p.GEOM_MESH, fileName="meshes/triangle_filled.stl", meshScale=[0.001]*3, rgbaColor=[1, 0, 0, 1])

top_plate_id = p.createMultiBody(
    baseMass=0.1,
    baseCollisionShapeIndex=top_col,
    baseVisualShapeIndex=top_vis,
    basePosition=[0, 0, 0.18]
)

for i in range(p.getNumJoints(robot_id)):
    print(i, p.getJointInfo(robot_id, i)[1].decode('utf-8'))

# Cámara (opcional)
p.resetDebugVisualizerCamera(
    cameraDistance=0.5,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

# Simular
while p.isConnected():
    p.stepSimulation()
    time.sleep(1. / 240.)  # Tiempo de simulación