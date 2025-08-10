# stewart/sim/utils.py
import numpy as np
import pybullet as p

# ---------- URDF helpers ----------
def link_index_by_name(body_id: int, name: str) -> int:
    if name == "base_link":
        return -1
    for i in range(p.getNumJoints(body_id)):
        if p.getJointInfo(body_id, i)[12].decode() == name:
            return i
    raise RuntimeError(f"link not found: {name}")

def joint_index_by_name(body_id: int, name: str) -> int:
    for i in range(p.getNumJoints(body_id)):
        if p.getJointInfo(body_id, i)[1].decode() == name:
            return i
    raise RuntimeError(f"joint not found: {name}")

# ---------- frames y geometría ----------
def axes_from_quat(q):
    m = p.getMatrixFromQuaternion(q)
    X = np.array([m[0], m[3], m[6]], dtype=np.float64)
    Y = np.array([m[1], m[4], m[7]], dtype=np.float64)
    Z = np.array([m[2], m[5], m[8]], dtype=np.float64)
    return X, Y, Z

def to_local_xy(top_pos, top_orn, world_pt):
    inv_pos, inv_orn = p.invertTransform(top_pos, top_orn)
    (x, y, _), _ = p.multiplyTransforms(inv_pos, inv_orn, world_pt, [0,0,0,1])
    return float(x), float(y)

def triangle_incenter(A, B, C):
    A = np.asarray(A); B = np.asarray(B); C = np.asarray(C)
    a = np.linalg.norm(B - C)  # lado opuesto a A
    b = np.linalg.norm(C - A)  # opuesto a B
    c = np.linalg.norm(A - B)  # opuesto a C
    P = a + b + c + 1e-12
    I = (a*A + b*B + c*C) / P
    return I

def shrink_triangle_towards(A, B, C, center, t: float):
    t = float(np.clip(t, 0.0, 0.49))
    A2 = (1.0 - t) * A + t * center
    B2 = (1.0 - t) * B + t * center
    C2 = (1.0 - t) * C + t * center
    return A2, B2, C2

def sample_uniform_triangle(rng, A, B, C):
    r1 = rng.random(); r2 = rng.random()
    u = 1.0 - np.sqrt(r1)
    v = np.sqrt(r1) * (1.0 - r2)
    w = np.sqrt(r1) * r2
    P = u*np.asarray(A) + v*np.asarray(B) + w*np.asarray(C)
    return float(P[0]), float(P[1])

# ---------- cámara / imagen ----------
# stewart/sim/utils.py
import numpy as np
import pybullet as p
from .utils import to_local_xy, axes_from_quat  # si ya están en utils; si no, copia sus defs aquí

def camera_view_proj(robot_id, top_link, img_w, img_h, fov=60, near=0.01, far=2.0):
    # estado del top
    top_pos, top_orn = p.getLinkState(robot_id, top_link, True)[4], p.getLinkState(robot_id, top_link, True)[5]

    # busca tips
    tips = []
    for name in ("arm1_tip", "arm2_tip", "arm3_tip"):
        for i in range(p.getNumJoints(robot_id)):
            if p.getJointInfo(robot_id, i)[12].decode() == name:
                tips.append(p.getLinkState(robot_id, i, True)[4])
                break

    # a local XY
    A = np.array(to_local_xy(top_pos, top_orn, tips[0]))
    B = np.array(to_local_xy(top_pos, top_orn, tips[1]))
    C = np.array(to_local_xy(top_pos, top_orn, tips[2]))

    # circuncentro en local
    ax, ay = A; bx, by = B; cx, cy = C
    d = 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(d) < 1e-12:
        ux, uy = (ax+bx+cx)/3, (ay+by+cy)/3
    else:
        ax2=ax*ax+ay*ay; bx2=bx*bx+by*by; cx2=cx*cx+cy*cy
        ux = (ax2*(by-cy) + bx2*(cy-ay) + cx2*(ay-by)) / d
        uy = (ax2*(cx-bx) + bx2*(ax-cx) + cx2*(bx-ax)) / d

    X, Y, Z = axes_from_quat(top_orn)
    center = (top_pos[0] + X[0]*ux + Y[0]*uy,
              top_pos[1] + X[1]*ux + Y[1]*uy,
              top_pos[2] + X[2]*ux + Y[2]*uy)
    eye = (center[0] + Z[0]*0.15,
           center[1] + Z[1]*0.15,
           center[2] + Z[2]*0.15)

    view = p.computeViewMatrix(eye, center, Y.tolist())
    proj = p.computeProjectionMatrixFOV(fov, 1.0, near, far)
    return view, proj, center, eye, Y

def get_rgb_and_seg(img_w, img_h, view, proj):
    w, h, rgba, depth, seg = p.getCameraImage(
        img_w, img_h, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    )
    rgb = np.asarray(rgba, np.uint8).reshape(h, w, 4)[:, :, :3]
    rgb = np.ascontiguousarray(np.rot90(rgb, 2))
    seg = np.asarray(seg, np.int32)
    seg = np.ascontiguousarray(np.rot90(seg, 2))
    return rgb, seg