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

def joint_limits(body_id: int, joint_indices) -> list[tuple[float,float]]:
    """
    Devuelve [(lo, hi), ...] para cada joint. Si el URDF no define,
    retorna (-1e9, +1e9).
    """
    L = []
    for j in joint_indices:
        ji = p.getJointInfo(body_id, j)
        lo, hi = float(ji[8]), float(ji[9])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = -1e9, 1e9
        L.append((lo, hi))
    return L

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

def circumcenter_xy(A, B, C):
    ax, ay = A; bx, by = B; cx, cy = C
    d = 2.0 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(d) < 1e-12:
        # fallback: baricentro
        return (ax+bx+cx)/3.0, (ay+by+cy)/3.0
    ax2 = ax*ax + ay*ay; bx2 = bx*bx + by*by; cx2 = cx*cx + cy*cy
    ux = (ax2*(by-cy) + bx2*(cy-ay) + cx2*(ay-by)) / d
    uy = (ax2*(cx-bx) + bx2*(ax-cx) + cx2*(bx-ax)) / d
    return float(ux), float(uy)

def triangle_incenter(A, B, C):
    A = np.asarray(A); B = np.asarray(B); C = np.asarray(C)
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(A - B)
    P = a + b + c + 1e-12
    I = (a*A + b*B + c*C) / P
    return I

def shrink_triangle_towards(A, B, C, center, t: float):
    t = float(np.clip(t, 0.0, 0.49))
    A2 = (1.0 - t) * np.asarray(A) + t * np.asarray(center)
    B2 = (1.0 - t) * np.asarray(B) + t * np.asarray(center)
    C2 = (1.0 - t) * np.asarray(C) + t * np.asarray(center)
    return A2, B2, C2

def offset_triangle_metric(A, B, C, margin_m: float):
    """
    Offset hacia adentro una distancia métrica (en las coords XY locales).
    """
    A = np.asarray(A); B = np.asarray(B); C = np.asarray(C)
    centroid = (A + B + C) / 3.0

    def edge_offset(P, Q, m):
        e = Q - P
        n = np.array([-e[1], e[0]], dtype=np.float64)
        n /= (np.linalg.norm(n) + 1e-12)
        if np.dot(n, centroid - P) < 0:
            n = -n
        d = float(np.dot(n, P) + m)  # n·x = d
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

    area = 0.5 * abs(A2[0]*(B2[1]-C2[1]) + B2[0]*(C2[1]-A2[1]) + C2[0]*(A2[1]-B2[1]))
    if area < 1e-10:
        # fallback a incenter con shrink
        I = triangle_incenter(A, B, C)
        A2, B2, C2 = shrink_triangle_towards(A, B, C, I, 0.3)
    return A2, B2, C2

def sample_uniform_triangle(rng, A, B, C):
    r1 = rng.random(); r2 = rng.random()
    u = 1.0 - np.sqrt(r1)
    v = np.sqrt(r1) * (1.0 - r2)
    w = np.sqrt(r1) * r2
    P = u*np.asarray(A) + v*np.asarray(B) + w*np.asarray(C)
    return float(P[0]), float(P[1])

# ---------- cámara / imagen ----------
def camera_view_proj(robot_id, top_link, img_w, img_h, fov=60, near=0.01, far=2.0):
    # estado del top
    ls = p.getLinkState(robot_id, top_link, True)
    top_pos, top_orn = ls[4], ls[5]

    # links de los tips
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

    # centro del triángulo (circuncentro)
    ux, uy = circumcenter_xy(A, B, C)

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

def get_rgb_and_seg(img_w, img_h, view, proj, rotate180=True, renderer=p.ER_TINY_RENDERER):
    w, h, rgba, depth, seg = p.getCameraImage(
        img_w, img_h,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=renderer,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    )

    # RGB: (h, w, 4) -> (h, w, 3)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[..., :3]

    # Seg: garantizar (h, w)
    seg = np.asarray(seg, dtype=np.int32).reshape(h, w)

    if rotate180:
        rgb = np.ascontiguousarray(np.rot90(rgb, 2))
        seg = np.ascontiguousarray(np.rot90(seg,  2))

    return rgb, seg