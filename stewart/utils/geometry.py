import numpy as np

def distance_to_triangle_edges(P, A, B, C):
    def point_line_distance(P, Q, R):  
        PQ = np.array(P) - np.array(Q)
        QR = np.array(R) - np.array(Q)
        proj_len = np.dot(PQ, QR) / (np.linalg.norm(QR)**2 + 1e-8)
        proj_len = np.clip(proj_len, 0.0, 1.0)
        closest = np.array(Q) + proj_len * QR
        return np.linalg.norm(np.array(P) - closest)

    return min(
        point_line_distance(P, A, B),
        point_line_distance(P, B, C),
        point_line_distance(P, C, A),
    )

def barycentric(P, A, B, C):
    (x,y),(x1,y1),(x2,y2),(x3,y3) = P,A,B,C
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-12:
        return -1.0, -1.0, -1.0
    u = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    v = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    w = 1.0 - u - v
    return u, v, w