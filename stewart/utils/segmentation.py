import numpy as np

def decode_seg(seg: np.ndarray):
    raw = seg.astype(np.int32)
    obj_id   = raw >> 24
    link_plus = raw & ((1 << 24) - 1)
    link_idx = link_plus - 1
    return obj_id, link_idx