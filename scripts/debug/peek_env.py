import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from stewart.envs.steward_sim_env import StewartBalanceEnv

# ---------- helpers ----------
def parse_action(s: str | None):
    if not s:
        return None
    parts = [float(x) for x in s.split(",")]
    assert len(parts) == 3, "La acción debe tener 3 componentes: ax,ay,az"
    return np.array(parts, dtype=np.float32)

def decode_seg(seg: np.ndarray):
    """Decodifica el buffer de segmentación de PyBullet con ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX.
    seg es int32: fondo = -1; bits altos = objectUniqueId; bits bajos = (linkIndex + 1).
    Devuelve (obj_id, link_idx) con link_idx real (ya corregido el +1).
    """
    seg = seg.astype(np.int32, copy=False)
    obj_id   = seg >> 24
    link_idx = (seg & ((1 << 24) - 1)) - 1
    return obj_id, link_idx

def colorize_seg(rgb, obj_id, link_idx, robot_id, top_link, ball_id):
    """Devuelve una imagen RGB con top en verde, bola en rojo, resto negro."""
    top_mask  = (obj_id == robot_id) & (link_idx == top_link)
    ball_mask = (obj_id == ball_id)
    vis = np.zeros_like(rgb)
    vis[top_mask]  = [0, 255, 0]
    vis[ball_mask] = [255, 0, 0]
    return vis, top_mask, ball_mask

# ---------- main ----------
def main(img_size=84, use_gui=False,
         steps=100, fps=30, random_actions=True, action_str=None,
         show_seg=False, seed=None, hold=8, amp=0.5, smooth=0.85):
    env = StewartBalanceEnv(
        img_size=img_size,
        use_gui=use_gui,
        action_scale=0.35,
        spawn_margin_frac=0.2
    )
    rng = np.random.default_rng(seed)
    pause = 1.0 / max(1, fps)

    action_fixed = parse_action(action_str)
    obs, _ = env.reset(seed=seed)

    # Estado del generador de acciones
    rw = np.zeros(3, dtype=np.float32)                # acción “suavizada”
    goal = rng.uniform(-amp, amp, 3).astype(np.float32)
    hold_cnt = 0

    # figura
    if show_seg:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4), dpi=120)
        im0 = ax0.imshow(obs); ax0.set_title("RGB"); ax0.axis("off")
        im1 = ax1.imshow(np.zeros_like(obs)); ax1.set_title("Seg (top=verde, ball=rojo)"); ax1.axis("off")
        txt = ax0.text(5, 12, "", fontsize=8, color="w",
                       bbox=dict(facecolor="black", alpha=0.5, pad=2))
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(4.5, 4), dpi=120)
        im0 = ax0.imshow(obs); ax0.set_title("RGB"); ax0.axis("off")
        txt = ax0.text(5, 12, "", fontsize=8, color="w",
                       bbox=dict(facecolor="black", alpha=0.5, pad=2))

    plt.tight_layout()
    plt.pause(0.001)

    for t in range(1, steps+1):
        # Acción
        if random_actions and action_fixed is None:
            if hold_cnt % max(1, hold) == 0:
                goal = rng.uniform(-amp, amp, 3).astype(np.float32)
            hold_cnt += 1
            rw = smooth * rw + (1 - smooth) * goal
            act = np.clip(rw, -1.0, 1.0)
        else:
            act = action_fixed if action_fixed is not None else np.zeros(3, dtype=np.float32)

        # Paso
        obs, reward, terminated, truncated, info = env.step(act)

        # Overlay de debug
        inside = info.get("inside_top", None)
        cscore = info.get("center_score", 0.0)
        bw = info.get("ball_world", np.array([np.nan, np.nan, np.nan]))

        msg = (
            f"step={t}   r={reward:.4f}\n"
            f"inside={inside}   center_score={cscore:.3f}\n"
            f"action={act[0]:+.2f},{act[1]:+.2f},{act[2]:+.2f}\n"
            f"ball_world z={bw[2]:.3f}\n"
            f"obs uint8  min={obs.min():.0f} max={obs.max():.0f} mean={obs.mean():.1f}"
        )
        im0.set_data(obs)
        txt.set_text(msg)

        if show_seg:
            rgb, seg = env.core.get_rgb_and_seg()
            obj_id, link_idx = decode_seg(seg)

            robot_id = env.core.robot_id
            top_link = env.core.top
            ball_id  = env.core.ball_id

            seg_vis, top_mask, ball_mask = colorize_seg(rgb, obj_id, link_idx, robot_id, top_link, ball_id)
            im1.set_data(seg_vis)

            # — Inspector rápido de IDs únicos (muestra pocos para no saturar) —
            vals, cnts = np.unique(seg, return_counts=True)
            # Para cada valor crudo, decodifica par (obj,link)
            pairs = []
            for v, c in zip(vals[:8], cnts[:8]):  # muestra hasta 8
                o = int(v) >> 24
                l = (int(v) & ((1<<24)-1)) - 1
                pairs.append(f"({o},{l}):{c}")
            ax1.set_xlabel("IDs: " + "  ".join(pairs), fontsize=7)

        plt.pause(pause)

        if terminated or truncated:
            print(f"[done] terminated={terminated} truncated={truncated} en step {t}")
            obs, _ = env.reset(seed=seed)
            rw[:] = 0
            goal = rng.uniform(-amp, amp, 3).astype(np.float32)
            hold_cnt = 0

    env.close()
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=84)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--action", type=str, default=None)
    ap.add_argument("--show-seg", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--hold", type=int, default=8, help="steps que se mantiene la misma dirección")
    ap.add_argument("--amp", type=float, default=0.5, help="amplitud en [-amp, amp]")
    ap.add_argument("--smooth", type=float, default=0.85, help="suavizado hacia la meta (0=rápido, 1=lento)")
    args = ap.parse_args()

    main(
        img_size=args.img_size,
        use_gui=args.gui,
        steps=args.steps,
        fps=args.fps,
        random_actions=args.random,
        action_str=args.action,
        show_seg=args.show_seg,
        seed=args.seed,
        hold=args.hold,
        amp=args.amp,
        smooth=args.smooth,
    )