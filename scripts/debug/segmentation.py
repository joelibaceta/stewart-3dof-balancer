# seg_viewer.py
import argparse, numpy as np, matplotlib.pyplot as plt
from stewart.envs.steward_sim_env import StewartBalanceEnv

def decode_seg(seg: np.ndarray):
    seg = seg.astype(np.int32, copy=False)          # fondo = -1
    obj_id   = seg >> 24
    link_idx = (seg & ((1 << 24) - 1)) - 1          # corrige +1
    return obj_id, link_idx

def outline(mask: np.ndarray):
    # borde simple: dilatar 1px y XOR con el original
    from scipy.ndimage import binary_dilation
    dil = binary_dilation(mask, iterations=1)
    return np.logical_and(dil, ~mask)

def main(img_size=128, use_gui=False, steps=1, save=None, rotate_note=True):
    env = StewartBalanceEnv(img_size=img_size, use_gui=use_gui, action_scale=0.0)
    obs, _ = env.reset()

    # 1) pedir frame y seg
    rgb, seg = env.core.get_rgb_and_seg()  # usa tu helper interno

    # 2) imprimir IDs crudos y decodificados
    vals, cnts = np.unique(seg, return_counts=True)
    print(f"[seg uniques] {len(vals)} valores (head):", list(zip(vals[:10], cnts[:10])))

    obj_id, link_idx = decode_seg(seg)
    robot_id = env.core.robot_id
    top_link = int(env.core.top)
    ball_id  = env.core.ball_id

    print(f"[expect] robot_id={robot_id}  top_link={top_link}  ball_id={ball_id}")
    print("[present] obj_id:", np.unique(obj_id)[:10],
          " link_idx:", np.unique(link_idx)[:10])

    # 3) máscaras
    top_mask  = (obj_id == robot_id) & (link_idx == top_link)
    ball_mask = (obj_id == ball_id)

    if not np.any(top_mask) and (robot_id in np.unique(obj_id)):
        print("[warn] top_mask vacío; usando cualquier link del robot (debug)")
        top_mask = (obj_id == robot_id)

    # 4) visuales
    seg_vis = np.zeros_like(rgb)
    seg_vis[top_mask]  = [0, 255, 0]
    seg_vis[ball_mask] = [255, 0, 0]

    raw_vis = np.zeros(seg.shape, dtype=np.uint8)
    raw_vis[seg != -1] = 255  # todo lo “visible” para la cámara

    # opcional: dibujar contornos para ver bordes claramente
    try:
        b_top  = outline(top_mask)
        b_ball = outline(ball_mask)
        seg_vis[b_top]  = [0, 120, 0]
        seg_vis[b_ball] = [120, 0, 0]
    except Exception:
        pass

    # 5) plot
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=120)
    ax0, ax1, ax2 = axes
    ax0.imshow(obs);     ax0.set_title("RGB"); ax0.axis("off")
    ax1.imshow(seg_vis); ax1.set_title("Seg (top=verde, ball=rojo)"); ax1.axis("off")
    ax2.imshow(raw_vis, cmap="gray"); ax2.set_title("Seg RAW (≠-1)"); ax2.axis("off")

    msg = (f"robot_id={robot_id} top_link={top_link} ball_id={ball_id}\n"
           f"obj_ids:{np.unique(obj_id)[:6]} links:{np.unique(link_idx)[:6]}")
    ax1.set_xlabel(msg, fontsize=7)
    if rotate_note:
        ax0.set_xlabel("Nota: tu helper rota 180° (rotate180=True).", fontsize=7)

    plt.tight_layout()

    if save:
        import imageio
        imageio.imwrite(f"{save}_rgb.png", obs)
        imageio.imwrite(f"{save}_seg.png", seg_vis)
        imageio.imwrite(f"{save}_raw.png", raw_vis)
        print(f"[saved] {save}_rgb.png / {save}_seg.png / {save}_raw.png")

    plt.show()
    env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--save", type=str, default=None, help="prefijo para guardar PNGs")
    ap.add_argument("--no-rotate-note", action="store_true", help="oculta nota de rotación")
    args = ap.parse_args()
    main(img_size=args.img_size, use_gui=args.gui, steps=args.steps,
         save=args.save, rotate_note=not args.no_rotate_note)