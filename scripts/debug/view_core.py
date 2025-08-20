import time
import argparse
import numpy as np
import pybullet as p

from stewart.sim.core import StewartSimCore

ESC_KEY = 27  # ASCII de ESC

def main(img_size=84, sync_cam=True, fps=60):
    sim = StewartSimCore(use_gui=True, img_size=img_size, sync_gui_camera=sync_cam)
    dt = 1.0 / fps
    try:
        print("[controles]")
        print("  R : reset")
        print("  B : respawn bola (random)")
        print("  ESC / Q : salir\n")
        while True:
            # lee teclado
            keys = p.getKeyboardEvents()
            if keys:
                # salir
                if ESC_KEY in keys or ord('q') in keys or ord('Q') in keys:
                    break
                # reset total
                if ord('r') in keys or ord('R') in keys:
                    sim.reset(random_ball=True)
                # respawn bola sin tocar juntas
                if ord('b') in keys or ord('B') in keys:
                    pos = sim.sample_point_on_top(margin_frac=None, margin_m=sim.ball_r + 0.015)
                    if sim.ball_id is not None:
                        try:
                            p.removeBody(sim.ball_id)
                        except:
                            pass
                        sim.ball_id = None
                    sim._spawn_ball(pos=pos)

            # un step “idle” (sin cambiar ángulos) para ver la física
            sim.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            time.sleep(dt)
    finally:
        sim.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=84)
    ap.add_argument("--no-sync-cam", action="store_true",
                    help="no sincroniza la cámara del GUI con la offscreen")
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()
    main(img_size=args.img_size, sync_cam=not args.no_sync_cam, fps=args.fps)

