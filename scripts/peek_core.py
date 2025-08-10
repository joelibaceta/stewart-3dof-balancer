import argparse
import numpy as np
import matplotlib.pyplot as plt

from stewart.sim.core import StewartSimCore

def main(gui=False, img_size=84, save=None):
    sim = StewartSimCore(use_gui=gui, img_size=img_size)
    try:
        # frame inicial
        img0 = sim.get_rgb()

        # un paso con ángulos pequeños (rad)
        a = np.array([0.05, -0.04, 0.03], dtype=np.float32)
        sim.step(a)

        # frame tras un paso
        img1 = sim.get_rgb()

        # mostrar
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), dpi=120)
        ax[0].imshow(img0); ax[0].set_title("t=0"); ax[0].axis("off")
        ax[1].imshow(img1); ax[1].set_title("t=1 step"); ax[1].axis("off")
        plt.tight_layout()

        if save:
            plt.savefig(save, bbox_inches="tight")
            print(f"[guardado] {save}")

        plt.show()
    finally:
        sim.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="abre PyBullet GUI")
    parser.add_argument("--img_size", type=int, default=84)
    parser.add_argument("--save", type=str, default=None, help="ruta para guardar la figura")
    args = parser.parse_args()
    main(gui=args.gui, img_size=args.img_size, save=args.save)