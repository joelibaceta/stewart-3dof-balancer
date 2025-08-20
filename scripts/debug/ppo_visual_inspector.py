import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from stewart.envs.steward_sim_env import StewartBalanceEnv
from stewart.rl.models.actor_critic import ActorCriticCNN

def run_live_inspector(img_size=84, device="cpu", steps=300, fps=10):
    env = StewartBalanceEnv(
        img_size=img_size,
        use_gui=False,
        reward_mode="image",
        render_camera=True,
    )

    # Modelo recién inicializado
    model = ActorCriticCNN(action_dim=3).to(device)
    model.eval()

    obs, _ = env.reset(seed=None, options={"init_joint_deg": 20})
    pause = 1.0 / fps

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4), dpi=100)
    im0 = ax0.imshow(obs)
    im1 = ax1.imshow(np.zeros_like(obs))
    ax0.set_title("Obs PPO"); ax1.set_title("Segmentación")
    txt = ax0.text(5, 12, "", fontsize=8, color="w",
                   bbox=dict(facecolor="black", alpha=0.5, pad=2))
    plt.tight_layout()
    plt.pause(0.01)

    for t in range(steps):
        # Preprocesamiento como PPO
        obs_t = torch.from_numpy(obs).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            action, logp, value = model.act(obs_t)
        act_np = action.squeeze(0).cpu().numpy()
        value_np = value.item()

        obs_next, reward, terminated, truncated, info = env.step(act_np)

        # Segmentación
        rgb, seg = env.core.get_rgb_and_seg()
        obj_id, link_idx = env._decode_seg(seg)
        robot_id = env.core.robot_id
        top_link = env.core.top
        ball_id = env.core.ball_id

        top_mask = (obj_id == robot_id) & (link_idx == top_link)
        ball_mask = (obj_id == ball_id)
        seg_vis = np.zeros_like(rgb)
        seg_vis[top_mask] = [0, 255, 0]
        seg_vis[ball_mask] = [255, 0, 0]

        cscore = info.get("center_score", 0.0)
        inside = info.get("inside_top", False)
        z = info.get("ball_world", [None, None, 0.0])[2]

        im0.set_data(obs)
        im1.set_data(seg_vis)
        txt.set_text(
            f"step={t}\n"
            f"reward={reward:.3f}   inside={inside}\n"
            f"value={value_np:.3f}\n"
            f"action=({act_np[0]:+.2f},{act_np[1]:+.2f},{act_np[2]:+.2f})\n"
            f"center_score={cscore:.3f}   z={z:.3f}"
        )
        plt.pause(pause)

        obs = obs_next
        if terminated or truncated:
            obs, _ = env.reset(seed=None, options={"init_joint_deg": 20})

    env.close()
    plt.show()

if __name__ == "__main__":
    run_live_inspector()
    