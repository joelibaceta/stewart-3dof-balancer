# scripts/train_ppo.py
import os, numpy as np, torch, time
from torch.utils.tensorboard import SummaryWriter

from stewart.envs.steward_sim_env import StewartBalanceEnv  # tu clase env actual
from stewart.rl.storage import RolloutBuffer
from stewart.rl.models.actor_critic import ActorCriticCNN
from stewart.rl.ppo import PPO

def main(total_steps=200_000, n_steps=1024, img_size=84,
         device="cuda" if torch.cuda.is_available() else "cpu",
         log_dir="runs/stewart_ppo"):

    # --- setup ---
    env = StewartBalanceEnv(img_size=img_size, use_gui=False)
    obs_shape = (img_size, img_size, 3)
    action_dim = 3

    model = ActorCriticCNN(action_dim=action_dim)
    algo = PPO(model, device=device)
    buffer = RolloutBuffer(n_steps, obs_shape, action_dim, device)

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("hparams",
                    f"total_steps={total_steps}, n_steps={n_steps}, img_size={img_size}, device={device}")

    obs, _ = env.reset()
    ep_return, ep_len = 0.0, 0
    step = 0
    update_idx = 0
    img_log_every = 10_000  # loguear imagen cada N steps

    t0 = time.time()

    while step < total_steps:
        buffer.reset()

        # --- recolecta n_steps ---
        for t in range(n_steps):
            # (1) política
            obs_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(device)  # (1,H,W,3)
            with torch.no_grad():
                action, logp, value = model.act(obs_t)  # asumes que tu .act ya maneja NHWC
            action_np = action.squeeze(0).cpu().numpy()

            # (2) step env
            next_obs, reward, term, trunc, _ = env.step(action_np)
            done = term or trunc

            # (3) buffer
            buffer.add(obs, action_np, logp.item(), value.item(), reward, done)

            # (4) episodios
            ep_return += reward
            ep_len += 1
            step += 1
            obs = next_obs

            # log de imagen ocasional (usa el último obs del bucle)
            if step % img_log_every == 0:
                # obs es HWC uint8; TensorBoard quiere CHW
                img_t = torch.from_numpy(obs).permute(2,0,1)  # (3,H,W)
                writer.add_image("obs/frame", img_t, global_step=step)

            if done:
                # logs episodios
                writer.add_scalar("episode/return", ep_return, global_step=step)
                writer.add_scalar("episode/length", ep_len,   global_step=step)
                print(f"episode: return={ep_return:.2f} len={ep_len}")
                obs, _ = env.reset()
                ep_return, ep_len = 0.0, 0

        # --- bootstrap ---
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(device)
            _, last_value = model.forward(obs_t)
            last_value = last_value.item()
        buffer.compute_returns_adv(last_value)

        # --- update PPO ---
        losses = algo.update(buffer, epochs=4, batch_size=64)
        # losses es lista de [policy_loss, value_loss, entropy] por minibatch (según tu implementación)
        arr = np.array(losses)
        pl, vl, ent = arr[:,0].mean(), arr[:,1].mean(), arr[:,2].mean()

        # logs de update
        writer.add_scalar("loss/policy", pl, global_step=step)
        writer.add_scalar("loss/value",  vl, global_step=step)
        writer.add_scalar("loss/entropy", ent, global_step=step)

        # si tienes clip frac, kl, lr en algo.update() podrías loguearlos aquí también

        update_idx += 1
        print(f"update[{update_idx}] step={step}: policy={pl:.3f} value={vl:.3f} ent={ent:.3f}")

        # guardado
        if step % 50_000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/ppo_cnn_{step}.pth")

    dt = time.time() - t0
    writer.add_text("stats", f"finished in {dt/60:.1f} min")
    writer.close()
    env.close()

if __name__ == "__main__":
    main()