# scripts/train_ppo.py
import os, numpy as np, torch
from stewart.envs.steward_sim_env import StewartSimEnv  # tu clase env actual
from stewart.rl.storage import RolloutBuffer
from stewart.rl.models.actor_critic import ActorCriticCNN
from stewart.rl.ppo import PPO

def main(total_steps=200_000, n_steps=1024, img_size=84, device="cuda" if torch.cuda.is_available() else "cpu"):
    env = StewartSimEnv(img_size=img_size, use_gui=False)   # usa tu env
    obs_shape = (img_size, img_size, 3)
    action_dim = 3

    model = ActorCriticCNN(action_dim=action_dim)
    algo = PPO(model, device=device)
    buffer = RolloutBuffer(n_steps, obs_shape, action_dim, device)

    obs, _ = env.reset()
    ep_return, ep_len = 0.0, 0
    step = 0

    while step < total_steps:
        buffer.reset()
        # recolecta n_steps
        for t in range(n_steps):
            obs_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                action, logp, value = model.act(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, term, trunc, _ = env.step(action_np)
            done = term or trunc
            buffer.add(obs, action_np, logp.item(), value.item(), reward, done)

            ep_return += reward; ep_len += 1; step += 1
            obs = next_obs
            if done:
                print(f"episode: return={ep_return:.2f} len={ep_len}")
                obs, _ = env.reset(); ep_return, ep_len = 0.0, 0

        # bootstrap
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(device)
            _, last_value = model.forward(obs_t)
            last_value = last_value.item()
        buffer.compute_returns_adv(last_value)

        losses = algo.update(buffer, epochs=4, batch_size=64)
        pl, vl, ent = np.mean(np.array(losses), axis=0)
        print(f"update: policy={pl:.3f} value={vl:.3f} ent={ent:.3f}")

        # guardado
        if step % 50_000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/ppo_cnn_{step}.pth")

    env.close()

if __name__ == "__main__":
    main()