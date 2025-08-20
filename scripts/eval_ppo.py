import torch
import numpy as np
import time
import argparse
from stewart.envs.steward_sim_env import StewartBalanceEnv
from stewart.rl.models.actor_critic import ActorCriticCNN

def evaluate(model_path, episodes=5, img_size=84, device="cuda" if torch.cuda.is_available() else "cpu"):
    env = StewartBalanceEnv.make(img_size=img_size, use_gui=True)

    obs_shape = (12, img_size, img_size)
    action_dim = 3

    model = ActorCriticCNN(action_dim=action_dim, obs_shape=obs_shape).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device).float() / 255.0
            with torch.no_grad():
                action, _, _ = model.act(obs_t)
            obs, reward, term, trunc, info = env.step(action.squeeze(0).cpu().numpy())
            ep_return += reward
            done = term or trunc
            time.sleep(1 / 60)

        print(f"[Eval] Episode {ep+1}: Return = {ep_return:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path al modelo PPO entrenado (.pth)")
    parser.add_argument("--episodes", type=int, default=5, help="Cantidad de episodios de evaluación")
    parser.add_argument("--img_size", type=int, default=84, help="Tamaño de imagen de entrada")
    args = parser.parse_args()

    evaluate(args.model, episodes=args.episodes, img_size=args.img_size)