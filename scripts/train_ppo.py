"""
Entrenamiento PPO para StewartBalanceEnv usando imágenes RGB como observación.

Este script entrena un agente PPO para balancear una bola sobre una plataforma
tipo Stewart. Utiliza observaciones visuales (84x84 RGB) y una política 
actor-crítico convolucional (CNN). Los resultados se loguean en TensorBoard.

Requisitos:
- entornos Gymnasium personalizados (`StewartBalanceEnv`)
- PPO con GAE y clipping
- Imagenes en CHW (usando wrapper `ToCHW`)
- Rollout buffer externo
"""
import cv2
import numpy as np
import random
import os, json, time, numpy as np, torch
from torch.utils.tensorboard import SummaryWriter

from stewart.envs.stewart_env import StewartBalanceEnv
from stewart.buffers.rollout_buffer import RolloutBuffer
from stewart.models.actor_critic_cnn import ActorCriticCNN
from stewart.agent.ppo import PPO
from gymnasium.wrappers import FrameStackObservation

def _make_logdir(log_dir_base="runs/stewart_ppo", run_name=None):
    """
    Crea una carpeta de logs única basada en timestamp o un nombre personalizado.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = ts if run_name is None else run_name
    log_dir = os.path.join(log_dir_base, run_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def set_global_seed(seed):
    """
    Configura la semilla global para reproducibilidad.
    """
    if seed is None: return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mostrar_observacion(obs_tensor, wait=200):
    """
    Muestra la última imagen RGB de un stack de frames (esperado: (B, 12, H, W)).
    """
    obs_tensor = obs_tensor.squeeze(0)  # (12,H,W)

    if obs_tensor.ndim != 3:
        print("Error: la observación no tiene 3 dimensiones (C,H,W)")
        return

    c, h, w = obs_tensor.shape
    if c != 12:
        print(f"Advertencia: número de canales inesperado ({c}), se esperaba 12 para stack de 4 frames RGB")

    # Tomar solo los últimos 3 canales (último frame)
    obs_tensor = obs_tensor[-3:, :, :]  # (3,H,W)

    # Convertir a numpy y reescalar
    obs_np = obs_tensor.detach().cpu().numpy()
    obs_np = np.transpose(obs_np, (1, 2, 0))  # (H,W,C)
    obs_np = np.clip(obs_np * 255.0, 0, 255).astype(np.uint8)

    # Convertir de RGB a BGR
    bgr = cv2.cvtColor(obs_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("Input a CNN", bgr)
    cv2.waitKey(wait)


def main(
    total_steps=500_000,
    n_steps=1024,
    img_size=84,
    seed=random.randint(0, 2**32 - 1),
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_dir_base="runs/stewart_ppo",
    run_name=None,
    print_every=1000,   # << NUEVO
):
    """
    Entrena un agente PPO en el entorno StewartBalanceEnv.

    Argumentos:
        total_steps     (int): número total de pasos de entrenamiento.
        n_steps         (int): pasos por rollout antes de cada update.
        img_size        (int): tamaño de imagen cuadrado (HxW).
        seed            (int): semilla para reproducibilidad.
        device         (str): "cuda" o "cpu".
        log_dir_base   (str): carpeta base para guardar logs.
        run_name       (str): nombre opcional del experimento.
        print_every     (int): cada cuántos pasos imprimir info.
    """

    log_dir = _make_logdir(log_dir_base, run_name)
    set_global_seed(seed)


    # ------------------------------
    # Inicialización del entorno
    # ------------------------------
    env = StewartBalanceEnv.make()

    obs_shape = (12, img_size, img_size)
    action_dim = 3

    # ------------------------------
    # Modelo + algoritmo PPO
    # ------------------------------
    model = ActorCriticCNN(action_dim=action_dim, obs_shape=obs_shape).to(device)
    algo = PPO(model=model, obs_shape=obs_shape, action_dim=action_dim, device=device, total_steps=total_steps)
    buffer = RolloutBuffer(n_steps, obs_shape, action_dim, device)

    # ------------------------------
    # Logger
    # ------------------------------
    writer = SummaryWriter(log_dir=log_dir)
    hparams = dict(total_steps=total_steps, n_steps=n_steps, img_size=img_size,
                   seed=seed, device=device, log_dir=log_dir, print_every=print_every)
    writer.add_text("hparams", json.dumps(hparams, indent=2))
    with open(os.path.join(log_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=2)

    
    # ------------------------------
    # Bucle principal de entrenamiento
    # ------------------------------
    obs, _ = env.reset(seed=seed, options={"init_joint_deg": 20})
    ep_return, ep_len = 0.0, 0
    ep_idx = 0     
    step = 0
    update_idx = 0
    img_log_every = 10_000
    t0 = time.time()

    while step < total_steps:
        buffer.reset()

        for _ in range(n_steps):
            # (1) política
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)  # (1,C,H,W)
            obs_t = obs_t.float().div(255.0)

            with torch.no_grad():
                action, logp, value = model.act(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

            writer.add_scalar("value/estimate", float(value.item()), global_step=step+1)

            # (2) env step
            next_obs, reward, term, trunc, info = env.step(action_np)
            done = bool(term or trunc)

            buffer.add(obs, action_np, float(logp.item()), float(value.item()), float(reward), done)

            # --- DEBUG / LOG: estado
            inside = info.get("inside_top", None)
            cscore = float(info.get("center_score", 0.0))
            ball_w = info.get("ball_world", np.array([np.nan, np.nan, np.nan]))
            q = info.get("joint_angles_rad", [np.nan, np.nan, np.nan]) 

            # prints periódicos o al final de episodio
            if done or (print_every and ((step + 1) % print_every == 0)):
                print(
                    f"[s={step+1}] r={reward:+.3f} ep_ret={ep_return:+.2f} ep_len={ep_len} "
                    f"inside={inside} cscore={cscore:.3f} "
                    f"ball_w=({ball_w[0]:+.3f},{ball_w[1]:+.3f},{ball_w[2]:+.3f}) "
                    f"q(rad)=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f}) "
                    f"act(scaled)=({action_np[0]:+.3f},{action_np[1]:+.3f},{action_np[2]:+.3f})"
                )

            # TB scalars útiles
            writer.add_scalar("state/inside_top", float(bool(inside)), global_step=step+1)
            writer.add_scalar("state/center_score", cscore, global_step=step+1)
            writer.add_scalar("joints/j1_rad", float(q[0]), global_step=step+1)
            writer.add_scalar("joints/j2_rad", float(q[1]), global_step=step+1)
            writer.add_scalar("joints/j3_rad", float(q[2]), global_step=step+1)
            writer.add_scalar("action/a1_scaled", float(action_np[0]), step+1)
            writer.add_scalar("action/a2_scaled", float(action_np[1]), step+1)
            writer.add_scalar("action/a3_scaled", float(action_np[2]), step+1)

            # (4) episodios
            ep_return += reward
            ep_len += 1
            step += 1
            obs = next_obs

            if step % img_log_every == 0:
                last3 = torch.from_numpy(obs[-3:, :, :])  # (3,H,W), uint8
                writer.add_image("obs/last_frame", last3, global_step=step, dataformats="CHW")

            if done:
                ep_idx += 1
                
                writer.add_scalar("episode/return", ep_return, global_step=step)
                writer.add_scalar("episode/length", ep_len,   global_step=step)
                print(f"episode {ep_idx}: return={ep_return:.2f} len={ep_len}")
                obs, _ = env.reset(seed=None, options={"init_joint_deg": 20})
                ep_return, ep_len = 0.0, 0

            if step >= total_steps:
                break

        # --- bootstrap y GAE ---
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)  # (1,C,H,W)
            obs_t = obs_t.float().div(255.0)
            
            _, _, last_value = model.forward(obs_t)
            last_value = float(last_value.item())
        buffer.compute_returns_adv(last_value)
        algo.last_mean_advantage = float(buffer.advantages[:buffer.ptr].mean())

        # --- update PPO ---
        losses, ent_coef_used = algo.update(buffer=buffer)
        arr = np.array(losses)
        pl, vl, ent = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
        grad = arr[:, 3].mean()

        writer.add_scalar("loss/policy",  pl,  global_step=step)
        writer.add_scalar("loss/value",   vl,  global_step=step)
        writer.add_scalar("loss/entropy", ent, global_step=step)
        writer.add_scalar("loss/entropy_coef", ent_coef_used, global_step=step)
        writer.add_scalar("loss/grad_norm", grad, global_step=step)
        writer.add_scalar("advantage/mean", algo.last_mean_advantage, global_step=step)
        writer.add_scalar("advantage/std", buffer.advantages[:buffer.ptr].std(), global_step=step)

        update_idx += 1
        print(f"update[{update_idx}] step={step}: policy={pl:.3f} value={vl:.3f} ent={ent:.3f}")

        if step % 50_000 == 0:
            ckpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ppo_cnn_{step}.pth"))

    dt = time.time() - t0
    writer.add_text("stats", f"finished in {dt/60:.1f} min")
    writer.close()
    env.close()

if __name__ == "__main__":
    main()