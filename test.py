import gymnasium as gym
import torch
import numpy as np
import os
import glob
import imageio
import module
import argparse
import yaml
from types import SimpleNamespace

os.environ["MUJOCO_GL"] = "egl"

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="Humanoid-v5")
parser.add_argument("--ckpt", type=str, default="06-06-25_19:08:43")
args = parser.parse_args()

# load configuration
cfg_path = f"config/{args.env}.yaml"
assert os.path.exists(cfg_path), f"config file not found: {cfg_path}"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg)


# locate the best checkpoint
extract_reward = lambda f: float(f.split("reward")[-1].replace(".pt", ""))
ckpt_dir = os.path.join(cfg.ckpt_root_dir, args.env, args.ckpt) 
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch*.pt")))
assert ckpts, "No checkpoints found in directory."
best_ckpt = max(ckpts, key=extract_reward)
print(f"\nLoading checkpoint: {best_ckpt}\n")

# environment setup
env = gym.make(args.env, render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model
network_class = getattr(module, cfg.network_class)
actor_critic = network_class(obs_dim, act_dim).to(device)
actor_critic.load_state_dict(torch.load(best_ckpt, map_location=device))
actor_critic.eval()

# rollout and capture frames
obs, _ = env.reset()
frames = []
episode_reward = 0

with torch.no_grad():
    terminated, truncated = False, False
    while not (terminated or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_dist, _ = actor_critic(obs_tensor)
        action = action_dist.mean[0].cpu().numpy()  # Deterministic action

        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        episode_reward += reward

env.close()

# save gif
gif_run_dir = os.path.join(cfg.gif_root_dir, args.env, args.ckpt)
os.makedirs(gif_run_dir, exist_ok=True)

gif_filename = best_ckpt.split("/")[-1].replace(".pt", ".gif")
gif_path = os.path.join(gif_run_dir, gif_filename)
imageio.mimsave(gif_path, frames, fps=30)
print(f"\nSaved rollout to {gif_path}.\n")
print(f"The reward achieved in this rollout is {episode_reward:.2f}")