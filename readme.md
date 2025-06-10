# PPO for OpenAI Gym

This project implements a Proximal Policy Optimization (PPO) algorithm to train agents in OpenAI Gym environments. It includes modular support for environment configuration, checkpointing, and visualizing policy behavior. By default, it demonstrates locomotion training in `BipedalWalker-v3` and `Humanoid-v5`.

Below are visual rollouts of trained policies:

**BipedalWalker-v3**  
A PPO agent learns to walk across uneven terrain.  
![BipedalWalker](gif/BipedalWalker-v3/06-05-25_03:09:03/epoch960_reward310.37.gif)

**Humanoid-v5**  
A high-dimensional humanoid agent learns to walk stably.  
![Humanoid](gif/Humanoid-v5/06-06-25_19:08:43/epoch3730_reward6089.30.gif)

## 1. Getting Started: Train PPO on BipedalWalker-v3

### Train From Scratch

Start training a PPO agent from scratch in the BipedalWalker-v3 environment:

```
python train.py --env BipedalWalker-v3
```

Checkpoints are saved under:

```
policy_ckpt/BipedalWalker-v3/TRAIN_TIMESTAMP/
```

- Only the **top 5 checkpoints** with the highest episode rewards are retained automatically.
- Training metrics such as reward, episode length, and loss are logged to **Weights & Biases** (`wandb`) for easy visualization and comparison.

### Resume Training

Provide the timestamp of a previous run to resume training:

```
python train.py --env BipedalWalker-v3 --resume_ckpt TRAIN_TIMESTAMP
```

- `TRAIN_TIMESTAMP` is formatted as `MM-DD-YY_HH:MM:SS`. It is also the name of the checkpoint folder under `policy_ckpt/BipedalWalker-v3/`. A pretrained policy is provided at timestamp `06-05-25_03:09:03`.
- The script will automatically load the checkpoint with the highest reward from the specified run.

### Visualize Policy Rollout

To visualize the agent's behavior after training, run:

```
python test.py --env BipedalWalker-v3 --ckpt TRAIN_TIMESTAMP
```

This command automatically selects and loads the best-performing checkpoint of that training, then saves a GIF under `gif/BipedalWalker-v3/TRAIN_TIMESTAMP`. You may also generate policy rollouts using the pretrained policy at timestamp `06-05-25_03:09:03`.

> Tip: A pretrained PPO agent for `Humanoid-v5` is also available under timestamp `06-06-25_19:08:43`. Feel free to visualize it by running the test script with:
> 
> ```
> python test.py --env Humanoid-v5 --ckpt 06-06-25_19:08:43
> ```

## 2. Train PPO on Any Gym Environment

To train a PPO agent on a different OpenAI Gym environment:

1. **Create a Configuration File**  
   Define a YAML file named exactly after the OpenAI Gym environment you plan to train on (e.g., `BipedalWalker-v3.yaml`, `Humanoid-v5.yaml`) and place it inside the `config/` directory. Use existing config files as templates. The YAML file should specify all training hyperparameters.

2. **(Optional) Define a Custom Network**  
   If the new environment requires a specialized architecture, define a new Actor-Critic class in `module.py` and reference its name in your YAML config using the `network_class` field.

3. **Start Training**  
   Begin training from scratch:

   ```
   python train.py --env ENV_NAME
   ```

   To resume training from a previous run:

   ```
   python train.py --env ENV_NAME --resume_ckpt TRAIN_TIMESTAMP
   ```

   The script will automatically load the highest-reward checkpoint from the previous run.

4. **Visualize Policy Behavior**  
   To render the trained agent and save a GIF:

   ```
   python test.py --env ENV_NAME --ckpt TRAIN_TIMESTAMP
   ```

   This generates a GIF in `gif/ENV_NAME/TRAIN_TIMESTAMP/` using the best-performing checkpoint.



## Requirements

Install the required dependencies with:

```
pip install gymnasium[box2d,mujoco] torch numpy wandb pyyaml imageio
```

This project uses:

- `gymnasium` for environment simulation (ensure `[box2d,mujoco]` extras are included for BipedalWalker and Humanoid), respectively.
- `torch` for neural network modeling and optimization.
- `wandb` for experiment tracking and visualization.
- `pyyaml` for loading training configuration files.
- `imageio` for saving rollout visualizations as GIFs.

If you plan to use a GPU (suggested for this task), ensure that your PyTorch installation is CUDA-compatible.

## Highlights

- **Parallelized Data Collection:** Efficiently samples rollouts using `gym.vector.SyncVectorEnv` for multi-environment training.
- **Stable and Efficient PPO Training:** Uses Generalized Advantage Estimation (GAE), PPO and Gradient Clipping, and optional Layer Normalization for stable gradient updates and improved convergence.
- **Easy Environment Adaptation:** Extend to new OpenAI Gym environments by simply adding a new YAML configuration file—no changes to core training code required.
- **Trackable Experiments:** Full Weights & Biases (`wandb`) integration for tracking metrics and comparing runs.

Last Updated on Jun 7 2025, UC Berkeley.