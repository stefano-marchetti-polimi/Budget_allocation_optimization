# optimization_parallel.py
# Fast PPO with proper multiprocessing guard on macOS (SubprocVecEnv + DummyVecEnv fallback)

import os
import math
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# ---- Reduce thread thrash across workers ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from utils.environment_placeholder import TrialEnv  # must be importable at top level

# -------------------- User parameters --------------------
num_nodes = 8
years = 50
year_step = 1
RL_steps = 500000

# Per-asset footprint areas (m^2)
area = np.array([100, 150, 150, 50, 50, 50, 200, 300], dtype=np.float32)

# Weights: [w_gas, w_electricity, w_gas_loss, w_gas_social, w_electricity_loss, w_electricity_social]
weights = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]

env_kwargs = dict(
    num_nodes=num_nodes,
    years=years,
    weights=weights,
    budget=200000,
    year_step=year_step,
    area=area,
    mc_samples=10000,
    csv_path='outputs/coastal_inundation_samples.csv',
    gpd_k=0.8019,
    gpd_sigma=0.1959,
    max_depth=8.0,
    threshold_depth=0.5,
)

RESULTS_DIR = "results"
LOG_DIR = "log"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(seed: int, rank: int):
    """Factory for vectorized envs (must be at module top level for pickling)."""
    def _thunk():
        e = TrialEnv(**env_kwargs)
        # Gymnasium seeding: use reset(seed=...) instead of e.seed(...)
        e.reset(seed=seed + rank)
        return e
    return _thunk

def build_vec_env(n_envs: int, seed: int, prefer_subproc: bool = True):
    """Create a vectorized env. Falls back to DummyVecEnv if Subproc fails."""
    thunks = [make_env(seed, i) for i in range(n_envs)]
    if prefer_subproc:
        try:
            env = SubprocVecEnv(thunks, start_method="spawn")
            return VecMonitor(env)
        except Exception as e:
            print(f"[WARN] SubprocVecEnv failed ({type(e).__name__}: {e}). Falling back to DummyVecEnv.")
    # DummyVecEnv runs envs sequentially in one process (compatible everywhere)
    env = DummyVecEnv(thunks)
    return VecMonitor(env)

def main():
    seed = 42
    set_random_seed(seed)

    # ---------- Vectorization & batching ----------
    cpu_count = os.cpu_count()
    n_envs = max(4, min(14, cpu_count // 2))  # tune 4–14

    # Rollout length per env; total batch = n_envs * n_steps
    n_steps = 1024
    total_batch = n_envs * n_steps
    target_bs = 8192 if 8192 <= total_batch else total_batch
    batch_size = math.gcd(total_batch, target_bs)  # clean divisor

    n_epochs = 5
    device = "cpu"  # try "mps" and compare wall-clock if your net is large

    policy_kwargs = dict(net_arch=[128, 128], ortho_init=False)

    # ---------- Build training env ----------
    train_env = build_vec_env(n_envs=n_envs, seed=seed, prefer_subproc=True)

    # ---------- PPO ----------
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.02,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./log",
        device=device,
        seed=seed,
    )

    model.learn(total_timesteps=RL_steps)
    model.save(os.path.join(RESULTS_DIR, "policy"))

    # ---------- Evaluation on a single env ----------
    eval_env = TrialEnv(**env_kwargs)
    obs, _ = eval_env.reset(seed=seed + 10)

    actions_log = []
    obs_log = []
    rewards_log = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        actions_log.append(np.asarray(action, dtype=np.float32))  # shape (num_nodes,) for Box OR scalar for Discrete
        obs_log.append(np.asarray(obs))
        obs, reward, terminated, truncated, info = eval_env.step(action)
        rewards_log.append(float(reward))

    # ---------- Save logs ----------
    # If actions are scalar (Discrete), convert to one-hot columns for consistency
    if np.ndim(actions_log[0]) == 0:
        # Discrete: map to one-hot of length (num_nodes+1) with index 0 = no-invest
        one_hots = []
        for a in actions_log:
            oh = np.zeros(num_nodes + 1, dtype=np.float32)
            oh[int(a)] = 1.0
            one_hots.append(oh)
        actions_arr = np.vstack(one_hots)
        action_cols = {f"action_onehot_{i}": actions_arr[:, i] for i in range(num_nodes + 1)}
    else:
        actions_arr = np.vstack(actions_log)
        action_cols = {f"action_{i}": actions_arr[:, i] for i in range(actions_arr.shape[1])}

    obs_str = [json.dumps(np.asarray(o).tolist()) for o in obs_log]

    df = pd.DataFrame({**action_cols, "reward": rewards_log, "obs_json": obs_str})
    df.to_csv(os.path.join(RESULTS_DIR, "evaluation_logs.csv"), index=False)

    # ---------- Plots ----------
    # For Discrete actions, plot the chosen index over time and a histogram of choices
    years_axis = [i * year_step for i in range(len(actions_log))]

    plt.figure(figsize=(10, 4))
    # Convert possibly array actions to ints
    chosen = [int(a) if np.ndim(a) == 0 else int(np.argmax(a)) for a in actions_log]
    plt.plot(years_axis, chosen, linewidth=1.5, marker='o')
    xticks = list(range(num_nodes + 1))
    labels = ['NoInv'] + [f'Asset_{i+1}' for i in range(num_nodes)]
    plt.yticks(xticks, labels, rotation=0)
    plt.xlabel("Year")
    plt.ylabel("Choice")
    plt.title("Chosen Action Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "action_over_time.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(chosen, bins=np.arange(num_nodes + 2) - 0.5, rwidth=0.9)
    plt.xticks(xticks, labels, rotation=45)
    plt.xlabel("Chosen Action")
    plt.ylabel("Frequency")
    plt.title("Action Choice Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "action_histogram.png"))
    plt.close()

    # Reward over time
    plt.figure(figsize=(10, 4))
    plt.plot(years_axis, rewards_log, linewidth=1.5)
    plt.xlabel("Year")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reward_over_time.png"))
    plt.close()

    # ---------- Cleanup ----------
    eval_env.close()
    train_env.close()

    print(
        f"Done.\n"
        f"Train envs: {n_envs}, n_steps: {n_steps}, total_batch: {n_envs * n_steps}, "
        f"batch_size: {batch_size}, n_epochs: {n_epochs}, device: {device}\n"
        f"Saved: {RESULTS_DIR}/policy, evaluation_logs.csv, and plots."
    )

if __name__ == "__main__":
    # Important for macOS / spawn start method
    # Run this file as: python optimization_parallel.py
    main()
