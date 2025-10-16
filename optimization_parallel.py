# optimization_parallel.py
# Fast PPO with proper multiprocessing guard on macOS (SubprocVecEnv + DummyVecEnv fallback)

import os
import math
import json
import csv
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym

SCENARIO_NAME = "neutral"  # decision-maker preferences

#CAN TRY GAE_LAMBDA = 0.9 OR LARGER CRITIC NETWORK OR DIFFERENT LEARNING RATE IF EXPLAINED VARIANCE IS STILL LOW/NEGATIVE 

#WHAT IF WE PUT UNCERTAINTY IN THE CLIMATE CHANGE SCENARIO, SO THAT EACH EPISODE IS NOT THE SAME? WITHOUT THAT, EACH EPISODE IS EXACTLY THE SAME FROM THE HAZARD POINT OF VIEW?
#GAUSSIANA CENTRATA SU MEDIANA CHE A DUE SIGMA HA 5 PERCENTILE E 95 PERCENTILE
#NEED TO ADD RISE_RATE SAMPLING IN THE RESET

# ---- Reduce thread thrash across workers ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
    CheckpointCallback,
)
from utils.environment_placeholder import TrialEnv  # must be importable at top level

WEIGHT_VECTOR_KEYS = ("W_g", "W_e", "W_ge", "W_gs", "W_ee", "W_es")


def load_dm_weight_schedule(csv_path: str, scenario: str, years: int, year_step: int) -> tuple[np.ndarray, list[int]]:
    """Load time-dependent weights for the requested scenario."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Scenario file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    if "scenario" not in df.columns or "weight" not in df.columns:
        raise ValueError("Scenario CSV must include 'scenario' and 'weight' columns.")

    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["weight"] = df["weight"].astype(str).str.strip()

    scenario_df = df[df["scenario"] == scenario].copy()
    if scenario_df.empty:
        available = sorted(df["scenario"].astype(str).str.strip().unique())
        raise ValueError(f"Scenario '{scenario}' not found. Available scenarios: {available}")

    duplicate_weights = scenario_df["weight"][scenario_df["weight"].duplicated()].unique()
    if len(duplicate_weights):
        raise ValueError(f"Duplicate weight rows for {duplicate_weights.tolist()} in scenario '{scenario}'.")

    scenario_df = scenario_df.set_index("weight")
    scenario_df = scenario_df.drop(columns=["scenario", "commodity"], errors="ignore")

    missing_keys = [key for key in WEIGHT_VECTOR_KEYS if key not in scenario_df.index]
    if missing_keys:
        raise ValueError(f"Scenario '{scenario}' missing entries for {missing_keys}.")

    value_cols = [col for col in scenario_df.columns if col.strip()]
    if not value_cols:
        raise ValueError(f"No time-step columns found for scenario '{scenario}'.")

    try:
        sorted_cols = sorted(value_cols, key=lambda c: int(float(c)))
    except ValueError as exc:
        raise ValueError("Time-step columns must be numeric (years).") from exc

    required_points = math.ceil(years / year_step) + 1
    if len(sorted_cols) < required_points:
        raise ValueError(
            f"Scenario '{scenario}' provides {len(sorted_cols)} time points; "
            f"{required_points} required for {years} years with step {year_step}."
        )

    selected_cols = sorted_cols[:required_points]
    weights_df = scenario_df.reindex(WEIGHT_VECTOR_KEYS)[selected_cols].apply(pd.to_numeric, errors="coerce")

    if weights_df.isnull().any().any():
        missing_cols = weights_df.columns[weights_df.isnull().any()].tolist()
        raise ValueError(f"Scenario '{scenario}' contains NaNs for years {missing_cols}.")

    schedule = weights_df.to_numpy(dtype=np.float32).T
    decision_years = [int(float(col)) for col in selected_cols]
    return schedule, decision_years

# -------------------- User parameters --------------------
num_nodes = 8
years = 75 # until 2100
year_step = 5 # 15 decisions
RL_steps = 5000000

# Per-asset footprint areas (m^2)
area = np.array([100, 150, 150, 50, 50, 50, 200, 300], dtype=np.float32)

DM_SCENARIOS_PATH = os.path.join("Decision Makers Preferences", "DM_Scenarios.csv")
weights_schedule, weight_years = load_dm_weight_schedule(
    DM_SCENARIOS_PATH,
    SCENARIO_NAME,
    years,
    year_step,
)

env_kwargs = dict(
    num_nodes=num_nodes,
    years=years,
    weights=weights_schedule,
    budget=200000,
    year_step=year_step,
    area=area,
    mc_samples=1000,
    csv_path='outputs/coastal_inundation_samples.csv',
    max_depth=8.0,
    threshold_depth=0.5,
    weight_years=weight_years,
)

RESULTS_DIR = "results"
LOG_DIR = "log"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
BEST_MODEL_DIR = os.path.join(RESULTS_DIR, "best_models")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

ASSET_NAMES = [
    "PV",
    "Substation1",
    "Substation2",
    "Compressor1",
    "Compressor2",
    "Compressor3",
    "ThermalUnit",
    "LNG",
]


def linear_schedule(start: float, end: float):
    """Create a linear schedule function compatible with SB3 progress callback."""
    start = float(start)
    end = float(end)

    def schedule(progress_remaining: float) -> float:
        # progress_remaining ∈ [0,1], where 1 is beginning of training
        return end + (start - end) * float(progress_remaining)

    return schedule

class ActionDistributionCallback(BaseCallback):
    """Log action usage per rollout to TensorBoard and CSV."""

    def __init__(self, log_dir: str, log_every: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_every = max(1, log_every)
        self._rollouts = 0
        self._csv_path = os.path.join(log_dir, "train_action_distribution.csv")
        self._header_written = False

    def _on_training_start(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self._header_written = os.path.exists(self._csv_path)

    def _on_rollout_end(self) -> bool:
        self._rollouts += 1
        if self._rollouts % self.log_every != 0:
            return True

        # Rollout buffer stores latest batch of transitions collected by PPO
        actions = self.model.rollout_buffer.actions  # type: ignore[attr-defined]
        if actions is None:
            return True
        actions = np.array(actions)
        if actions.ndim == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        action_space = self.model.action_space  # type: ignore[attr-defined]
        if isinstance(action_space, gym.spaces.Discrete):
            flat_actions = actions.reshape(-1).astype(np.int64)
            counts = np.bincount(flat_actions, minlength=action_space.n)
            total = counts.sum()
            if total == 0:
                return True
            freqs = counts / total

            entropy = -(freqs * np.log(freqs + 1e-8)).sum()
            self.logger.record("train/action_entropy", float(entropy))
            self.logger.record(
                "train/action_histogram/discrete",
                flat_actions,
                exclude=("stdout", "log", "csv"),
            )

            labels = [f"a{idx}" for idx in range(action_space.n)]
            self._write_csv(self.num_timesteps, counts, freqs, labels)

        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            rollout_actions = actions.reshape(-1, len(action_space.nvec))
            all_counts = []
            all_freqs = []
            labels = []
            for dim, n_choices in enumerate(action_space.nvec):
                asset_label = ASSET_NAMES[dim] if dim < len(ASSET_NAMES) else f"dim_{dim}"
                dim_actions = rollout_actions[:, dim].astype(np.int64)
                dim_counts = np.bincount(dim_actions, minlength=n_choices)
                total = dim_counts.sum()
                if total == 0:
                    dim_freqs = np.zeros_like(dim_counts, dtype=np.float64)
                else:
                    dim_freqs = dim_counts / total
                self.logger.record(
                    f"train/action_histogram/{asset_label}",
                    dim_actions,
                    exclude=("stdout", "log", "csv"),
                )
                all_counts.append(dim_counts)
                all_freqs.append(dim_freqs)
                labels.extend([f"{asset_label}_opt{val}" for val in range(n_choices)])

            counts = np.concatenate(all_counts, axis=0)
            freqs = np.concatenate(all_freqs, axis=0)
            self._write_csv(self.num_timesteps, counts, freqs, labels)

        elif isinstance(action_space, gym.spaces.Box):
            rollout_actions = actions.reshape(-1, action_space.shape[0])
            means = rollout_actions.mean(axis=0)
            stds = rollout_actions.std(axis=0)
            for idx, (mean, std) in enumerate(zip(means, stds)):
                self.logger.record(f"train/action_mean/dim_{idx}", float(mean))
                self.logger.record(f"train/action_std/dim_{idx}", float(std))
        else:
            if self.verbose:
                print(f"[ActionDistributionCallback] Unsupported action space type: {type(action_space)}")

        return True

    def _write_csv(self, timesteps: int, counts: np.ndarray, freqs: np.ndarray, labels: list[str]) -> None:
        assert len(counts) == len(freqs) == len(labels), "Counts, freqs, and labels must align."
        header = (
            ["timesteps"]
            + [f"count_{lbl}" for lbl in labels]
            + [f"freq_{lbl}" for lbl in labels]
        )
        mode = "a"
        if not self._header_written:
            mode = "w"
        with open(self._csv_path, mode, newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not self._header_written:
                writer.writerow(header)
                self._header_written = True
            writer.writerow([timesteps] + counts.tolist() + freqs.tolist())

    def _on_step(self) -> bool:
        return True


class EntropyScheduleCallback(BaseCallback):
    """Update PPO entropy coefficient following a schedule driven by training progress."""

    def __init__(self, schedule_fn, verbose: int = 0):
        super().__init__(verbose)
        self.schedule_fn = schedule_fn

    def _apply_schedule(self) -> None:
        # `_current_progress_remaining` is managed by SB3 (1.0 -> 0.0)
        progress = getattr(self.model, "_current_progress_remaining", None)
        if progress is None:
            total = getattr(self.model, "_total_timesteps", None) or 1.0
            progress = 1.0 - (self.model.num_timesteps / float(total))
        new_coef = float(self.schedule_fn(progress))
        self.model.ent_coef = new_coef

    def _on_training_start(self) -> None:
        self._apply_schedule()

    def _on_step(self) -> bool:
        self._apply_schedule()
        return True

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

def make_eval_env(seed: int):
    """Single-environment factory for evaluation."""
    def _factory():
        env = TrialEnv(**env_kwargs)
        env.reset(seed=seed)
        return env
    return VecMonitor(DummyVecEnv([_factory]))

def reset_output_dirs() -> None:
    """Remove checkpoint/best model directories so each run starts clean."""
    for path in (BEST_MODEL_DIR, CHECKPOINT_DIR):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
def main():
    seed = 42
    set_random_seed(seed)
    reset_output_dirs()

    # ---------- Vectorization & batching ----------
    cpu_count = os.cpu_count()
    n_envs = min(max(6, cpu_count - 2), 12) 

    # Rollout length per env; total batch = n_envs * n_steps
    n_steps = 2048
    total_batch = n_envs * n_steps
    target_bs = 8192 if 8192 <= total_batch else total_batch
    batch_size = math.gcd(total_batch, target_bs)  # clean divisor

    n_epochs = 5
    device = "cpu"  # try "mps" and compare wall-clock if your net is large

    policy_kwargs = dict(
    net_arch=dict(pi=[128, 128], vf=[128,128]),  # vf is critic
    ortho_init=False,
    )

    # ---------- Build training env ----------
    train_env = build_vec_env(n_envs=n_envs, seed=seed, prefer_subproc=True)
    vec_envs = train_env.num_envs
    rollout_timesteps = vec_envs * n_steps  # actual environment transitions gathered per PPO update
    eval_every_rollouts = 10

    # SB3 callbacks count environment steps (i.e. calls to env.step), not aggregated timesteps.
    # Convert desired frequencies (expressed in actual timesteps) into callback steps so saves fire as expected.
    steps_per_callback_call = max(1, vec_envs)
    checkpoint_save_freq = max(1, math.ceil(rollout_timesteps / steps_per_callback_call))*5
    eval_callback_freq = max(1, math.ceil((eval_every_rollouts * rollout_timesteps) / steps_per_callback_call))
    print(
        f"[train] vector envs={vec_envs} | checkpoint every {checkpoint_save_freq * vec_envs} timesteps "
        f"| eval every {eval_callback_freq * vec_envs} timesteps"
    )

    # ---------- PPO ----------
    ent_schedule = linear_schedule(start=0.03, end=0.0)

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=2e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=float(ent_schedule(1.0)),
        vf_coef=1.0,                  # upweight critic loss
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./log",
        device=device,
        seed=seed,
    )

    action_callback = ActionDistributionCallback(log_dir=LOG_DIR, log_every=1, verbose=0)
    entropy_callback = EntropyScheduleCallback(schedule_fn=ent_schedule, verbose=0)

    eval_env_vec = make_eval_env(seed + 1000)
    eval_callback = EvalCallback(
        eval_env_vec,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=RESULTS_DIR,
        eval_freq=eval_callback_freq,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )
    # Checkpoint every full rollout (n_steps * num_envs actual timesteps)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq,
        save_path=CHECKPOINT_DIR,
        name_prefix="policy_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=0,
    )
    callback = CallbackList([entropy_callback, action_callback, eval_callback, checkpoint_callback])

    model.learn(total_timesteps=RL_steps, callback=callback)
    model.save(os.path.join(RESULTS_DIR, "policy"))
    eval_env_vec.close()

if __name__ == "__main__":
    # Important for macOS / spawn start method
    # Run this file as: python optimization_parallel.py
    main()
