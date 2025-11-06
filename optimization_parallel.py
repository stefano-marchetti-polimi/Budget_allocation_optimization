# optimization_parallel.py
# Fast PPO with proper multiprocessing guard on macOS (SubprocVecEnv + DummyVecEnv fallback)

import os
import math
import json
import csv
import shutil
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym

from utils.environment import TrialEnv, _build_default_network  # must be importable at top level

# -------------------- User parameters --------------------
SCENARIO_NAME = "All"  # decision-maker preferences (neutral, gas-economic, gas-social, electricity-economic, electricity-social, All)
CLIMATE_SCENARIO = "All"  # sea level rise projections (SSP1-1.9, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5, All)
BUDGET = 500000
MC_SAMPLES = 1000

PROJECT_ROOT = Path(__file__).resolve().parent
_NETWORK_CONFIG = _build_default_network(PROJECT_ROOT / "data")
num_nodes = len(_NETWORK_CONFIG.components)
years = 75 # until 2100
year_step = 5 # 15 decisions
RL_steps = 20000000
learning_rate = 5e-4

# Per-asset footprint areas (m^2)
area = np.array([cfg.area for cfg in _NETWORK_CONFIG.components], dtype=np.float32)


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

WEIGHT_VECTOR_KEYS = ("W_g", "W_e", "W_ge", "W_gs", "W_ee", "W_es")

_NORMAL_95_Z = 1.6448536269514722  # z-score for the 95th percentile of the standard normal


def load_dm_weight_schedules(
    csv_path: str,
    years: int,
    year_step: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Load time-dependent weights for every decision-maker scenario in the CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Scenario file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    if "scenario" not in df.columns or "weight" not in df.columns:
        raise ValueError("Scenario CSV must include 'scenario' and 'weight' columns.")

    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["weight"] = df["weight"].astype(str).str.strip()

    duplicate_rows = df[df.duplicated(subset=["scenario", "weight"], keep=False)]
    if not duplicate_rows.empty:
        duplicates = (
            duplicate_rows[["scenario", "weight"]]
            .drop_duplicates()
            .apply(lambda row: f"{row['scenario']}/{row['weight']}", axis=1)
            .tolist()
        )
        raise ValueError(f"Duplicate weight rows detected for {duplicates}.")

    scenario_params: dict[str, np.ndarray] = {}
    value_cols = [col for col in df.columns if col not in {"scenario", "weight", "commodity"}]
    if not value_cols:
        raise ValueError("Scenario CSV does not contain any time-step columns.")

    try:
        sorted_cols = sorted(value_cols, key=lambda c: int(float(c)))
    except ValueError as exc:
        raise ValueError("Time-step columns must be numeric (years).") from exc

    required_points = math.ceil(years / year_step) + 1
    if len(sorted_cols) < required_points:
        raise ValueError(
            f"Decision-maker CSV provides {len(sorted_cols)} time points; "
            f"{required_points} required for {years} years with step {year_step}."
        )

    selected_cols = sorted_cols[:required_points]
    decision_years = [int(float(col)) for col in selected_cols]

    for scenario_name, scenario_df in df.groupby("scenario"):
        pivot = (
            scenario_df.drop(columns=["scenario", "commodity"], errors="ignore")
            .set_index("weight")
            .reindex(WEIGHT_VECTOR_KEYS)
        )
        if pivot.isnull().any().any():
            missing_rows = pivot.index[pivot.isnull().any(axis=1)].tolist()
            raise ValueError(
                f"Scenario '{scenario_name}' missing entries for {missing_rows} or contains NaNs."
            )

        weights_df = pivot[selected_cols].apply(pd.to_numeric, errors="coerce")
        if weights_df.isnull().any().any():
            missing_cols = weights_df.columns[weights_df.isnull().any()].tolist()
            raise ValueError(
                f"Scenario '{scenario_name}' contains NaNs in the selected years {missing_cols}."
            )

        schedule = weights_df.to_numpy(dtype=np.float32).T
        scenario_params[str(scenario_name)] = schedule

    if not scenario_params:
        raise ValueError("No decision-maker scenarios were parsed from the CSV.")

    return scenario_params, decision_years


def load_sea_level_scenarios(
    csv_path: str,
    years: int,
    year_step: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[int]]:
    """Load truncated-normal parameters for sea level deltas per scenario and decision step."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Sea level projection file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    required_cols = {"scenario", "quantile"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Sea level CSV must include columns: {sorted(required_cols)}")

    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["quantile"] = pd.to_numeric(df["quantile"], errors="coerce")
    if df["quantile"].isna().any():
        raise ValueError("Sea level CSV contains non-numeric quantile entries.")

    metadata_cols = {"lon", "lat", "process", "confidence", "scenario", "quantile"}
    value_cols = [col for col in df.columns if col not in metadata_cols]
    if not value_cols:
        raise ValueError("Sea level CSV does not contain any projection columns.")

    try:
        year_pairs = sorted(
            ((col, int(float(col)))) for col in value_cols if str(col).strip()
        )
    except ValueError as exc:
        raise ValueError("Projection column headers must be numeric years.") from exc

    if not year_pairs:
        raise ValueError("Sea level CSV does not contain valid numeric year columns.")

    n_steps = math.ceil(years / year_step)
    required_points = n_steps + 1
    if len(year_pairs) < required_points:
        raise ValueError(
            f"Sea level CSV provides {len(year_pairs)} year columns; "
            f"{required_points} are required for {years} years with step {year_step}."
        )

    selected_pairs = year_pairs[: required_points]
    selected_cols = [name for name, _ in selected_pairs]
    decision_years = [year for _, year in selected_pairs]

    needed_quantiles = {5, 50, 95}
    scenario_params: dict[str, dict[str, np.ndarray]] = {}

    for scenario_name, scenario_df in df.groupby("scenario"):
        pivot = (
            scenario_df.drop_duplicates(subset=["quantile"])
            .set_index("quantile")[selected_cols]
            .apply(pd.to_numeric, errors="coerce")
        )
        missing = needed_quantiles - set(pivot.index)
        if missing:
            raise ValueError(
                f"Scenario '{scenario_name}' missing quantiles {sorted(missing)} "
                f"in projection file."
            )

        if pivot[selected_cols].isna().any().any():
            raise ValueError(
                f"Scenario '{scenario_name}' contains NaNs in the selected projection years."
            )

        q5 = pivot.loc[5].to_numpy(dtype=np.float32)
        q50 = pivot.loc[50].to_numpy(dtype=np.float32)
        q95 = pivot.loc[95].to_numpy(dtype=np.float32)

        if np.any(q95 < q5):
            raise ValueError(
                f"Scenario '{scenario_name}' has 95th percentile below 5th percentile."
            )

        spread = np.maximum(q95 - q5, 1e-6)
        sigma = spread / (2.0 * _NORMAL_95_Z)

        scenario_params[scenario_name] = {
            "mu": q50.astype(np.float32),
            "sigma": sigma.astype(np.float32),
            "lower": q5.astype(np.float32),
            "upper": q95.astype(np.float32),
        }

    if not scenario_params:
        raise ValueError("No scenarios were parsed from the sea level projection file.")

    return scenario_params, decision_years

DM_SCENARIOS_PATH = os.path.join("Decision Makers Preferences", "DM_Scenarios.csv")
dm_weight_schedules, weight_years = load_dm_weight_schedules(
    DM_SCENARIOS_PATH,
    years,
    year_step,
)

SEA_LEVEL_PROJECTIONS_PATH = os.path.join("Sea Level Projections", "Projections.csv")
sea_level_scenarios, _ = load_sea_level_scenarios(
    SEA_LEVEL_PROJECTIONS_PATH,
    years,
    year_step,
)

env_kwargs = dict(
    num_nodes=num_nodes,
    years=years,
    weight_schedules=dm_weight_schedules,
    dm_scenario=SCENARIO_NAME,
    budget=BUDGET,
    year_step=year_step,
    area=area,
    mc_samples=MC_SAMPLES,
    csv_path='outputs/coastal_inundation_samples.csv',
    max_depth=8.0,
    threshold_depth=0.5,
    weight_years=weight_years,
    sea_level_scenarios=sea_level_scenarios,
    climate_scenario=CLIMATE_SCENARIO,
)

def _capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text

LOG_BASE_DIR = "log"
scenario_log_name = _capitalize_first(SCENARIO_NAME.replace("_", "-").replace(" ", "-"))
climate_log_name = CLIMATE_SCENARIO.replace("_", "-").replace(" ", "-")
LOG_RUN_NAME = f"{scenario_log_name}-{climate_log_name}"
LOG_DIR = os.path.join(LOG_BASE_DIR, LOG_RUN_NAME)

scenario_results_name = SCENARIO_NAME.replace(" ", "_").replace("-", "_")
climate_results_name = CLIMATE_SCENARIO.replace(" ", "_")
RESULTS_DIR = f"results_{scenario_results_name}_{climate_results_name}"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_DIR = os.path.join(RESULTS_DIR, "best_models")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

ASSET_NAMES = [cfg.name for cfg in _NETWORK_CONFIG.components]


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


class InfoMetricsCallback(BaseCallback):
    """Aggregate environment info metrics and log them to TensorBoard."""

    def __init__(
        self,
        keys: Sequence[str],
        prefix: str = "train/info",
        log_std: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.keys = list(keys)
        self.prefix = prefix.rstrip("/")
        self.log_std = bool(log_std)
        self._buffer: dict[str, list[float]] = {key: [] for key in self.keys}
        self._aliases = {
            "over_budget_penalty_signed": "over_pen_signed",
            "over_budget_penalty": "over_pen",
            "unused_budget_penalty_signed": "unused_pen_signed",
            "unused_budget_penalty": "unused_pen",
            "gas_supply_loss_mean": "gas_supply",
            "gas_industrial_loss_mean": "gas_ind",
            "elec_supply_loss_mean": "elec_supply",
            "elec_industrial_loss_mean": "elec_ind",
            "gas_social_loss_mean": "gas_soc",
            "elec_social_loss_mean": "elec_soc",
        }

    def _init_buffer(self) -> None:
        self._buffer = {key: [] for key in self.keys}

    def _safe_append(self, key: str, value) -> None:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(scalar):
            return
        self._buffer[key].append(scalar)

    def _on_training_start(self) -> None:
        self._init_buffer()

    def _on_rollout_end(self) -> bool:
        for key, values in self._buffer.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float32)
            alias = self._aliases.get(key, key)
            self.logger.record(f"{self.prefix}/{alias}_mean", float(arr.mean()))
            if self.log_std:
                self.logger.record(f"{self.prefix}/{alias}_std", float(arr.std()))
        self._init_buffer()
        return True

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True
        for info in infos:
            if not info:
                continue
            for key in self.keys:
                value = info.get(key)
                if value is None:
                    continue
                self._safe_append(key, value)
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
        learning_rate=learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=float(ent_schedule(1.0)),
        vf_coef=1.0,                  # upweight critic loss
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_BASE_DIR,
        device=device,
        seed=seed,
    )

    action_callback = ActionDistributionCallback(log_dir=LOG_DIR, log_every=1, verbose=0)
    entropy_callback = EntropyScheduleCallback(schedule_fn=ent_schedule, verbose=0)
    info_keys = (
        "reward_delta",
        "unused_budget_penalty",
        "unused_budget_penalty_signed",
        "over_budget_penalty",
        "over_budget_penalty_signed",
        "repeat_penalty",
        "normalized_unused_budget",
        "action_gain",
        "total_cost",
        "unused_budget",
        "over_budget_amount",
        "gas_supply_loss_mean",
        "gas_industrial_loss_mean",
        "elec_supply_loss_mean",
        "elec_industrial_loss_mean",
        "gas_social_loss_mean",
        "elec_social_loss_mean",
    )
    info_callback = InfoMetricsCallback(keys=info_keys, prefix="train/info", verbose=0)

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
    callback = CallbackList([entropy_callback, action_callback, info_callback, eval_callback, checkpoint_callback])

    model.learn(total_timesteps=RL_steps, callback=callback, tb_log_name=LOG_RUN_NAME)
    model.save(os.path.join(RESULTS_DIR, "policy"))
    eval_env_vec.close()

if __name__ == "__main__":
    # Important for macOS / spawn start method
    # Run this file as: python optimization_parallel.py
    main()
