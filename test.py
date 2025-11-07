"""
Evaluate a saved PPO policy on the TrialEnv and reproduce the logging/plotting
performed at the end of `optimization_parallel.py`.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from functools import lru_cache
from itertools import product
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# -------------------- Evaluation configuration --------------------
# Set USE_BEST_MODEL to True to load the best model saved during evaluation (results/best_models/best_model.zip by default).
# If False, you can fall back to the latest checkpoint or a manually specified policy path.
USE_BEST_MODEL = True
BEST_MODEL_FILENAME = "best_model.zip"

# Optional override to force evaluation under a specific decision-maker scenario.
# Set to None to reuse the scenario defined in `optimization_parallel.py`.
DM_SCENARIO_OVERRIDE = "electricity-to-gas-economic"

# Optional absolute/relative path to override the training `RESULTS_DIR`.
# Leave as None to reuse the path from `optimization_parallel.py`.
RESULTS_DIR_OVERRIDE = "results_random_All"

# Optional alternate DM scenario data sources (CSV or Excel). The first existing file is used.
DM_SCENARIOS_TEST_FILES = [
    os.path.join("Decision Makers Preferences", "DM_Scenarios_Test.csv"),
    os.path.join("Decision Makers Preferences", "DM_Scenarios_Test.xlsx"),
]

# Set USE_LATEST_CHECKPOINT to True to automatically load the newest checkpoint inside
# results/checkpoints. Set it to False and provide POLICY_PATH (with or without .zip)
# to evaluate a specific file. Ignored when USE_BEST_MODEL is True.
USE_LATEST_CHECKPOINT = False
POLICY_PATH = None  # Only used when USE_LATEST_CHECKPOINT is False

# Stable-Baselines device string (e.g., "cpu", "mps", "cuda").
DEVICE = "cpu"

# Evaluation seed. For stochastic runs the seed is incremented by episode index.
EVAL_SEED = 1042

# Deterministic evaluation runs a single greedy episode.
# Set to False to sample actions; STOCHASTIC_EPISODES controls how many rollouts to average.
DETERMINISTIC_EVAL = True
STOCHASTIC_EPISODES = 10

# Expert policy configuration: index 1 corresponds to a 0.5 m increment.
EXPERT_UPGRADE_LEVEL_INDEX = 1

from optimization_parallel import (
    ASSET_NAMES,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    BEST_MODEL_DIR,
    WEIGHT_VECTOR_KEYS,
    TrialEnv,
    env_kwargs,
    year_step,
    load_dm_weight_schedules,
)

BASE_RESULTS_DIR = RESULTS_DIR
if RESULTS_DIR_OVERRIDE:
    RESULTS_DIR = os.path.expanduser(RESULTS_DIR_OVERRIDE)
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    BEST_MODEL_DIR = os.path.join(RESULTS_DIR, "best_models")

EVAL_ENV_KWARGS = dict(env_kwargs)
_EVAL_WEIGHT_SCHEDULES = dict(env_kwargs.get("weight_schedules", {}))
if _EVAL_WEIGHT_SCHEDULES:
    EVAL_ENV_KWARGS["weight_schedules"] = _EVAL_WEIGHT_SCHEDULES


def _load_dm_schedules_from_source(path: str, years: int, step: int):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return load_dm_weight_schedules(path, years, step)
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        try:
            df.to_csv(tmp_file.name, index=False)
            tmp_file.flush()
            return load_dm_weight_schedules(tmp_file.name, years, step)
        finally:
            tmp_file.close()
            os.unlink(tmp_file.name)
    raise ValueError(f"Unsupported DM scenario file extension for '{path}'.")


def _ensure_dm_scenario_available(scenario: str) -> None:
    if not scenario:
        return
    scenario_clean = scenario.strip()
    if not scenario_clean or scenario_clean == "All" or scenario_clean.lower() == "random":
        return
    if scenario_clean in _EVAL_WEIGHT_SCHEDULES:
        return
    source_path = next((p for p in DM_SCENARIOS_TEST_FILES if os.path.exists(p)), None)
    if source_path is None:
        raise FileNotFoundError(
            "DM_Scenarios_Test file not found. Checked: "
            + ", ".join(DM_SCENARIOS_TEST_FILES)
        )
    years_required = int(EVAL_ENV_KWARGS.get("years") or env_kwargs.get("years") or 0)
    if years_required <= 0:
        raise ValueError("Invalid 'years' configuration; cannot load DM scenarios.")
    step_required = int(EVAL_ENV_KWARGS.get("year_step", year_step))
    schedules, decision_years = _load_dm_schedules_from_source(
        source_path, years_required, step_required
    )
    if scenario_clean not in schedules:
        available = ", ".join(sorted(schedules.keys()))
        raise ValueError(
            f"Scenario '{scenario_clean}' not found in {source_path}. "
            f"Available: {available}"
        )
    _EVAL_WEIGHT_SCHEDULES[scenario_clean] = schedules[scenario_clean]
    EVAL_ENV_KWARGS["weight_schedules"] = _EVAL_WEIGHT_SCHEDULES
    current_years = EVAL_ENV_KWARGS.get("weight_years")
    if current_years is None:
        EVAL_ENV_KWARGS["weight_years"] = decision_years
    elif list(current_years) != list(decision_years):
        raise ValueError(
            "Decision year grid mismatch between training and DM_Scenarios_Test file."
        )


if DM_SCENARIO_OVERRIDE is not None:
    _ensure_dm_scenario_available(DM_SCENARIO_OVERRIDE)
    EVAL_ENV_KWARGS["dm_scenario"] = DM_SCENARIO_OVERRIDE

PLOT_SCENARIO_NAME = (
    DM_SCENARIO_OVERRIDE
    or EVAL_ENV_KWARGS.get("dm_scenario")
    or env_kwargs.get("dm_scenario")
    or "default"
)
PLOT_SCENARIO_SAFE = str(PLOT_SCENARIO_NAME).replace(" ", "_").replace("/", "_")
PLOTS_DIR = os.path.join(RESULTS_DIR, PLOT_SCENARIO_SAFE)


@lru_cache(maxsize=1)
def _component_category_map() -> Dict[str, str]:
    """Return a cached mapping from asset name to its category."""
    env = TrialEnv(**EVAL_ENV_KWARGS)
    try:
        return dict(env.component_categories)
    finally:
        env.close()


def resolve_model_path(path: str | None) -> str:
    """Return an existing `.zip` checkpoint path, trying common variants."""
    base = path or os.path.join(RESULTS_DIR, "policy")
    candidates = [base]
    if not base.endswith(".zip"):
        candidates.append(f"{base}.zip")
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find a saved policy. Checked: {', '.join(candidates)}"
    )


def latest_checkpoint_path() -> str:
    """Return the most recent checkpoint saved under `CHECKPOINT_DIR`."""
    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(f"No checkpoint directory found at {CHECKPOINT_DIR}.")

    checkpoints = [
        os.path.join(CHECKPOINT_DIR, entry)
        for entry in os.listdir(CHECKPOINT_DIR)
        if entry.endswith(".zip")
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint .zip files found inside {CHECKPOINT_DIR}."
        )

    pattern = re.compile(r"(\d+)_steps\.zip$")

    def sort_key(path: str) -> tuple[int, float]:
        match = pattern.search(os.path.basename(path))
        step = int(match.group(1)) if match else -1
        return (step, os.path.getmtime(path))

    checkpoints.sort(key=sort_key)
    return checkpoints[-1]


def best_model_path(filename: str | None = None) -> str:
    """Return the path to the best-evaluated model saved during training."""
    target_name = filename or BEST_MODEL_FILENAME
    best_path = os.path.join(BEST_MODEL_DIR, target_name)
    if os.path.exists(best_path):
        return best_path
    # Fall back to locating the first .zip inside the directory if the default name is missing.
    if not os.path.isdir(BEST_MODEL_DIR):
        raise FileNotFoundError(f"No best model directory found at {BEST_MODEL_DIR}.")
    candidates = [
        os.path.join(BEST_MODEL_DIR, entry)
        for entry in os.listdir(BEST_MODEL_DIR)
        if entry.endswith(".zip")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No .zip files found inside {BEST_MODEL_DIR}; cannot locate a best model."
        )
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def rollout_episode(
    action_selector: Callable[[TrialEnv, np.ndarray], np.ndarray],
    seed: int,
    *,
    episode_idx: int | None = None,
    episode_label: str = "Episode",
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[float],
    List[dict],
    Dict[str, Dict[str, float]],
]:
    """Run one evaluation episode with the provided action selector, tracking penalties."""
    env = TrialEnv(**EVAL_ENV_KWARGS)
    obs, _ = env.reset(seed=seed)

    actions_log: List[np.ndarray] = []
    obs_log: List[np.ndarray] = []
    rewards_log: List[float] = []
    infos_log: List[dict] = []

    penalty_summary: Dict[str, Dict[str, float]] = {
        "repeat_penalty": {"steps": 0.0, "total": 0.0, "assets": 0.0},
        "budget_penalty": {
            "steps": 0.0,
            "total": 0.0,
            "normalized_total": 0.0,
            "trimmed_steps": 0.0,
            "trimmed_assets": 0.0,
        },
        "over_budget_penalty": {
            "steps": 0.0,
            "total": 0.0,
            "amount": 0.0,
        },
    }

    terminated = False
    truncated = False
    step_idx = 0

    while not (terminated or truncated):
        action = action_selector(env, obs)
        action_array = np.asarray(action, dtype=np.int64)
        if action_array.shape != (env.N,):
            raise ValueError(
                f"Expected action of shape ({env.N},), received {action_array.shape} from action selector."
            )
        actions_log.append(action_array)
        obs_log.append(np.asarray(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        current_weights = np.asarray(env.weights, dtype=np.float32).copy()
        info["weights"] = current_weights
        info["gas_loss_mean"] = float(getattr(env, "gas_loss_mean", np.nan))
        info["elec_loss_mean"] = float(getattr(env, "elec_loss_mean", np.nan))
        infos_log.append(info)
        rewards_log.append(float(reward))

        action_pairs: List[str] = []
        intended = info.get("intended_heights")
        executed = info.get("executed_heights")
        for idx in range(action_array.size):
            asset_label = ASSET_NAMES[idx] if idx < len(ASSET_NAMES) else f"Asset {idx}"
            segment = f"{asset_label}=lvl{int(action_array[idx])}"
            if intended is not None and executed is not None:
                segment += f" (int:{float(intended[idx]):.2f}, exec:{float(executed[idx]):.2f})"
            action_pairs.append(segment)

        penalty_msgs: List[str] = []
        if "repeat_penalty" in info:
            penalty_summary["repeat_penalty"]["steps"] += 1
            penalty_summary["repeat_penalty"]["total"] += float(info["repeat_penalty"])
            assets = info.get("repeat_penalty_assets", [])
            penalty_summary["repeat_penalty"]["assets"] += float(len(assets))
            penalty_msgs.append(
                f"repeat_penalty={info['repeat_penalty']:.3f} assets={assets}"
            )
        penalty_summary["budget_penalty"]["steps"] += 1
        penalty_summary["budget_penalty"]["total"] += float(info.get("unused_budget_penalty", 0.0))
        penalty_summary["budget_penalty"]["normalized_total"] += float(
            info.get("normalized_unused_budget", 0.0)
        )
        over_budget_pen = float(info.get("over_budget_penalty_signed", 0.0))
        if over_budget_pen:
            penalty_summary["over_budget_penalty"]["steps"] += 1
            penalty_summary["over_budget_penalty"]["total"] += float(info.get("over_budget_penalty", 0.0))
            penalty_summary["over_budget_penalty"]["amount"] += float(info.get("over_budget_amount", 0.0))
            penalty_msgs.append(
                f"over_budget_penalty={over_budget_pen:.3f} amount={info.get('over_budget_amount', 0.0):.3f}"
            )

        budget_parts: List[str] = []
        if "total_cost" in info:
            budget_parts.append(f"used={info['total_cost']:.3f}")
        if "unused_budget" in info:
            budget_parts.append(f"unused={info['unused_budget']:.3f}")
        if "normalized_cost" in info:
            budget_parts.append(f"normalized={info['normalized_cost']:.3f}")
        if over_budget_pen:
            budget_parts.append(f"over={info.get('over_budget_amount', 0.0):.3f}")
        trimmed = info.get("trimmed_assets", [])
        if trimmed:
            penalty_summary["budget_penalty"]["trimmed_steps"] += 1
            penalty_summary["budget_penalty"]["trimmed_assets"] += float(len(trimmed))
            penalty_msgs.append(f"trimmed_assets={trimmed}")

        current_year = step_idx * year_step
        if episode_idx is not None:
            if episode_label:
                episode_prefix = f"{episode_label} {episode_idx:02d} | "
            else:
                episode_prefix = f"{episode_idx:02d} | "
        else:
            episode_prefix = f"{episode_label} | " if episode_label else ""
        print(f"{episode_prefix}Step {step_idx:02d} | Year {current_year}")
        print(f"  Actions: {', '.join(action_pairs)}")
        print(f"  Reward: {reward:.3f}")
        weight_pairs = [f"{key}={float(current_weights[idx]):.3f}" for idx, key in enumerate(WEIGHT_VECTOR_KEYS)]
        print(f"  Weights: {', '.join(weight_pairs)}")
        if budget_parts:
            print(f"  Budget: {', '.join(budget_parts)}")
        if penalty_msgs:
            print(f"  Penalties: {', '.join(penalty_msgs)}")
        else:
            print("  Penalties: none")
        print()

        step_idx += 1

    env.close()
    return actions_log, obs_log, rewards_log, infos_log, penalty_summary


def make_model_action_selector(model: PPO, deterministic: bool) -> Callable[[TrialEnv, np.ndarray], np.ndarray]:
    """Return a callable that mimics the Stable-Baselines predict interface."""

    def selector(_: TrialEnv, observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.int64)

    return selector


def _failure_probabilities_for_year(env: TrialEnv, target_year: float) -> np.ndarray:
    """Estimate per-asset failure probabilities for the specified simulation year."""
    rng_state = None
    if hasattr(env, "rng") and hasattr(env.rng, "bit_generator"):
        rng_state = env.rng.bit_generator.state
    try:
        _, state_payload = env._compute_metrics(
            env.improvement_height,
            year=target_year,
            return_states=True,
        )
    finally:
        if rng_state is not None:
            env.rng.bit_generator.state = rng_state

    failure_probs = np.zeros(env.N, dtype=np.float32)
    if not state_payload:
        return failure_probs
    functional = state_payload.get("functional")
    if not functional:
        return failure_probs
    for idx, name in enumerate(env.component_names):
        status = functional[name]
        failure_probs[idx] = 1.0 - float(np.mean(status))
    return failure_probs


def expert_action_selector(env: TrialEnv, _: np.ndarray) -> np.ndarray:
    """Allocate upgrades across the three most fragile upgradable assets within the budget."""
    action = np.zeros(env.N, dtype=np.int64)
    if env.N == 0 or float(env.budget) <= 0.0:
        return action
    if not hasattr(env, "component_upgradable"):
        return action

    upgradable_mask = env.component_upgradable.astype(bool)
    last_positive = getattr(env, "_last_positive_mask", None)
    if last_positive is not None:
        upgradable_mask &= ~np.asarray(last_positive, dtype=bool)
    if not np.any(upgradable_mask):
        return action

    target_year = env._year + env.year_step
    failure_probs = _failure_probabilities_for_year(env, float(target_year)).astype(np.float32, copy=False)
    failure_probs[~upgradable_mask] = -np.inf

    sorted_indices = np.argsort(failure_probs)[::-1]
    candidate_indices = [
        idx
        for idx in sorted_indices
        if upgradable_mask[idx] and np.isfinite(failure_probs[idx]) and failure_probs[idx] > 0.0
    ][:3]
    if not candidate_indices:
        return action

    height_levels = np.asarray(env.height_levels, dtype=np.float32)
    if height_levels.size <= 1:
        return action
    budget = float(env.budget)
    num_levels = int(height_levels.size)

    per_asset_costs: Dict[int, List[float]] = {}
    for asset_idx in candidate_indices:
        level_costs = [0.0] * num_levels
        for level_idx in range(1, num_levels):
            deltas = np.zeros(env.N, dtype=np.float32)
            deltas[asset_idx] = float(height_levels[level_idx])
            level_costs[level_idx] = float(env._compute_costs(deltas)[asset_idx])
        per_asset_costs[asset_idx] = level_costs

    best_combo: Tuple[int, ...] | None = None
    best_key = (-1.0, -1, -1.0)
    for level_choices in product(range(num_levels), repeat=len(candidate_indices)):
        total_cost = 0.0
        upgrades = 0
        failure_score = 0.0
        over_budget = False
        for asset_idx, level_idx in zip(candidate_indices, level_choices):
            if level_idx <= 0:
                continue
            cost = per_asset_costs[asset_idx][level_idx]
            total_cost += cost
            if total_cost > budget + 1e-6:
                over_budget = True
                break
            upgrades += 1
            failure_score += failure_probs[asset_idx] * level_idx
        if over_budget:
            continue
        candidate_key = (total_cost, upgrades, failure_score)
        if candidate_key > best_key:
            best_key = candidate_key
            best_combo = level_choices

    if best_combo is None or best_key[0] <= 0.0:
        return action

    for asset_idx, level_idx in zip(candidate_indices, best_combo):
        action[asset_idx] = level_idx

    return action


def plot_reward_comparison(
    learned_df: pd.DataFrame,
    expert_df: pd.DataFrame,
    *,
    output_path: str,
    multi_episode: bool,
) -> None:
    """Plot cumulative rewards for the learned and expert policies."""
    if learned_df.empty or expert_df.empty:
        print("Skipping reward comparison plot because one of the datasets is empty.")
        return

    def cumulative_summary(df: pd.DataFrame) -> pd.DataFrame:
        ordered = (
            df.sort_values(["episode", "step_in_episode"])
            .reset_index(drop=True)
            .copy()
        )
        ordered["cumulative_reward"] = ordered.groupby("episode")["reward"].cumsum()
        return ordered.groupby("year")["cumulative_reward"].agg(["mean", "std"])

    learned_summary = cumulative_summary(learned_df)
    expert_summary = cumulative_summary(expert_df)

    all_years = sorted(set(learned_summary.index).union(expert_summary.index))
    if not all_years:
        print("No overlapping years found; skipping reward comparison plot.")
        return

    plt.figure(figsize=(10, 4))
    years_array = np.asarray(all_years, dtype=np.float32)
    for summary, label, color in (
        (learned_summary, "Learned policy", "#1f77b4"),
        (expert_summary, "Expert policy", "#ff7f0e"),
    ):
        stats = summary.reindex(all_years)
        means = stats["mean"].to_numpy()
        stds = stats["std"].fillna(0.0).to_numpy()
        valid = np.isfinite(means)
        if not np.any(valid):
            continue
        if multi_episode:
            plt.errorbar(
                years_array[valid],
                means[valid],
                yerr=stds[valid],
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=label,
                color=color,
            )
        else:
            plt.plot(
                years_array[valid],
                means[valid],
                marker="o",
                linewidth=1.5,
                label=label,
                color=color,
            )
    plt.xlabel("Year")
    plt.ylabel("Cumulative reward")
    plt.title("Cumulative Reward: Learned vs Expert Policy")
    plt.grid(True, axis="both", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_expected_loss_comparison(
    learned_df: pd.DataFrame,
    expert_df: pd.DataFrame,
    *,
    output_path: str,
    multi_episode: bool,
) -> None:
    """Compare (1 - expected loss) for gas and electricity between learned and expert policies."""
    if learned_df.empty or expert_df.empty:
        print("Skipping loss comparison plot because one of the datasets is empty.")
        return

    def loss_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
        trimmed = df.dropna(subset=[column]).copy()
        if trimmed.empty:
            return pd.DataFrame(columns=["mean", "std"])
        complement = 1.0 - trimmed[column].astype(float)
        trimmed["complement"] = complement
        return trimmed.groupby("year")["complement"].agg(["mean", "std"])

    metrics = [
        ("gas_loss_mean", "Expected gas availability", "#2ca02c"),
        ("elec_loss_mean", "Expected electricity availability", "#d62728"),
    ]

    summaries = []
    all_years: set[int] = set()
    for column, _, _ in metrics:
        learned_stats = loss_summary(learned_df, column)
        expert_stats = loss_summary(expert_df, column)
        summaries.append((column, learned_stats, expert_stats))
        all_years.update(learned_stats.index.tolist())
        all_years.update(expert_stats.index.tolist())

    if not all_years:
        print("No loss statistics available; skipping loss comparison plot.")
        return

    years_sorted = sorted(all_years)
    years_array = np.asarray(years_sorted, dtype=np.float32)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (column, learned_stats, expert_stats), metric_info in zip(
        axes,
        summaries,
        metrics,
    ):
        _, title, _ = metric_info
        for stats, label, color in (
            (learned_stats.reindex(years_sorted), "Learned policy", "#1f77b4"),
            (expert_stats.reindex(years_sorted), "Expert policy", "#ff7f0e"),
        ):
            means = stats["mean"].to_numpy()
            stds = stats["std"].fillna(0.0).to_numpy()
            valid = np.isfinite(means)
            if not np.any(valid):
                continue
            if multi_episode:
                ax.errorbar(
                    years_array[valid],
                    means[valid],
                    yerr=stds[valid],
                    marker="o",
                    linewidth=1.5,
                    capsize=3,
                    label=label,
                    color=color,
                )
            else:
                ax.plot(
                    years_array[valid],
                    means[valid],
                    marker="o",
                    linewidth=1.5,
                    label=label,
                    color=color,
                )
        ax.set_ylabel(title)
        ax.grid(True, axis="both", alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Year")
    fig.suptitle("Expected Availability Comparison: Learned vs Expert")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path)
    plt.close(fig)


def save_logs(
    actions_log: List[np.ndarray],
    obs_log: List[np.ndarray],
    rewards_log: List[float],
    infos_log: List[dict],
    *,
    episode_ids: List[int] | None = None,
    step_years: List[int] | None = None,
    step_indices: List[int] | None = None,
) -> None:
    """Persist evaluation traces (single or multi-episode) to CSV and summary plots."""
    if not actions_log:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    actions_arr = np.vstack(actions_log)
    action_cols: Dict[str, np.ndarray] = {}
    for idx in range(actions_arr.shape[1]):
        asset_label = ASSET_NAMES[idx] if idx < len(ASSET_NAMES) else f"asset_{idx}"
        action_cols[f"action_level_{asset_label}"] = actions_arr[:, idx].astype(int)

    obs_str = [json.dumps(np.asarray(obs).tolist()) for obs in obs_log]

    intended_stack = np.vstack([info["intended_heights"] for info in infos_log])
    executed_stack = np.vstack([info["executed_heights"] for info in infos_log])

    intended_cols = {}
    executed_cols = {}
    for idx in range(intended_stack.shape[1]):
        asset_label = ASSET_NAMES[idx] if idx < len(ASSET_NAMES) else f"asset_{idx}"
        intended_cols[f"intended_height_{asset_label}"] = intended_stack[:, idx]
        executed_cols[f"executed_height_{asset_label}"] = executed_stack[:, idx]

    weights_stack = np.vstack([info["weights"] for info in infos_log])
    weight_cols = {
        f"weight_{key}": weights_stack[:, idx] for idx, key in enumerate(WEIGHT_VECTOR_KEYS)
    }

    cost_stack = np.vstack([info["costs"] for info in infos_log])
    cost_cols: Dict[str, np.ndarray] = {}
    for idx in range(cost_stack.shape[1]):
        asset_label = ASSET_NAMES[idx] if idx < len(ASSET_NAMES) else f"asset_{idx}"
        cost_cols[f"cost_{asset_label}"] = cost_stack[:, idx]

    impact_cols = {
        "econ_impact": [info.get("econ_impact", np.nan) for info in infos_log],
        "social_impact": [info.get("social_impact", np.nan) for info in infos_log],
        "action_gain": [info.get("action_gain", np.nan) for info in infos_log],
        "climate_drift": [info.get("climate_drift", np.nan) for info in infos_log],
        "prev_loss": [info.get("prev_loss", np.nan) for info in infos_log],
        "base_loss": [info.get("base_loss", np.nan) for info in infos_log],
        "new_loss": [info.get("new_loss", np.nan) for info in infos_log],
        "gas_loss_mean": [info.get("gas_loss_mean", np.nan) for info in infos_log],
        "elec_loss_mean": [info.get("elec_loss_mean", np.nan) for info in infos_log],
    }

    expected_depth_mean_cols = {
        f"expected_depth_mean_{asset}": [info.get(f"expected_depth_mean_{asset}", np.nan) for info in infos_log]
        for asset in ASSET_NAMES
    }
    expected_depth_p95_cols = {
        f"expected_depth_p95_{asset}": [info.get(f"expected_depth_p95_{asset}", np.nan) for info in infos_log]
        for asset in ASSET_NAMES
    }
    expected_depth_max_cols = {
        f"expected_depth_max_{asset}": [info.get(f"expected_depth_max_{asset}", np.nan) for info in infos_log]
        for asset in ASSET_NAMES
    }

    penalty_cols = {
        "repeat_penalty": [info.get("repeat_penalty", 0.0) for info in infos_log],
        "unused_budget_penalty": [info.get("unused_budget_penalty", 0.0) for info in infos_log],
        "over_budget_penalty": [info.get("over_budget_penalty", 0.0) for info in infos_log],
        "over_budget_amount": [info.get("over_budget_amount", 0.0) for info in infos_log],
        "normalized_cost": [info.get("normalized_cost", 0.0) for info in infos_log],
        "normalized_unused_budget": [info.get("normalized_unused_budget", 0.0) for info in infos_log],
        "repeat_penalty_assets": [
            json.dumps(info.get("repeat_penalty_assets", [])) for info in infos_log
        ],
        "trimmed_assets": [
            json.dumps(info.get("trimmed_assets", [])) for info in infos_log
        ],
        "total_cost": [info.get("total_cost", 0.0) for info in infos_log],
        "unused_budget": [info.get("unused_budget", 0.0) for info in infos_log],
        "sea_level_offset": [
            float(info["sea_level_offset"]) if info.get("sea_level_offset") is not None else np.nan
            for info in infos_log
        ],
        "sea_level_delta": [
            float(info["sea_level_delta"]) if info.get("sea_level_delta") is not None else np.nan
            for info in infos_log
        ],
        "climate_scenario": [info.get("climate_scenario", "") for info in infos_log],
        "dm_scenario": [info.get("dm_scenario", "") for info in infos_log],
    }

    total_steps = len(actions_log)
    if step_years is None:
        step_years = [idx * year_step for idx in range(total_steps)]
    if episode_ids is None:
        episode_ids = [0] * total_steps
    if step_indices is None:
        step_indices = list(range(total_steps))

    df = pd.DataFrame(
        {
            **action_cols,
            **intended_cols,
            **executed_cols,
            **weight_cols,
            **cost_cols,
            **impact_cols,
            **expected_depth_mean_cols,
            **expected_depth_p95_cols,
            **expected_depth_max_cols,
            **penalty_cols,
            "reward": rewards_log,
            "obs_json": obs_str,
            "year": step_years,
            "episode": episode_ids,
            "step_in_episode": step_indices,
        }
    )

    category_map = _component_category_map()
    gas_assets = [name for name, cat in category_map.items() if cat in {"lng", "compressor"}]
    elec_assets = [name for name in ASSET_NAMES if name not in gas_assets]
    gas_cost_cols = [f"cost_{asset}" for asset in gas_assets if f"cost_{asset}" in df.columns]
    elec_cost_cols = [f"cost_{asset}" for asset in elec_assets if f"cost_{asset}" in df.columns]
    if gas_cost_cols:
        df["gas_spend_step"] = df[gas_cost_cols].sum(axis=1)
        df["gas_cumulative_spend"] = df.groupby("episode")["gas_spend_step"].cumsum()
    else:
        df["gas_spend_step"] = 0.0
        df["gas_cumulative_spend"] = 0.0
    if elec_cost_cols:
        df["elec_spend_step"] = df[elec_cost_cols].sum(axis=1)
        df["elec_cumulative_spend"] = df.groupby("episode")["elec_spend_step"].cumsum()
    else:
        df["elec_spend_step"] = 0.0
        df["elec_cumulative_spend"] = 0.0

    multi_episode = df["episode"].nunique() > 1
    years_axis = df["year"].to_numpy()

    budget_cap = float(EVAL_ENV_KWARGS.get("budget", 0.0))
    df["cumulative_spend"] = df.groupby("episode")["total_cost"].cumsum()
    if budget_cap > 0.0:
        df["cumulative_budget_cap"] = (df["step_in_episode"] + 1) * budget_cap
    else:
        df["cumulative_budget_cap"] = np.nan

    executed_columns = list(executed_cols.keys())
    cumulative_height_columns: List[str] = []
    for column in executed_columns:
        asset_label = column.replace("executed_height_", "")
        cum_col = f"cumulative_height_{asset_label}"
        df[cum_col] = df.groupby("episode")[column].cumsum()
        cumulative_height_columns.append(cum_col)
    if cumulative_height_columns:
        df["total_cumulative_height"] = df[cumulative_height_columns].sum(axis=1)
    else:
        df["total_cumulative_height"] = 0.0

    df["reward_delta_component"] = (
        df["reward"]
        + df["unused_budget_penalty"].astype(float)
        + df["over_budget_penalty"].astype(float)
        - df["repeat_penalty"].astype(float)
    )
    df["unused_budget_penalty_signed"] = -df["unused_budget_penalty"].astype(float)
    df["over_budget_penalty_signed"] = -df["over_budget_penalty"].astype(float)

    df.to_csv(os.path.join(RESULTS_DIR, "evaluation_logs.csv"), index=False)

    df = df.sort_values(["episode", "step_in_episode"]).reset_index(drop=True)

    fig, axes = plt.subplots(
        actions_arr.shape[1],
        1,
        figsize=(10, 2.0 * actions_arr.shape[1]),
        sharex=True,
    )
    if actions_arr.shape[1] == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        asset_label = ASSET_NAMES[idx] if idx < len(ASSET_NAMES) else f"Asset {idx}"
        column = f"action_level_{asset_label}"
        if multi_episode:
            stats = (
                df.groupby("year")[column]
                .agg(["mean", "std"])
                .rename(columns={"mean": "avg", "std": "std"})
            )
            ax.errorbar(
                stats.index,
                stats["avg"],
                yerr=stats["std"].fillna(0.0),
                marker="o",
                linewidth=1.5,
                capsize=3,
            )
        else:
            ax.plot(years_axis, df[column].to_numpy(), marker="o", linewidth=1.5)
        unique_levels = np.unique(actions_arr[:, idx])
        ax.set_yticks(unique_levels)
        ax.set_ylabel(asset_label)
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xlabel("Year")
    title = "Action Levels Over Time"
    if multi_episode:
        title += " (mean ± std)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(os.path.join(PLOTS_DIR, "action_over_time.png"))
    plt.close(fig)

    plt.figure(figsize=(10, 4))
    weight_columns = [f"weight_{key}" for key in WEIGHT_VECTOR_KEYS]
    if multi_episode:
        weight_stats = (
            df.groupby("year")[weight_columns]
            .agg(["mean", "std"])
            .rename(columns={"mean": "avg", "std": "std"})
        )
        years = weight_stats.index.to_numpy()
        for key in WEIGHT_VECTOR_KEYS:
            mean_vals = weight_stats[(f"weight_{key}", "avg")].to_numpy()
            std_vals = weight_stats[(f"weight_{key}", "std")].fillna(0.0).to_numpy()
            plt.errorbar(
                years,
                mean_vals,
                yerr=std_vals,
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=key,
            )
    else:
        for idx, key in enumerate(WEIGHT_VECTOR_KEYS):
            plt.plot(years_axis, df[f"weight_{key}"].to_numpy(), marker="o", linewidth=1.5, label=key)
    plt.xlabel("Year")
    plt.ylabel("Weight value")
    weight_title = "Decision-Maker Weights Over Time"
    if multi_episode:
        weight_title += " (mean ± std)"
    plt.title(weight_title)
    plt.legend(loc="best")
    plt.grid(True, axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "weights_over_time.png"))
    plt.close()

    # Budget utilisation: cumulative spend vs theoretical budget cap
    fig, ax = plt.subplots(figsize=(10, 4))
    if multi_episode:
        spend_stats = (
            df.groupby("year")["cumulative_spend"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "avg", "std": "std"})
        )
        years = spend_stats.index.to_numpy()
        ax.errorbar(
            years,
            spend_stats["avg"],
            yerr=spend_stats["std"].fillna(0.0),
            marker="o",
            linewidth=1.5,
            capsize=3,
            label="Cumulative spend (mean ± std)",
        )
        if budget_cap > 0.0:
            cap_series = df.groupby("year")["cumulative_budget_cap"].mean()
            ax.plot(
                cap_series.index,
                cap_series.to_numpy(),
                linestyle="--",
                linewidth=1.5,
                color="#ED7D31",
                label="Budget cap",
            )
    else:
        episode0 = df[df["episode"] == df["episode"].iloc[0]]
        ax.plot(
            episode0["year"],
            episode0["cumulative_spend"],
            marker="o",
            linewidth=1.5,
            label="Cumulative spend",
        )
        if budget_cap > 0.0:
            ax.plot(
                episode0["year"],
                episode0["cumulative_budget_cap"],
                linestyle="--",
                linewidth=1.5,
                color="#ED7D31",
                label="Budget cap",
            )
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative spend")
    ax.set_title("Budget Utilisation Over Time")
    ax.grid(True, axis="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "budget_utilisation.png"))
    plt.close(fig)

    plt.figure(figsize=(10, 4))
    if multi_episode:
        unused_stats = (
            df.groupby("year")["unused_budget"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "avg", "std": "std"})
        )
        plt.errorbar(
            unused_stats.index,
            unused_stats["avg"],
            yerr=unused_stats["std"].fillna(0.0),
            marker="o",
            linewidth=1.5,
            capsize=3,
            label="Unused budget (mean ± std)",
        )
    else:
        episode_unused = df[df["episode"] == df["episode"].iloc[0]]
        plt.plot(
            episode_unused["year"],
            episode_unused["unused_budget"],
            marker="o",
            linewidth=1.5,
        )
    plt.xlabel("Year")
    plt.ylabel("Unused budget")
    plt.title("Unused Budget Per Decision Step")
    plt.grid(True, axis="both", alpha=0.3)
    if multi_episode:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "unused_budget_over_time.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    cost_columns = [f"cost_{ASSET_NAMES[idx] if idx < len(ASSET_NAMES) else f'asset_{idx}'}" for idx in range(cost_stack.shape[1])]
    if multi_episode:
        cost_stats = (
            df.groupby("year")[cost_columns]
            .agg(["mean", "std"])
            .rename(columns={"mean": "avg", "std": "std"})
        )
        years = cost_stats.index.to_numpy()
        for idx, column in enumerate(cost_columns):
            asset_label = ASSET_NAMES[idx] if idx < len(ASSET_NAMES) else f"Asset {idx}"
            mean_vals = cost_stats[(column, "avg")].to_numpy()
            std_vals = cost_stats[(column, "std")].fillna(0.0).to_numpy()
            plt.errorbar(
                years,
                mean_vals,
                yerr=std_vals,
                marker="o",
                linewidth=1.5,
                capsize=3,
                label=asset_label,
            )
    else:
        for idx, column in enumerate(cost_columns):
            asset_label = ASSET_NAMES[idx] if idx < len(ASSET_NAMES) else f"Asset {idx}"
            plt.plot(years_axis, df[column].to_numpy(), marker="o", linewidth=1.5, label=asset_label)
    plt.xlabel("Year")
    plt.ylabel("Cost spent")
    cost_title = "Cost Spent Per Asset Over Time"
    if multi_episode:
        cost_title += " (mean ± std)"
    plt.title(cost_title)
    plt.legend(loc="best")
    plt.grid(True, axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cost_over_time.png"))
    plt.close()

    total_cost_per_asset = cost_stack.sum(axis=0)
    plt.figure(figsize=(10, 4))
    asset_labels = [ASSET_NAMES[idx] if idx < len(ASSET_NAMES) else f"Asset {idx}" for idx in range(cost_stack.shape[1])]
    plt.bar(asset_labels, total_cost_per_asset, color="#4C72B0")
    plt.ylim(0, 1.2*10**7)
    plt.ylabel("Total cost spent")
    plt.title("Total Cost Spent Per Asset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cost_by_asset.png"))
    plt.close()

    category_map = _component_category_map()
    asset_categories = [category_map.get(name, "unclassified") for name in ASSET_NAMES]
    category_totals: Dict[str, float] = {}
    for idx, category in enumerate(asset_categories):
        category_totals[category] = category_totals.get(category, 0.0) + float(total_cost_per_asset[idx])

    gas_categories = {"lng", "compressor"}
    gas_total = sum(category_totals.get(cat, 0.0) for cat in gas_categories)
    electricity_total = sum(category_totals.values()) - gas_total
    group_labels = ["Gas assets", "Electricity assets"]
    grouped_totals = [gas_total, electricity_total]

    plt.figure(figsize=(10, 4))
    plt.bar(group_labels, grouped_totals, color=["#55A868", "#4C72B0"])
    plt.ylim(0, 5*10**6)
    plt.ylabel("Total cost spent")
    plt.title("Total Cost by Asset Group")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cost_by_group.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    if multi_episode:
        econ_stats = (
            df.groupby("year")[["econ_impact", "social_impact"]]
            .agg(["mean", "std"])
            .rename(columns={"mean": "avg", "std": "std"})
        )
        years = econ_stats.index.to_numpy()
        plt.errorbar(
            years,
            econ_stats[("econ_impact", "avg")],
            yerr=econ_stats[("econ_impact", "std")].fillna(0.0),
            marker="o",
            linewidth=1.5,
            capsize=3,
            label="Economic impact",
        )
        plt.errorbar(
            years,
            econ_stats[("social_impact", "avg")],
            yerr=econ_stats[("social_impact", "std")].fillna(0.0),
            marker="s",
            linewidth=1.5,
            capsize=3,
            label="Social impact",
        )
    else:
        episode_impacts = df[df["episode"] == df["episode"].iloc[0]]
        plt.plot(
            episode_impacts["year"],
            episode_impacts["econ_impact"],
            marker="o",
            linewidth=1.5,
            label="Economic impact",
        )
        plt.plot(
            episode_impacts["year"],
            episode_impacts["social_impact"],
            marker="s",
            linewidth=1.5,
            label="Social impact",
        )
    plt.xlabel("Year")
    plt.ylabel("Impact")
    plt.title("Economic and Social Impact Over Time")
    plt.legend(loc="best")
    plt.grid(True, axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "impact_over_time.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    if multi_episode:
        reward_stats = (
            df.groupby("year")[["reward_delta_component", "unused_budget_penalty_signed", "over_budget_penalty_signed", "repeat_penalty"]]
            .agg(["mean", "std"])
            .rename(columns={"mean": "avg", "std": "std"})
        )
        years = reward_stats.index.to_numpy()
        plt.errorbar(
            years,
            reward_stats[("reward_delta_component", "avg")],
            yerr=reward_stats[("reward_delta_component", "std")].fillna(0.0),
            marker="o",
            linewidth=1.5,
            capsize=3,
            label="Reward delta",
        )
        plt.errorbar(
            years,
            reward_stats[("unused_budget_penalty_signed", "avg")],
            yerr=reward_stats[("unused_budget_penalty_signed", "std")].fillna(0.0),
            marker="s",
            linewidth=1.5,
            capsize=3,
            label="Unused-budget penalty",
        )
        plt.errorbar(
            years,
            reward_stats[("over_budget_penalty_signed", "avg")],
            yerr=reward_stats[("over_budget_penalty_signed", "std")].fillna(0.0),
            marker="d",
            linewidth=1.5,
            capsize=3,
            label="Over-budget penalty",
        )
        plt.errorbar(
            years,
            reward_stats[("repeat_penalty", "avg")],
            yerr=reward_stats[("repeat_penalty", "std")].fillna(0.0),
            marker="^",
            linewidth=1.5,
            capsize=3,
            label="Repeat penalty",
        )
    else:
        plt.bar(
            df["year"],
            df["reward_delta_component"],
            label="Reward delta",
            width=2.0,
            alpha=0.7,
        )
        plt.bar(
            df["year"],
            df["unused_budget_penalty_signed"],
            label="Unused-budget penalty",
            width=2.0,
            alpha=0.7,
        )
        plt.bar(
            df["year"],
            df["over_budget_penalty_signed"],
            label="Over-budget penalty",
            width=2.0,
            alpha=0.7,
        )
        if df["repeat_penalty"].abs().sum() != 0.0:
            plt.bar(
                df["year"],
                df["repeat_penalty"],
                label="Repeat penalty",
                width=2.0,
                alpha=0.7,
            )
    plt.xlabel("Year")
    plt.ylabel("Reward components")
    plt.title("Reward Decomposition")
    plt.grid(True, axis="both", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "reward_decomposition.png"))
    plt.close()

    # Action heatmap (average action level per asset-year)
    heatmap_years = sorted(df["year"].unique())
    action_columns = list(action_cols.keys())
    heatmap_data = np.zeros((len(action_columns), len(heatmap_years)), dtype=np.float32)
    for idx, column in enumerate(action_columns):
        mean_levels = df.groupby("year")[column].mean()
        heatmap_data[idx, :] = mean_levels.reindex(heatmap_years).to_numpy()
    plt.figure(figsize=(12, max(4, 0.6 * len(action_columns))))
    im = plt.imshow(
        heatmap_data,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar(im, label="Average action level")
    plt.xticks(
        range(len(heatmap_years)),
        heatmap_years,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        range(len(action_columns)),
        [col.replace("action_level_", "") for col in action_columns],
    )
    plt.xlabel("Year")
    plt.title("Action Heatmap (Average Level per Asset-Year)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "action_heatmap.png"))
    plt.close()

    if df["sea_level_offset"].notna().any():
        # Per-asset sea level vs cumulative height plots
        for cum_col in cumulative_height_columns:
            asset_label = cum_col.replace("cumulative_height_", "")
            depth_mean_col = f"expected_depth_mean_{asset_label}"
            depth_p95_col = f"expected_depth_p95_{asset_label}"
            depth_max_col = f"expected_depth_max_{asset_label}"
            file_safe = asset_label.lower()
            plt.figure(figsize=(10, 4))
            if multi_episode:
                depth_mean_stats = (
                    df.groupby("year")[depth_mean_col]
                    .agg(["mean", "std"])
                    .rename(columns={"mean": "avg", "std": "std"})
                )
                depth_p95_stats = (
                    df.groupby("year")[depth_p95_col]
                    .agg(["mean", "std"])
                    .rename(columns={"mean": "avg", "std": "std"})
                )
                depth_max_stats = (
                    df.groupby("year")[depth_max_col]
                    .agg(["mean", "std"])
                    .rename(columns={"mean": "avg", "std": "std"})
                )
                height_stats = (
                    df.groupby("year")[cum_col]
                    .agg(["mean", "std"])
                    .rename(columns={"mean": "avg", "std": "std"})
                )
                plt.errorbar(
                    depth_mean_stats.index,
                    depth_mean_stats["avg"],
                    yerr=depth_mean_stats["std"].fillna(0.0),
                    marker="o",
                    linewidth=1.5,
                    capsize=3,
                    label="Mean depth",
                )
                plt.errorbar(
                    depth_p95_stats.index,
                    depth_p95_stats["avg"],
                    yerr=depth_p95_stats["std"].fillna(0.0),
                    marker="^",
                    linewidth=1.5,
                    capsize=3,
                    label="95th percentile depth",
                )
                plt.errorbar(
                    depth_max_stats.index,
                    depth_max_stats["avg"],
                    yerr=depth_max_stats["std"].fillna(0.0),
                    marker="v",
                    linewidth=1.5,
                    capsize=3,
                    label="Max depth",
                )
                plt.errorbar(
                    height_stats.index,
                    height_stats["avg"],
                    yerr=height_stats["std"].fillna(0.0),
                    marker="s",
                    linewidth=1.5,
                    capsize=3,
                    label=f"{asset_label} height",
                )
            else:
                episode_data = df[df["episode"] == df["episode"].iloc[0]]
                plt.plot(
                    episode_data["year"],
                    episode_data[depth_mean_col],
                    marker="o",
                    linewidth=1.5,
                    label="Mean depth",
                )
                plt.plot(
                    episode_data["year"],
                    episode_data[depth_p95_col],
                    marker="^",
                    linewidth=1.5,
                    label="95th percentile depth",
                )
                plt.plot(
                    episode_data["year"],
                    episode_data[depth_max_col],
                    marker="v",
                    linewidth=1.5,
                    label="Max depth",
                )
                plt.plot(
                    episode_data["year"],
                    episode_data[cum_col],
                    marker="s",
                    linewidth=1.5,
                    label=f"{asset_label} height",
                )
            plt.xlabel("Year")
            plt.ylabel("Meters")
            title = f"Flood Depth vs. {asset_label} Height"
            climate_labels = sorted(label for label in df["climate_scenario"].dropna().unique().tolist() if label)
            dm_labels = sorted(label for label in df["dm_scenario"].dropna().unique().tolist() if label)
            label_parts = []
            if climate_labels:
                label_parts.append(f"Climate: {', '.join(climate_labels)}")
            if dm_labels:
                label_parts.append(f"DM: {', '.join(dm_labels)}")
            if label_parts:
                title += f" ({'; '.join(label_parts)})"
            plt.title(title)
            plt.grid(True, axis="both", alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"expected_depth_vs_height_{file_safe}.png"))
            plt.close()

        # Per-asset cumulative height plot (unchanged)
        fig, ax = plt.subplots(figsize=(12, 4))
        for cum_col in cumulative_height_columns:
            asset_label = cum_col.replace("cumulative_height_", "")
            if multi_episode:
                height_stats = (
                    df.groupby("year")[cum_col]
                    .agg(["mean", "std"])
                    .rename(columns={"mean": "avg", "std": "std"})
                )
                ax.errorbar(
                    height_stats.index,
                    height_stats["avg"],
                    yerr=height_stats["std"].fillna(0.0),
                    marker="o",
                    linewidth=1.2,
                    capsize=3,
                    label=asset_label,
                )
            else:
                episode_heights = df[df["episode"] == df["episode"].iloc[0]]
                ax.plot(
                    episode_heights["year"],
                    episode_heights[cum_col],
                    marker="o",
                    linewidth=1.2,
                    label=asset_label,
                )
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative height (m)")
        ax.set_title("Cumulative Wall Height per Asset")
        ax.grid(True, axis="both", alpha=0.3)
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "cumulative_height_per_asset.png"))
        plt.close(fig)

    # Cumulative spend by preference group (gas vs electricity)
    spend_summary = (
        df.groupby(["episode", "year"])[
            ["gas_cumulative_spend", "elec_cumulative_spend", "gas_spend_step", "elec_spend_step"]
        ]
        .last()
        .reset_index()
    )
    if not spend_summary.empty:
        spend_stats = (
            spend_summary.groupby("year")[
                ["gas_cumulative_spend", "elec_cumulative_spend", "gas_spend_step", "elec_spend_step"]
            ]
            .mean()
            .sort_index()
        )
        years_axis_spend = spend_stats.index.to_numpy()
        gas_series = spend_stats["gas_cumulative_spend"].to_numpy()
        elec_series = spend_stats["elec_cumulative_spend"].to_numpy()
        gas_step_series = spend_stats["gas_spend_step"].to_numpy()
        elec_step_series = spend_stats["elec_spend_step"].to_numpy()
        fig_spend, ax_spend = plt.subplots(figsize=(12, 4))
        ax_spend.stackplot(
            years_axis_spend,
            gas_series,
            elec_series,
            labels=["Gas (LNG + compressors)", "Electricity (others)"],
            colors=["#1f77b4", "#ff7f0e"],
            alpha=0.85,
        )
        ax_spend.set_xlabel("Year")
        ax_spend.set_ylabel("Cumulative spend (currency units)")
        ax_spend.set_title("Cumulative Spend by Decision-Maker Preference Group")
        ax_spend.grid(True, axis="both", alpha=0.3)

        weight_cols_present = {"weight_W_g", "weight_W_e"}.issubset(df.columns)
        if weight_cols_present:
            weight_stats = (
                df.groupby("year")[["weight_W_g", "weight_W_e"]]
                .mean()
                .reindex(years_axis_spend, method="nearest")
            )
            ax_weights = ax_spend.twinx()
            ax_weights.plot(
                years_axis_spend,
                weight_stats["weight_W_g"],
                color="#0d3b66",
                linestyle="--",
                linewidth=1.5,
                label="Weight W_g",
            )
            ax_weights.plot(
                years_axis_spend,
                weight_stats["weight_W_e"],
                color="#f95738",
                linestyle=":",
                linewidth=1.5,
                label="Weight W_e",
            )
            ax_weights.set_ylabel("DM weight")
            ax_weights.set_ylim(0.0, 1.05)
            handles, labels = ax_spend.get_legend_handles_labels()
            handles_w, labels_w = ax_weights.get_legend_handles_labels()
            ax_spend.legend(handles + handles_w, labels + labels_w, loc="upper left")
        else:
            ax_spend.legend(loc="upper left")

        fig_spend.tight_layout()
        fig_spend.savefig(os.path.join(PLOTS_DIR, "cumulative_spend_by_group.png"))
        plt.close(fig_spend)

        diff_series = gas_series - elec_series
        step_diff_series = gas_step_series - elec_step_series
        fig_diff, ax_diff = plt.subplots(figsize=(12, 3))
        ax_diff.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        for idx in range(len(years_axis_spend) - 1):
            x0, x1 = years_axis_spend[idx], years_axis_spend[idx + 1]
            y = diff_series[idx]
            color = "#1f77b4" if y >= 0 else "#ff7f0e"
            ax_diff.fill_between(
                [x0, x1],
                [0, 0],
                [y, y],
                color=color,
                alpha=0.4,
                step="post",
            )
        ax_diff.step(
            years_axis_spend,
            diff_series,
            where="post",
            color="#333333",
            linewidth=1.5,
            label="Gas - Electricity cumulative spend",
        )
        ax_diff.set_xlabel("Year")
        ax_diff.set_ylabel("Δ cumulative spend (gas - electricity)")
        ax_diff.set_title("Spend Advantage Over Time")
        ax_diff.grid(True, axis="y", alpha=0.3)
        ax_diff.legend(loc="upper left")
        fig_diff.tight_layout()
        fig_diff.savefig(os.path.join(PLOTS_DIR, "cumulative_spend_difference.png"))
        plt.close(fig_diff)

        fig_step_diff, ax_step_diff = plt.subplots(figsize=(12, 3))
        ax_step_diff.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        colors = np.where(step_diff_series >= 0, "#1f77b4", "#ff7f0e")
        ax_step_diff.bar(
            years_axis_spend,
            step_diff_series,
            width=np.diff(np.concatenate([years_axis_spend, [years_axis_spend[-1] + year_step]])),
            color=colors,
            alpha=0.6,
            align="edge",
        )
        ax_step_diff.set_xlabel("Year")
        ax_step_diff.set_ylabel("Gas - Electricity spend (per step)")
        ax_step_diff.set_title("Instantaneous Spend Difference by Year")
        ax_step_diff.grid(True, axis="y", alpha=0.3)
        fig_step_diff.tight_layout()
        fig_step_diff.savefig(os.path.join(PLOTS_DIR, "step_spend_difference.png"))
        plt.close(fig_step_diff)

    n_assets = actions_arr.shape[1]
    ncols = min(3, n_assets)
    nrows = math.ceil(n_assets / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.0 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    for idx in range(n_assets):
        ax = axes_flat[idx]
        asset_label = ASSET_NAMES[idx] if idx < len(ASSET_NAMES) else f"Asset {idx}"
        data = actions_arr[:, idx]
        bins = np.arange(data.max() + 2) - 0.5
        ax.hist(data, bins=bins, rwidth=0.8)
        ax.set_xticks(range(int(data.max()) + 1))
        ax.set_xlabel("Action level")
        ax.set_ylabel("Frequency")
        ax.set_title(asset_label)
        ax.grid(True, axis="y", alpha=0.3)
    for idx in range(n_assets, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    fig.suptitle("Action Choice Distribution per Asset")
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(os.path.join(PLOTS_DIR, "action_histogram.png"))
    plt.close(fig)

    plt.figure(figsize=(10, 4))
    if multi_episode:
        reward_stats = (
            df.groupby("year")["reward"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "avg", "std": "std"})
        )
        plt.errorbar(
            reward_stats.index,
            reward_stats["avg"],
            yerr=reward_stats["std"].fillna(0.0),
            marker="o",
            linewidth=1.5,
            capsize=3,
        )
    else:
        plt.plot(years_axis, df["reward"].to_numpy(), linewidth=1.5)
    plt.xlabel("Year")
    plt.ylabel("Reward")
    reward_title = "Reward Over Time"
    if multi_episode:
        reward_title += " (mean ± std)"
    plt.title(reward_title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "reward_over_time.png"))
    plt.close()

    if df["sea_level_offset"].notna().any():
        plt.figure(figsize=(10, 4))
        offset_stats = (
            df.dropna(subset=["sea_level_offset"])
            .groupby("year")["sea_level_offset"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "avg", "std": "std"})
        )
        if not offset_stats.empty:
            plt.errorbar(
                offset_stats.index,
                offset_stats["avg"],
                yerr=offset_stats["std"].fillna(0.0),
                marker="o",
                linewidth=1.5,
                capsize=3,
                label="Sea level",
            )
        if df["sea_level_delta"].notna().any():
            delta_stats = (
                df.dropna(subset=["sea_level_delta"])
                .groupby("year")["sea_level_delta"]
                .agg(["mean"])
                .rename(columns={"mean": "avg"})
            )
            if not delta_stats.empty:
                plt.step(
                    delta_stats.index,
                    delta_stats["avg"],
                    where="post",
                    linestyle="--",
                    linewidth=1.2,
                    label="Delta",
                )
        title = "Sea Level Path"
        climate_labels = sorted(label for label in df["climate_scenario"].dropna().unique().tolist() if label)
        dm_labels = sorted(label for label in df["dm_scenario"].dropna().unique().tolist() if label)
        label_parts = []
        if climate_labels:
            label_parts.append(f"Climate: {', '.join(climate_labels)}")
        if dm_labels:
            label_parts.append(f"DM: {', '.join(dm_labels)}")
        if label_parts:
            title += f" ({'; '.join(label_parts)})"
        plt.xlabel("Year")
        plt.ylabel("Sea level (m)")
        plt.title(title)
        plt.grid(True, axis="both", alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "sea_level_over_time.png"))
        plt.close()

    return df


def main() -> None:
    if USE_BEST_MODEL:
        policy_path = best_model_path()
        if not USE_LATEST_CHECKPOINT or POLICY_PATH is not None:
            print("Loading best model; ignoring USE_LATEST_CHECKPOINT and POLICY_PATH settings.")
    elif USE_LATEST_CHECKPOINT:
        policy_path = latest_checkpoint_path()
        if POLICY_PATH is not None:
            print(
                f"Ignoring POLICY_PATH={POLICY_PATH!r} because USE_LATEST_CHECKPOINT=True."
            )
    else:
        policy_path = resolve_model_path(POLICY_PATH)

    model = PPO.load(policy_path, device=DEVICE)

    deterministic_eval = bool(DETERMINISTIC_EVAL)
    if deterministic_eval:
        if STOCHASTIC_EPISODES != 10:
            print(
                f"Ignoring STOCHASTIC_EPISODES={STOCHASTIC_EPISODES} because "
                "deterministic evaluation performs a single episode."
            )
        num_episodes = 1
    else:
        try:
            target_episodes = int(STOCHASTIC_EPISODES)
        except (TypeError, ValueError):
            raise ValueError("STOCHASTIC_EPISODES must be convertible to int when DETERMINISTIC_EVAL is False.")
        num_episodes = max(1, target_episodes)

    aggregated_penalties: Dict[str, Dict[str, float]] = {
        "repeat_penalty": {"steps": 0.0, "total": 0.0, "assets": 0.0},
        "budget_penalty": {
            "steps": 0.0,
            "total": 0.0,
            "normalized_total": 0.0,
            "trimmed_steps": 0.0,
            "trimmed_assets": 0.0,
        },
        "over_budget_penalty": {
            "steps": 0.0,
            "total": 0.0,
            "amount": 0.0,
        },
    }

    ppo_action_selector = make_model_action_selector(model, deterministic_eval)
    ppo_episode_label = "Episode" if num_episodes > 1 else ""
    episode_seeds: List[int] = []
    all_actions: List[np.ndarray] = []
    all_obs: List[np.ndarray] = []
    all_rewards: List[float] = []
    all_infos: List[dict] = []
    episode_ids: List[int] = []
    step_years: List[int] = []
    episode_lengths: List[int] = []
    step_indices: List[int] = []

    for episode_idx in range(num_episodes):
        episode_seed = EVAL_SEED if deterministic_eval else EVAL_SEED + episode_idx
        episode_seeds.append(episode_seed)
        actions_log, obs_log, rewards_log, infos_log, penalty_summary = rollout_episode(
            ppo_action_selector,
            seed=episode_seed,
            episode_idx=episode_idx if num_episodes > 1 else None,
            episode_label=ppo_episode_label,
        )

        episode_length = len(actions_log)
        episode_lengths.append(episode_length)

        for step_idx, (action, obs, reward, info) in enumerate(
            zip(actions_log, obs_log, rewards_log, infos_log)
        ):
            all_actions.append(action)
            all_obs.append(obs)
            all_rewards.append(reward)
            all_infos.append(info)
            episode_ids.append(episode_idx)
            step_years.append(step_idx * year_step)
            step_indices.append(step_idx)

        for key, stats in penalty_summary.items():
            for stat_key, value in stats.items():
                aggregated_penalties[key][stat_key] += float(value)

    ppo_df = save_logs(
        all_actions,
        all_obs,
        all_rewards,
        all_infos,
        episode_ids=episode_ids,
        step_years=step_years,
        step_indices=step_indices,
    )

    total_steps = len(all_actions)
    print(
        f"Evaluation complete.\n"
        f"Policy: {policy_path}\n"
        f"Episodes: {num_episodes}\n"
        f"Steps per episode: {episode_lengths}\n"
        f"Total steps: {total_steps}\n"
        f"Device: {DEVICE}\n"
        f"Deterministic: {deterministic_eval}\n"
        f"Outputs saved under {RESULTS_DIR}/"
    )

    total_penalty_steps = int(aggregated_penalties["repeat_penalty"].get("steps", 0.0))
    if total_penalty_steps == 0:
        print("No penalties were incurred during evaluation.")
    else:
        print("Penalties encountered:")
        repeat_stats = aggregated_penalties["repeat_penalty"]
        if repeat_stats["steps"]:
            print(
                f"- Repeat asset: {int(repeat_stats['steps'])} step(s), "
                f"total {repeat_stats['total']:.3f}, assets affected {int(repeat_stats['assets'])}"
            )
    budget_stats = aggregated_penalties["budget_penalty"]
    if budget_stats["steps"]:
        avg_budget_penalty = budget_stats["total"] / budget_stats["steps"]
        avg_norm_unused = budget_stats["normalized_total"] / budget_stats["steps"]
        print(
            f"Average normalized unused budget per step: {avg_norm_unused:.3f} "
            f"(unused-budget penalty contribution: {avg_budget_penalty:.3f})"
        )
        if budget_stats["trimmed_steps"]:
            print(
                f"Trimmed adjustments: {int(budget_stats['trimmed_steps'])} step(s), "
                f"assets trimmed {int(budget_stats['trimmed_assets'])}"
            )
    over_budget_stats = aggregated_penalties["over_budget_penalty"]
    if over_budget_stats["steps"]:
        avg_over_penalty = over_budget_stats["total"] / over_budget_stats["steps"]
        avg_over_amount = over_budget_stats["amount"] / over_budget_stats["steps"]
        print(
            f"Average over-budget amount per offending step: {avg_over_amount:.3f} "
            f"(penalty contribution: {avg_over_penalty:.3f})"
        )

    expert_episode_lengths: List[int] = []
    expert_rewards: List[float] = []
    expert_episode_ids: List[int] = []
    expert_step_years: List[int] = []
    expert_step_indices: List[int] = []
    expert_gas_losses: List[float] = []
    expert_elec_losses: List[float] = []

    for expert_idx, seed in enumerate(episode_seeds):
        actions_log, _, rewards_log, infos_log, _ = rollout_episode(
            expert_action_selector,
            seed=seed,
            episode_idx=expert_idx if num_episodes > 1 else None,
            episode_label="Expert",
        )
        expert_episode_lengths.append(len(actions_log))
        for step_idx, (reward, info) in enumerate(zip(rewards_log, infos_log)):
            expert_rewards.append(float(reward))
            expert_episode_ids.append(expert_idx)
            expert_step_years.append(step_idx * year_step)
            expert_step_indices.append(step_idx)
            expert_gas_losses.append(float(info.get("gas_loss_mean", np.nan)))
            expert_elec_losses.append(float(info.get("elec_loss_mean", np.nan)))

    expert_df = pd.DataFrame(
        {
            "reward": expert_rewards,
            "year": expert_step_years,
            "episode": expert_episode_ids,
            "step_in_episode": expert_step_indices,
            "gas_loss_mean": expert_gas_losses,
            "elec_loss_mean": expert_elec_losses,
        }
    )
    comparison_path = os.path.join(PLOTS_DIR, "reward_comparison_expert.png")
    loss_comparison_path = os.path.join(PLOTS_DIR, "loss_comparison_expert.png")
    if expert_df.empty:
        print("Expert policy produced no steps; skipping reward comparison plot.")
    else:
        plot_reward_comparison(
            ppo_df.loc[:, ["year", "reward", "episode", "step_in_episode"]],
            expert_df,
            output_path=comparison_path,
            multi_episode=num_episodes > 1,
        )
        plot_expected_loss_comparison(
            ppo_df.loc[:, ["year", "episode", "step_in_episode", "gas_loss_mean", "elec_loss_mean"]],
            expert_df.loc[:, ["year", "episode", "step_in_episode", "gas_loss_mean", "elec_loss_mean"]],
            output_path=loss_comparison_path,
            multi_episode=num_episodes > 1,
        )
        print(
            f"Expert policy steps per episode: {expert_episode_lengths}\n"
            f"Reward comparison saved to {comparison_path}\n"
            f"Availability comparison saved to {loss_comparison_path}"
        )


if __name__ == "__main__":
    main()
