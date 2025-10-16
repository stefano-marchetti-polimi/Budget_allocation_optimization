"""
Evaluate a saved PPO policy on the TrialEnv and reproduce the logging/plotting
performed at the end of `optimization_parallel.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

checkpoint = 1

from optimization_parallel import (
    ASSET_NAMES,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    WEIGHT_VECTOR_KEYS,
    TrialEnv,
    env_kwargs,
    year_step,
)


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


def rollout_episode(
    model: PPO, seed: int
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[float],
    List[dict],
    Dict[str, Dict[str, float]],
]:
    """Run one evaluation episode, logging actions and tracking penalties."""
    env = TrialEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)

    actions_log: List[np.ndarray] = []
    obs_log: List[np.ndarray] = []
    rewards_log: List[float] = []
    infos_log: List[dict] = []

    penalty_summary: Dict[str, Dict[str, float]] = {
        "repeat_penalty": {"steps": 0.0, "total": 0.0, "assets": 0.0},
        "cost_penalty": {
            "steps": 0.0,
            "total": 0.0,
            "normalized_total": 0.0,
            "trimmed_steps": 0.0,
            "trimmed_assets": 0.0,
        },
    }

    terminated = False
    truncated = False
    step_idx = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action_array = np.asarray(action, dtype=np.int64)
        actions_log.append(action_array)
        obs_log.append(np.asarray(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        current_weights = np.asarray(env.weights, dtype=np.float32).copy()
        info["weights"] = current_weights
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
        penalty_summary["cost_penalty"]["steps"] += 1
        penalty_summary["cost_penalty"]["total"] += float(info.get("cost_penalty", 0.0))
        penalty_summary["cost_penalty"]["normalized_total"] += float(
            info.get("normalized_cost", 0.0)
        )

        budget_parts: List[str] = []
        if "total_cost" in info:
            budget_parts.append(f"used={info['total_cost']:.3f}")
        if "unused_budget" in info:
            budget_parts.append(f"unused={info['unused_budget']:.3f}")
        if "normalized_cost" in info:
            budget_parts.append(f"normalized={info['normalized_cost']:.3f}")
        trimmed = info.get("trimmed_assets", [])
        if trimmed:
            penalty_summary["cost_penalty"]["trimmed_steps"] += 1
            penalty_summary["cost_penalty"]["trimmed_assets"] += float(len(trimmed))
            penalty_msgs.append(f"trimmed_assets={trimmed}")

        current_year = step_idx * year_step
        print(f"Step {step_idx:02d} | Year {current_year}")
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


def save_logs(
    actions_log: List[np.ndarray],
    obs_log: List[np.ndarray],
    rewards_log: List[float],
    infos_log: List[dict],
) -> None:
    """Persist evaluation traces to CSV and multi-discrete plots."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

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

    penalty_cols = {
        "repeat_penalty": [info.get("repeat_penalty", 0.0) for info in infos_log],
        "cost_penalty": [info.get("cost_penalty", 0.0) for info in infos_log],
        "normalized_cost": [info.get("normalized_cost", 0.0) for info in infos_log],
        "repeat_penalty_assets": [
            json.dumps(info.get("repeat_penalty_assets", [])) for info in infos_log
        ],
        "trimmed_assets": [
            json.dumps(info.get("trimmed_assets", [])) for info in infos_log
        ],
        "total_cost": [info.get("total_cost", 0.0) for info in infos_log],
        "unused_budget": [info.get("unused_budget", 0.0) for info in infos_log],
    }

    df = pd.DataFrame(
        {
            **action_cols,
            **intended_cols,
            **executed_cols,
            **weight_cols,
            **penalty_cols,
            "reward": rewards_log,
            "obs_json": obs_str,
        }
    )
    df.to_csv(os.path.join(RESULTS_DIR, "evaluation_logs.csv"), index=False)

    years_axis = [i * year_step for i in range(len(actions_log))]

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
        ax.plot(years_axis, actions_arr[:, idx], marker="o", linewidth=1.5)
        unique_levels = np.unique(actions_arr[:, idx])
        ax.set_yticks(unique_levels)
        ax.set_ylabel(asset_label)
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Action Levels Over Time")
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.savefig(os.path.join(RESULTS_DIR, "action_over_time.png"))
    plt.close(fig)

    plt.figure(figsize=(10, 4))
    for idx, key in enumerate(WEIGHT_VECTOR_KEYS):
        plt.plot(years_axis, weights_stack[:, idx], marker="o", linewidth=1.5, label=key)
    plt.xlabel("Year")
    plt.ylabel("Weight value")
    plt.title("Decision-Maker Weights Over Time")
    plt.legend(loc="best")
    plt.grid(True, axis="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "weights_over_time.png"))
    plt.close()

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
    fig.savefig(os.path.join(RESULTS_DIR, "action_histogram.png"))
    plt.close(fig)

    plt.figure(figsize=(10, 4))
    plt.plot(years_axis, rewards_log, linewidth=1.5)
    plt.xlabel("Year")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reward_over_time.png"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO policy.")
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Path to the saved policy (with or without .zip). Defaults to results/policy.zip.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=checkpoint,
        help="Set to 1 to load the most recent checkpoint saved in results/checkpoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1042,
        help="Seed passed to the evaluation environment reset.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for loading the policy (e.g., cpu, mps, cuda).",
    )
    args = parser.parse_args()

    if args.checkpoint == 1:
        policy_path = latest_checkpoint_path()
        if args.policy_path is not None:
            print(
                f"Ignoring --policy-path={args.policy_path!r} because --checkpoint=1 was provided."
            )
    else:
        policy_path = resolve_model_path(args.policy_path)
    model = PPO.load(policy_path, device=args.device)

    (
        actions_log,
        obs_log,
        rewards_log,
        infos_log,
        penalty_summary,
    ) = rollout_episode(model, seed=args.seed)
    save_logs(actions_log, obs_log, rewards_log, infos_log)

    print(
        f"Evaluation complete.\n"
        f"Policy: {policy_path}\n"
        f"Steps: {len(actions_log)}\n"
        f"Outputs saved under {RESULTS_DIR}/"
    )

    total_penalty_steps = sum(
        int(summary.get("steps", 0.0))
        for key, summary in penalty_summary.items()
        if key != "cost_penalty"
    )
    if total_penalty_steps == 0:
        print("No penalties were incurred during evaluation.")
    else:
        print("Penalties encountered:")
        if penalty_summary["repeat_penalty"]["steps"]:
            stats = penalty_summary["repeat_penalty"]
            print(
                f"- Repeat asset: {int(stats['steps'])} step(s), "
                f"total {stats['total']:.3f}, assets affected {int(stats['assets'])}"
            )
    cost_stats = penalty_summary["cost_penalty"]
    if cost_stats["steps"]:
        avg_cost_penalty = cost_stats["total"] / cost_stats["steps"]
        avg_norm_cost = cost_stats["normalized_total"] / cost_stats["steps"]
        print(
            f"Average normalized cost per step: {avg_norm_cost:.3f} "
            f"(cost penalty contribution: {avg_cost_penalty:.3f})"
        )
        if cost_stats["trimmed_steps"]:
            print(
                f"Trimmed adjustments: {int(cost_stats['trimmed_steps'])} step(s), "
                f"assets trimmed {int(cost_stats['trimmed_assets'])}"
            )


if __name__ == "__main__":
    main()
