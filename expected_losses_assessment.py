"""
Utility script to inspect the reward components produced by TrialEnv for
specific actions/states. The script lets you fast‑forward to a decision
step (optionally replaying a sequence of prefix actions), clone the
environment, and evaluate multiple candidate action vectors without
rerunning training.  It prints the reward delta, loss components, and
penalties so you can sanity‑check magnitudes.
"""

from __future__ import annotations

import copy
import json
from typing import Iterable, Sequence

import numpy as np

from optimization_parallel import (
    ASSET_NAMES,
    TrialEnv,
    env_kwargs,
    year_step,
)

NUM_ASSETS = len(ASSET_NAMES)


def parse_action(raw: str) -> np.ndarray:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != NUM_ASSETS:
        raise ValueError(
            f"Expected {NUM_ASSETS} values for an action, received {len(parts)} in '{raw}'."
        )
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Action '{raw}' contains non-integer entries.") from exc
    arr = np.asarray(values, dtype=np.int64)
    if np.any(arr < 0):
        raise ValueError(f"Action '{raw}' contains negative levels, which are invalid.")
    return arr


def parse_action_list(items: Sequence[str]) -> list[np.ndarray]:
    return [parse_action(item) for item in items]


def apply_action_sequence(env: TrialEnv, actions: Iterable[np.ndarray]) -> None:
    """Advance the environment by sequentially applying the provided actions."""
    for step_idx, action in enumerate(actions):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            raise RuntimeError(
                f"Environment terminated while applying prefix at step {step_idx}. "
                "Consider using fewer prefix actions or a shorter horizon."
            )
        print(
            f"[prefix] applied step={step_idx} action={action.tolist()} "
            f"reward={info.get('reward_delta', 0.0):.3f}"
        )


def summarize(info: dict) -> dict[str, float]:
    """Extract key scalar metrics from the info dict (with safe defaults)."""
    def get(key: str, default: float = 0.0) -> float:
        value = info.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return {
        "reward_delta": get("reward_delta"),
        "action_gain": get("action_gain"),
        "climate_drift": get("climate_drift"),
        "unused_budget_penalty": get("unused_budget_penalty_signed"),
        "over_budget_penalty": get("over_budget_penalty_signed"),
        "repeat_penalty": get("repeat_penalty", 0.0),
        "total_cost": get("total_cost"),
        "unused_budget": get("unused_budget"),
        "over_budget_amount": get("over_budget_amount"),
        "normalized_unused_budget": get("normalized_unused_budget"),
        "prev_loss": get("prev_loss"),
        "base_loss": get("base_loss"),
        "new_loss": get("new_loss"),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Pretty-print the metrics dict on a single line."""
    ordered_keys = [
        "reward_delta",
        "action_gain",
        "climate_drift",
        "unused_budget_penalty",
        "over_budget_penalty",
        "repeat_penalty",
        "total_cost",
        "unused_budget",
        "over_budget_amount",
        "normalized_unused_budget",
        "prev_loss",
        "base_loss",
        "new_loss",
    ]
    segments: list[str] = []
    for key in ordered_keys:
        value = metrics.get(key)
        segments.append(f"{key}={value:.4f}")
    return " | ".join(segments)


def evaluate_actions(
    base_env: TrialEnv,
    candidate_actions: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, float, dict[str, float]]]:
    """Clone the environment for each action and evaluate its outcome."""
    results: list[tuple[np.ndarray, float, dict[str, float]]] = []
    for action in candidate_actions:
        env_copy = copy.deepcopy(base_env)
        _, reward, terminated, truncated, info = env_copy.step(action)
        metrics = summarize(info)
        results.append((action, float(reward), metrics))

        if terminated or truncated:
            print(
                f"[warning] action {action.tolist()} ended the episode "
                f"(terminated={terminated}, truncated={truncated})."
            )
    return results


SEED = 1042
TARGET_STEP = 2  # decision index (0 -> Year 0, 1 -> Year 5, 2 -> Year 10, ...)
# Actions applied before evaluating candidates; modify the list below.
PREFIX_ACTIONS = [
    "0,0,0,0,0,0,0,0",
    "0,0,0,0,0,0,0,0",
]
# Actions you want to evaluate at TARGET_STEP.
CANDIDATE_ACTIONS = [
    # Do nothing
    "0,0,0,0,0,0,0,0",
    # Upgrade only PV (level 2 for illustration)
    "2,0,0,0,0,0,0,0",
    # Upgrade both substations modestly
    "0,1,1,0,0,0,0,0",
    # Upgrade compressor chain and LNG terminal
    "0,0,0,1,1,1,0,1",
    # Larger mixed upgrade including thermal unit
    "1,1,1,1,0,1,1,1",
]
# Emit JSON payload after the human-readable summary?
PRINT_JSON = False


def main() -> None:
    if not CANDIDATE_ACTIONS:
        raise ValueError("Populate CANDIDATE_ACTIONS with at least one action vector to evaluate.")

    prefix_actions = parse_action_list(PREFIX_ACTIONS)
    candidate_actions = parse_action_list(CANDIDATE_ACTIONS)

    env = TrialEnv(**env_kwargs)
    obs, info = env.reset(seed=SEED)
    print(f"Reset environment. Initial info: {info}")

    # Apply prefix (if any)
    if prefix_actions:
        print(f"Applying {len(prefix_actions)} prefix action(s) before evaluation…")
        apply_action_sequence(env, prefix_actions)

    current_step = len(prefix_actions)
    if TARGET_STEP < current_step:
        raise ValueError(
            f"Requested evaluation at step {TARGET_STEP}, but {current_step} prefix "
            "action(s) have already been applied. Increase --step or use fewer prefix actions."
        )

    # Advance with no-ops if the desired step is later than the prefix length.
    noop = np.zeros(NUM_ASSETS, dtype=np.int64)
    while current_step < TARGET_STEP:
        _, _, terminated, truncated, _ = env.step(noop)
        if terminated or truncated:
            raise RuntimeError(
                f"Episode ended at step {current_step} while advancing to target step {TARGET_STEP}."
            )
        current_step += 1

    print(
        f"Evaluating {len(candidate_actions)} action(s) at step {current_step} "
        f"(year {current_step * year_step})."
    )

    results = evaluate_actions(env, candidate_actions)

    for action, reward, metrics in results:
        print("-" * 80)
        print(f"Action levels: {action.tolist()}")
        print(f"Reward: {reward:.4f}")
        print(format_metrics(metrics))
        print()

    if PRINT_JSON:
        payload = [
            {
                "action": action.tolist(),
                "reward": reward,
                "metrics": metrics,
            }
            for action, reward, metrics in results
        ]
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
