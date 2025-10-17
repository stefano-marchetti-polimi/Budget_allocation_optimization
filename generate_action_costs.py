"""Generate a CSV table listing the construction cost for every asset/level."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from optimization_parallel import ASSET_NAMES, TrialEnv, env_kwargs


def main() -> None:
    env = TrialEnv(**env_kwargs)
    levels = env.height_levels
    records = []

    # Baseline vector used to isolate the cost contribution per asset.
    zero = np.zeros(env.N, dtype=np.float32)

    for asset_idx, asset_name in enumerate(ASSET_NAMES):
        for level_idx, height in enumerate(levels):
            vec = zero.copy()
            vec[asset_idx] = height
            cost = float(env._compute_costs(vec)[asset_idx])
            records.append(
                {
                    "asset": asset_name,
                    "level_index": level_idx,
                    "height_increment_m": float(height),
                    "cost": cost,
                }
            )

    df = pd.DataFrame(records)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "action_costs.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved cost table to {output_path} ({df.shape[0]} rows).")


if __name__ == "__main__":
    main()
