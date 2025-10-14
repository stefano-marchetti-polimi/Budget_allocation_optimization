import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from scipy.stats import genpareto
import pandas as pd
from utils.fragility_curves import (
    fragility_PV,
    fragility_substation,
    fragility_compressor,
    fragility_thermal_unit,
    fragility_LNG_terminal,
)
from utils.repair_times import (compressor_repair_time, substation_repair_time, thermal_unit_repair_time, pv_repair_time, LNG_repair_time)

class TrialEnv(gym.Env):
    """
    Custom environment for sequential (yearly) investment over a network of nodes.

    This variant enforces a **single-improvement** decision at each step: the agent selects exactly one target component (or a no‑investment option), and the **entire yearly budget** is allocated to that target only.

    Action space: `Discrete(N+1)` where 0 = **no investment** (save all budget), and 1..N = invest 100% of the budget in that component.

    Observation normalization (optional): when enabled, the environment returns observations scaled to [0,1].
    """

    def __init__(
        self,
        num_nodes: int,
        years: int = 50,
        weights: list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        budget: float = 100000.0,
        year_step: int = 1,
        initial_wall_height = None,
        area = None,
        seed: Optional[int] = None,
        mc_samples: int = 200_000,
        csv_path: str = 'outputs/coastal_inundation_samples.csv',
        gpd_k: float = 0.8019,
        gpd_sigma: float = 0.1959,
        max_depth: float = 8.0,
        threshold_depth: float = 0.5,
        rise_rate : float = 0.03,
        normalize_observations: bool = True,
    ):
        super().__init__()
        assert num_nodes >= 1, "num_nodes must be >= 1"
        assert years >= 1, "years must be >= 1"
        self.N = int(num_nodes)
        self.T = int(years)
        self.budget = float(budget)
        self.year_step = int(year_step)
        self.weights = weights
        self.rise_rate = float(rise_rate)
        self.area = area
        # This environment currently models 8 components explicitly below
        assert num_nodes == 8, "num_nodes must be 8 to match the hardcoded component mapping (PV, 2 substations, 3 compressors, thermal, LNG)."

        # Store configuration
        self.mc_samples = int(mc_samples)
        self.csv_path = csv_path
        self.gpd_k = float(gpd_k)
        self.gpd_sigma = float(gpd_sigma)
        self.max_depth = float(max_depth)
        self.threshold_depth = float(threshold_depth)

        # Constants for cost-to-height conversion (used in step and obs normalization)
        self.alpha = 4.0   # shape factor (3.5–4.6)
        self.u0 = 1800.0   # €/m cost at 1 m wall height
        self.beta = 1.2    # cost-height exponent

        # Normalization toggle
        self.normalize_observations = bool(normalize_observations)

        # Action space: categorical choice of a single target (or no investment)
        # 0 = no investment; 1..N = invest all budget in that component
        self.action_space = spaces.Discrete(self.N + 1)

        # Observation: [wall_height (N), current_year (1), econ_impact (1), social_impact (1)]
        # Precompute reference wall-height increment for one full-budget step per asset
        self.h_ref = (self.budget / (self.alpha * self.u0 * np.sqrt(self.area))) ** (1.0 / self.beta)
        if self.normalize_observations:
            # All features scaled to [0,1]
            low_obs = np.zeros(self.N + 3, dtype=np.float32)
            high_obs = np.ones(self.N + 3, dtype=np.float32)
        else:
            # Raw ranges (year in [0, T]; impacts nonnegative; wall heights unbounded above)
            high_obs = np.array([1] * self.N + [float(self.T)] + [np.finfo(np.float32).max] * 2, dtype=np.float32)
            low_obs = np.zeros(self.N + 3, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # RNG seed / state
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.rng = self.np_random

        # Internal state
        self._year = 0
        self._econ_impact = 0.0
        self._social_impact = 0.0

        # Track current and previous component-wise means for delta-based reward
        self.gas_loss_mean = 0.0
        self.elec_loss_mean = 0.0
        self.gas_soc_mean = 0.0
        self.elec_soc_mean = 0.0
        self._prev_gas_loss_mean = 0.0
        self._prev_elec_loss_mean = 0.0
        self._prev_gas_soc_mean = 0.0
        self._prev_elec_soc_mean = 0.0

        if initial_wall_height is None:
            self.initial_wall_height = np.zeros(self.N, dtype=np.float32)
        else:
            self.initial_wall_height = np.asarray(initial_wall_height, dtype=np.float32)
            assert self.initial_wall_height.shape == (self.N,), "initial_wall_height must have length N"

        if area is None:
            self.area = np.array([100,150,150,50,50,50,200,300], dtype=np.float32)
        else:
            self.area = np.asarray(area, dtype=np.float32)
            assert self.area.shape == (self.N,), "area must have length N"

        # --- Cache CSV and per-row values for fast nearest-neighbour lookup ---
        df_raw = pd.read_csv(self.csv_path, sep=None, engine='python', header=0, index_col=0)
        df_raw.columns = [float(c) for c in df_raw.columns]
        df = df_raw.apply(pd.to_numeric, errors='coerce')
        self.input_grid = np.array(sorted(df.columns), dtype=float)

        self.ROW_FOR_COMPONENT = {
            'PV': 'P1',
            'Substation1': 'P2',
            'Substation2': 'P3',
            'Compressor1': 'P4',
            'Compressor2': 'P5',
            'Compressor3': 'P6',
            'ThermalUnit': 'P7',
            'LNG': 'P8',
        }
        missing_rows = [r for r in self.ROW_FOR_COMPONENT.values() if r not in df.index]
        if missing_rows:
            raise KeyError(f"Missing rows in CSV for points: {missing_rows}. Available rows: {list(df.index)}")

        # Precompute depth value arrays per component in the same input_grid order
        self.depth_vals = {
            name: df.loc[row].reindex(self.input_grid).to_numpy(dtype=np.float32)
            for name, row in self.ROW_FOR_COMPONENT.items()
        }

        # Pre-build the GPD distribution object
        self.gpd = genpareto(c=self.gpd_k, scale=self.gpd_sigma, loc=0.0)

        self.improvement_height = np.zeros(self.N, dtype=np.float32)

    # ------------------------
    # Helper computations
    # ------------------------
    def _compute_reward(self) -> tuple[float, float, float]:
        # Sample flood heights (exceedances + threshold) with an upper cap via resampling
        samples = self.gpd.rvs(size=self.mc_samples, random_state=self.rng).astype(np.float32)
        mask = samples > self.max_depth
        while np.any(mask):
            samples[mask] = self.gpd.rvs(size=int(np.sum(mask)), random_state=self.rng).astype(np.float32)
            mask = samples > self.max_depth
        samples += np.float32(self.threshold_depth)
        samples += self._year*self.rise_rate

        # Nearest-neighbour lookup helper using precomputed arrays
        def nn_lookup(values_array: np.ndarray, query_inputs: np.ndarray) -> np.ndarray:
            idx = np.searchsorted(self.input_grid, query_inputs, side='left')
            idx_right = np.clip(idx, 0, len(self.input_grid) - 1)
            idx_left = np.clip(idx - 1, 0, len(self.input_grid) - 1)
            choose_left = (idx_right == 0) | (
                (idx_right < len(self.input_grid)) & (np.abs(query_inputs - self.input_grid[idx_left]) <= np.abs(query_inputs - self.input_grid[idx_right]))
            )
            nearest_idx = np.where(choose_left, idx_left, idx_right)
            return values_array[nearest_idx]

        # Per-component hazard depths from cached CSV rows
        depth_PV    = nn_lookup(self.depth_vals['PV'],          samples)
        depth_sub1  = nn_lookup(self.depth_vals['Substation1'], samples)
        depth_sub2  = nn_lookup(self.depth_vals['Substation2'], samples)
        depth_comp1 = nn_lookup(self.depth_vals['Compressor1'], samples)
        depth_comp2 = nn_lookup(self.depth_vals['Compressor2'], samples)
        depth_comp3 = nn_lookup(self.depth_vals['Compressor3'], samples)
        depth_therm = nn_lookup(self.depth_vals['ThermalUnit'], samples)
        depth_LNG   = nn_lookup(self.depth_vals['LNG'],         samples)

        # Convert depths to fragilities via component-specific curves
        PV_fragility = np.clip(fragility_PV(depth_PV,    self.improvement_height[0]), 0.0, 1.0)
        substation1_fragility = np.clip(fragility_substation(depth_sub1, self.improvement_height[1]), 0.0, 1.0)
        substation2_fragility = np.clip(fragility_substation(depth_sub2, self.improvement_height[2]), 0.0, 1.0)
        compressor1_fragility = np.clip(fragility_compressor(depth_comp1, self.improvement_height[3]), 0.0, 1.0)
        compressor2_fragility = np.clip(fragility_compressor(depth_comp2, self.improvement_height[4]), 0.0, 1.0)
        compressor3_fragility = np.clip(fragility_compressor(depth_comp3, self.improvement_height[5]), 0.0, 1.0)
        thermal_unit_fragility = np.clip(fragility_thermal_unit(depth_therm, self.improvement_height[6]), 0.0, 1.0)
        LNG_terminal_fragility = np.clip(fragility_LNG_terminal(depth_LNG, self.improvement_height[7]), 0.0, 1.0)

        # Vectorized Monte Carlo: 1 means operational, 0 means failed
        U = self.rng.random((8, self.mc_samples))
        PV_up         = (U[0] >= PV_fragility).astype(np.uint8)
        substation_1  = (U[1] >= substation1_fragility).astype(np.uint8)
        substation_2  = (U[2] >= substation2_fragility).astype(np.uint8)
        compressor_1  = (U[3] >= compressor1_fragility).astype(np.uint8)
        compressor_2  = (U[4] >= compressor2_fragility).astype(np.uint8)
        compressor_3  = (U[5] >= compressor3_fragility).astype(np.uint8)
        thermal_unit  = (U[6] >= thermal_unit_fragility).astype(np.uint8)
        LNG_terminal_ = (U[7] >= LNG_terminal_fragility).astype(np.uint8)

        # Consumers split
        industrial_consumer_gas = 0.4
        industrial_consumer_electricity = 0.7
        residential_consumer_gas = 0.6
        residential_consumer_electricity = 0.3

        # Booleans for logic operations
        PV_b  = PV_up.astype(bool)
        sub1_b = substation_1.astype(bool)
        sub2_b = substation_2.astype(bool)
        comp1_b = compressor_1.astype(bool)
        comp2_b = compressor_2.astype(bool)
        comp3_b = compressor_3.astype(bool)
        therm_b = thermal_unit.astype(bool)
        LNG_b   = LNG_terminal_.astype(bool)

        # === Repair times (hours or days depending on the function) ===
        # Vectorize the repair time functions to accept numpy arrays
        pv_rt_fn = np.vectorize(lambda d: pv_repair_time(d, rng=self.rng), otypes=[np.float32])
        sub_rt_fn = np.vectorize(lambda d: substation_repair_time(d, rng=self.rng), otypes=[np.float32])
        comp_rt_fn = np.vectorize(lambda d: compressor_repair_time(d, rng=self.rng), otypes=[np.float32])
        therm_rt_fn = np.vectorize(lambda d: thermal_unit_repair_time(d, rng=self.rng), otypes=[np.float32])
        LNG_rt_fn = np.vectorize(lambda d: LNG_repair_time(d, rng=self.rng), otypes=[np.float32])

        # Sample per-sample repair times only where the component failed; 0 otherwise
        t_PV    = np.where(~PV_b,    pv_rt_fn(depth_PV),    0.0).astype(np.float32)
        t_sub1  = np.where(~sub1_b,  sub_rt_fn(depth_sub1), 0.0).astype(np.float32)
        t_sub2  = np.where(~sub2_b,  sub_rt_fn(depth_sub2), 0.0).astype(np.float32)
        t_comp1 = np.where(~comp1_b, comp_rt_fn(depth_comp1), 0.0).astype(np.float32)
        t_comp2 = np.where(~comp2_b, comp_rt_fn(depth_comp2), 0.0).astype(np.float32)
        t_comp3 = np.where(~comp3_b, comp_rt_fn(depth_comp3), 0.0).astype(np.float32)
        t_therm = np.where(~therm_b, therm_rt_fn(depth_therm), 0.0).astype(np.float32)
        t_LNG   = np.where(~LNG_b,   LNG_rt_fn(depth_LNG),   0.0).astype(np.float32)

        # Normalize all repair times by a fixed maximum (e.g., 1200) so they are in [0,1]
        TIME_NORM = np.float32(1200.0)
        t_PV    = (t_PV    / (TIME_NORM)).astype(np.float32)
        t_sub1  = (t_sub1  / (TIME_NORM)).astype(np.float32)
        t_sub2  = (t_sub2  / (TIME_NORM)).astype(np.float32)
        t_comp1 = (t_comp1 / (TIME_NORM)).astype(np.float32)
        t_comp2 = (t_comp2 / (TIME_NORM)).astype(np.float32)
        t_comp3 = (t_comp3 / (TIME_NORM)).astype(np.float32)
        t_therm = (t_therm / (TIME_NORM)).astype(np.float32)
        t_LNG   = (t_LNG   / (TIME_NORM)).astype(np.float32)

        # Source services
        PV_service  = PV_b.copy()
        LNG_service = LNG_b.copy()
        sub2_service = sub2_b & PV_service

        # Independent feeder availability (decoupled services)
        p_feeder_avail_c1 = 0.8
        p_feeder_avail_c2 = 0.8
        p_feeder_avail_c3 = 0.8
        feed_s1_c1 = (self.rng.random(self.mc_samples) < p_feeder_avail_c1)
        feed_s2_c1 = (self.rng.random(self.mc_samples) < p_feeder_avail_c1)
        feed_s1_c2 = (self.rng.random(self.mc_samples) < p_feeder_avail_c2)
        feed_s2_c2 = (self.rng.random(self.mc_samples) < p_feeder_avail_c2)
        feed_s1_c3 = (self.rng.random(self.mc_samples) < p_feeder_avail_c3)
        feed_s2_c3 = (self.rng.random(self.mc_samples) < p_feeder_avail_c3)

        sub1_service = sub1_b & False
        for _ in range(10):
            power_ok1 = (sub1_service & feed_s1_c1) | (sub2_service & feed_s2_c1)
            power_ok2 = (sub1_service & feed_s1_c2) | (sub2_service & feed_s2_c2)
            power_ok3 = (sub1_service & feed_s1_c3) | (sub2_service & feed_s2_c3)

            comp1_service = comp1_b & power_ok1 & LNG_service
            comp2_service = comp2_b & power_ok2 & LNG_service
            comp3_service = comp3_b & power_ok3 & LNG_service

            thermal_unit_service = therm_b & LNG_service & comp3_service
            sub1_new = sub1_b & thermal_unit_service
            if np.array_equal(sub1_new, sub1_service):
                sub1_service = sub1_new
                break
            sub1_service = sub1_new

        # Consumer services
        industrial_gas_service = LNG_service & comp1_service
        residential_gas_service = LNG_service & comp2_service
        industrial_elec_service = LNG_service & comp3_service & thermal_unit_service & sub1_service
        residential_elec_service = PV_service & sub2_service

        ig = industrial_gas_service.astype(float)
        rg = residential_gas_service.astype(float)
        ie = industrial_elec_service.astype(float)
        re = residential_elec_service.astype(float)

        # === Service-level maximum repair times (if multiple components cause the outage, use the longest repair time) ===
        # Gas (industrial): LNG + Comp1 + power chain (Sub1/Sub2/Thermal/Comp3)
        t_max_ig = np.maximum.reduce([t_LNG, t_comp1, t_sub1, t_sub2, t_therm, t_comp3])
        # Gas (residential): LNG + Comp2 + power chain (Sub1/Sub2/Thermal/Comp3)
        t_max_rg = np.maximum.reduce([t_LNG, t_comp2, t_sub1, t_sub2, t_therm, t_comp3])
        # Electricity (industrial): LNG + Comp3 + Thermal + Sub1
        t_max_ie = np.maximum.reduce([t_LNG, t_comp3, t_therm, t_sub1])
        # Electricity (residential): PV + Sub2
        t_max_re = np.maximum.reduce([t_PV, t_sub2])

        # === Time-weighted losses and social impacts ===
        # Multiply each outage indicator by the corresponding maximum repair time.
        gas_loss_samples = (1.0 - ig) * industrial_consumer_gas * t_max_ig + (1.0 - rg) * residential_consumer_gas * t_max_rg
        electricity_loss_samples = (1.0 - ie) * industrial_consumer_electricity * t_max_ie + (1.0 - re) * residential_consumer_electricity * t_max_re
        gas_social_samples = (1.0 - rg) * t_max_rg
        electricity_social_samples = (1.0 - re) * t_max_re

        w_gas, w_electricity, w_gas_loss, w_gas_social, w_electricity_loss, w_electricity_social = self.weights

        self.gas_loss_mean = float(gas_loss_samples.mean())
        self.elec_loss_mean = float(electricity_loss_samples.mean())
        self.gas_soc_mean = float(gas_social_samples.mean())
        self.elec_soc_mean = float(electricity_social_samples.mean())

        # Compute deltas (positive if impacts decreased)
        d_gas_loss = self._prev_gas_loss_mean - self.gas_loss_mean
        d_elec_loss = self._prev_elec_loss_mean - self.elec_loss_mean
        d_gas_soc  = self._prev_gas_soc_mean - self.gas_soc_mean
        d_elec_soc = self._prev_elec_soc_mean - self.elec_soc_mean

        w_gas, w_electricity, w_gas_loss, w_gas_social, w_electricity_loss, w_electricity_social = self.weights
        reward_mean = (
            w_gas * (d_gas_loss * w_gas_loss + d_gas_soc * w_gas_social)
            + w_electricity * (d_elec_loss * w_electricity_loss + d_elec_soc * w_electricity_social)
        )

        # Update previous means for next step
        self._prev_gas_loss_mean = self.gas_loss_mean
        self._prev_elec_loss_mean = self.elec_loss_mean
        self._prev_gas_soc_mean = self.gas_soc_mean
        self._prev_elec_soc_mean = self.elec_soc_mean

        econ_impact = self.gas_loss_mean + self.elec_loss_mean
        social_impact = self.gas_soc_mean + self.elec_soc_mean

        return reward_mean, econ_impact, social_impact

    def _obs(self) -> np.ndarray:
        if self.normalize_observations:
            # Normalize wall heights by (h_ref * T), clip to [0,1]
            denom = (self.h_ref * self.T).astype(np.float32)
            denom = np.where(denom <= 0.0, 1.0, denom)
            wh = np.clip(self.wall_height / denom, 0.0, 1.0).astype(np.float32)
            yr = np.float32(self._year / self.T)
            econ = np.float32(min(self._econ_impact / 2.0, 1.0))
            soc = np.float32(min(self._social_impact / 2.0, 1.0))
            tail = np.array([yr, econ, soc], dtype=np.float32)
            return np.concatenate([wh, tail])
        else:
            return np.concatenate([
                self.wall_height.astype(np.float32),
                np.array([self._year, self._econ_impact, self._social_impact], dtype=np.float32),
            ])

    # ------------------------
    # Gym API
    # ------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            self.rng = self.np_random
        # Start from new
        self.wall_height = self.initial_wall_height.copy()
        self.improvement_height = np.zeros(self.N, dtype=np.float32)
        self._year = 0
        _reward0, self._econ_impact, self._social_impact = self._compute_reward()
        # Set previous means to current so first step has zero delta reward
        self._prev_gas_loss_mean = self.gas_loss_mean
        self._prev_elec_loss_mean = self.elec_loss_mean
        self._prev_gas_soc_mean = self.gas_soc_mean
        self._prev_elec_soc_mean = self.elec_soc_mean
        return self._obs(), {}

    def step(self, action):
        # the assumption is that the improvements are always walls. so we estimate the height of the walls based on the investment.
        # the fragility is shifted using the hegith of the wall.

        # Single-improvement decision: choose exactly one target (or skip)
        # Preferred input: integer in [0, N] (Discrete action).
        # Backward-compat: if a vector is passed, treat it as logits/scores and take argmax.
        choice = None
        if np.isscalar(action) or (isinstance(action, (np.ndarray, list)) and np.array(action).ndim == 0):
            choice = int(action)
        else:
            scores = np.asarray(action, dtype=np.float32).reshape(-1)
            if scores.shape[0] == self.N:
                # assume no-investment score = 0
                scores = np.concatenate([np.array([0.0], dtype=np.float32), scores], axis=0)
            assert scores.shape[0] == self.N + 1, f"Expected {self.N+1} scores (including no-investment), got {scores.shape[0]}"
            choice = int(np.argmax(scores))

        # Clamp to valid range
        choice = int(max(0, min(self.N, choice)))

        # Build one-hot allocation over assets (length N); all budget to the chosen asset, or none if choice==0
        applied_action = np.zeros(self.N, dtype=np.float32)
        if choice > 0:
            applied_action[choice - 1] = 1.0
            unspent_share = 0.0
        else:
            unspent_share = 1.0

        # Improvement
        self.improvement_height = self.improvement_height + (self.budget * applied_action / (self.alpha * self.u0 * np.sqrt(self.area))) ** (1.0 / self.beta)
        self.improvement_height = np.maximum(self.improvement_height, 0.0)

        self.wall_height = self.improvement_height + self.initial_wall_height

        # Advance time
        self._year += self.year_step

        # Recompute impacts for current state
        reward, self._econ_impact, self._social_impact = self._compute_reward()

        terminated = bool(self._year >= self.T)
        truncated = False
        info = {
            "econ_impact": self._econ_impact,
            "social_impact": self._social_impact,
            "applied_action": applied_action.astype(np.float32),
            "choice": int(choice),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        print(
            f"Year {self._year}/{self.T} | EconImpact={self._econ_impact:.3f} | SocialImpact={self._social_impact:.3f}"
        )

    def close(self):
        pass
