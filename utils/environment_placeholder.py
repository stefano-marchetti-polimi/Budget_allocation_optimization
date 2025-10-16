import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Mapping, Optional, Sequence
import pandas as pd
from utils.fragility_curves import (
    fragility_PV,
    fragility_substation,
    fragility_compressor,
    fragility_thermal_unit,
    fragility_LNG_terminal,
)
from utils.repair_times import (compressor_repair_time, substation_repair_time, thermal_unit_repair_time, pv_repair_time, LNG_repair_time)
from utils.copula_sampler import sample_flood

class TrialEnv(gym.Env):
    """
    Custom environment for sequential (yearly) investment over a network of nodes.

    At each decision step the agent chooses wall-height increments for every component from
    the discrete set {0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0} meters. Positive choices incur
    component-specific construction costs; if their sum exceeds the yearly budget, the most
    expensive upgrades are dropped until the action is feasible. Reapplying an upgrade to the
    same component on consecutive steps is cancelled and penalised. Observations can optionally
    be normalised to lie in [0, 1]. The reward is shaped as the reduction in expected losses due
    to the action minus a normalized cost term, so inaction while sea level rises naturally
    incurs a penalty through worsening impacts.
    """

    def __init__(
        self,
        num_nodes: int,
        years: int = 50,
        weights: Sequence[float] = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        weight_years: Optional[Sequence[int]] = None,
        cost_per_meter: float = 10000.0,
        budget: float = 100000.0,
        year_step: int = 1,
        initial_wall_height = None,
        area = None,
        seed: Optional[int] = None,
        mc_samples: int = 500_000,
        csv_path: str = 'outputs/coastal_inundation_samples.csv',
        copula_theta: float = 3.816289,
        max_depth: float = 8.0,
        max_duration: float = 100,
        threshold_depth: float = 0.5,
        rise_rate: float = 0.02,
        sea_level_scenarios: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
        climate_scenario: str = "All",
        normalize_observations: bool = True,
        maximum_repair_time: float = 40.75,
        repeat_asset_penalty: float = -1.0,
        reward_scale: float = 5000.0,
        cost_weight: float = 2.0,
    ):
        super().__init__()
        assert num_nodes >= 1, "num_nodes must be >= 1"
        assert years >= 1, "years must be >= 1"
        self.N = int(num_nodes)
        self.T = int(years)
        self.budget = float(budget)
        self.year_step = int(year_step)
        weights_array = np.asarray(weights, dtype=np.float32)
        if weights_array.ndim == 1:
            if weights_array.shape[0] != 6:
                raise ValueError(f"weights must contain 6 entries, received shape {weights_array.shape}.")
            weights_array = weights_array.reshape(1, -1)
        elif weights_array.ndim == 2:
            if weights_array.shape[1] != 6:
                raise ValueError(
                    f"Weight schedule must provide 6 entries per step, received shape {weights_array.shape}."
                )
        else:
            raise ValueError("weights must be a 1D sequence of length 6 or a 2D schedule with 6 columns.")

        if np.isnan(weights_array).any():
            raise ValueError("Weight schedule contains NaN values.")

        required_points = ((self.T + self.year_step - 1) // self.year_step) + 1
        if weights_array.shape[0] < required_points:
            raise ValueError(
                f"Weight schedule length {weights_array.shape[0]} insufficient for "
                f"{self.T} years with step {self.year_step}."
            )

        self._weight_schedule = weights_array.astype(np.float32)
        self._current_weight_index = 0
        self.weights = self._weight_schedule[0].copy()

        self._weight_years = None
        self._weight_base_year = 0
        if weight_years is not None:
            weight_years_array = np.asarray(list(weight_years), dtype=np.int64)
            if weight_years_array.shape[0] != self._weight_schedule.shape[0]:
                raise ValueError(
                    f"weight_years length {weight_years_array.shape[0]} "
                    f"does not match weight schedule length {self._weight_schedule.shape[0]}."
                )
            if weight_years_array.shape[0] > 1 and np.any(np.diff(weight_years_array) < 0):
                raise ValueError("weight_years must be sorted in non-decreasing order.")
            self._weight_years = weight_years_array
            self._weight_base_year = int(weight_years_array[0])

        self.rise_rate = float(rise_rate)
        self._decision_points = required_points
        self._num_decision_steps = max(self._decision_points - 1, 0)
        self._sea_level_offsets: Optional[np.ndarray] = None
        self._sea_level_deltas: Optional[np.ndarray] = None
        self._active_climate_scenario: Optional[str] = None
        self._climate_scenario_selected: Optional[str] = None
        if sea_level_scenarios is not None:
            if not isinstance(sea_level_scenarios, Mapping) or not sea_level_scenarios:
                raise ValueError("sea_level_scenarios must be a non-empty mapping when provided.")
            expected_len = self._decision_points
            scenario_map: dict[str, dict[str, np.ndarray]] = {}
            for name, params in sea_level_scenarios.items():
                if not isinstance(params, Mapping):
                    raise ValueError(f"Sea level parameters for scenario '{name}' must be a mapping.")
                processed: dict[str, np.ndarray] = {}
                for key in ("mu", "sigma", "lower", "upper"):
                    if key not in params:
                        raise ValueError(f"Scenario '{name}' missing '{key}' entries.")
                    arr = np.asarray(params[key], dtype=np.float32).reshape(-1)
                    if arr.size != expected_len:
                        raise ValueError(
                            f"Scenario '{name}' expected {expected_len} entries for '{key}', received {arr.size}."
                        )
                    if key == "sigma" and np.any(arr < 0.0):
                        raise ValueError(f"Scenario '{name}' contains negative sigma values.")
                    arr_copy = arr.copy()
                    arr_copy.setflags(write=False)
                    processed[key] = arr_copy
                scenario_map[str(name)] = processed
            if not scenario_map:
                raise ValueError("sea_level_scenarios mapping is empty.")
            preference = str(climate_scenario).strip()
            if not preference:
                raise ValueError("climate_scenario must be a non-empty string.")
            if preference != "All" and preference not in scenario_map:
                available = sorted(scenario_map.keys())
                raise ValueError(
                    f"climate_scenario '{preference}' not found in available scenarios: {available}"
                )
            self._sea_level_scenarios = scenario_map
            self._climate_scenario_selected = preference
        else:
            self._sea_level_scenarios = None
            self._climate_scenario_selected = None
        self.maximum_repair_time = float(maximum_repair_time)
        self.repeat_asset_penalty = float(repeat_asset_penalty)
        self.max_duration = float(max_duration)
        self.reward_scale = float(reward_scale)
        self.cost_weight = float(cost_weight)
        # This environment currently models 8 components explicitly below
        assert num_nodes == 8, "num_nodes must be 8 to match the hardcoded component mapping (PV, 2 substations, 3 compressors, thermal, LNG)."

        # Store configuration
        self.mc_samples = int(mc_samples)
        self.csv_path = csv_path
        self.copula_theta = float(copula_theta)
        self.max_depth = float(max_depth)
        self.threshold_depth = float(threshold_depth)

        # Constants for cost-to-height conversion (used in step and obs normalization)
        self.alpha = 4.0   # shape factor (3.5–4.6)
        self.u0 = 1800.0   # €/m cost at 1 m wall height
        self.beta = 1.2    # cost-height exponent

        # Normalization toggle
        self.normalize_observations = bool(normalize_observations)

        if area is None:
            area_array = np.array([100,150,150,50,50,50,200,300], dtype=np.float32)
        else:
            area_array = np.asarray(area, dtype=np.float32)
            assert area_array.shape == (self.N,), "area must have length N"
        self.area = area_array

        # Placeholder supply/people values per compressor and substation (user can adjust later)
        self.compressor_order = ["Compressor1", "Compressor2", "Compressor3"]
        compressor_defaults = {
            "Compressor1": {"gas_supply": 50.0, "people": 0.0},
            "Compressor2": {"gas_supply": 35.0, "people": 10000.0},
            "Compressor3": {"gas_supply": 25.0, "people": 0.0},
        }
        self.substation_order = ["Substation1", "Substation2"]
        substation_defaults = {
            "Substation1": {"power_supply": 70.0, "people": 0.0},
            "Substation2": {"power_supply": 30.0, "people": 15000.0},
        }

        self.compressor_gas_supply = np.array(
            [compressor_defaults[name]["gas_supply"] for name in self.compressor_order],
            dtype=np.float32,
        )
        self.compressor_people = np.array(
            [compressor_defaults[name]["people"] for name in self.compressor_order],
            dtype=np.float32,
        )
        self.substation_power_supply = np.array(
            [substation_defaults[name]["power_supply"] for name in self.substation_order],
            dtype=np.float32,
        )
        self.substation_people = np.array(
            [substation_defaults[name]["people"] for name in self.substation_order],
            dtype=np.float32,
        )

        self.total_gas_supply = float(self.compressor_gas_supply.sum())
        self.total_gas_people = float(self.compressor_people.sum())
        self.total_power_supply = float(self.substation_power_supply.sum())
        self.total_power_people = float(self.substation_people.sum())

        if self.total_gas_supply > 0.0:
            self.compressor_supply_fraction = (self.compressor_gas_supply / self.total_gas_supply).astype(np.float32)
        else:
            self.compressor_supply_fraction = np.zeros_like(self.compressor_gas_supply, dtype=np.float32)

        if self.total_gas_people > 0.0:
            self.compressor_people_fraction = (self.compressor_people / self.total_gas_people).astype(np.float32)
        else:
            self.compressor_people_fraction = np.zeros_like(self.compressor_people, dtype=np.float32)

        if self.total_power_supply > 0.0:
            self.substation_supply_fraction = (self.substation_power_supply / self.total_power_supply).astype(np.float32)
        else:
            self.substation_supply_fraction = np.zeros_like(self.substation_power_supply, dtype=np.float32)

        if self.total_power_people > 0.0:
            self.substation_people_fraction = (self.substation_people / self.total_power_people).astype(np.float32)
        else:
            self.substation_people_fraction = np.zeros_like(self.substation_people, dtype=np.float32)

        if initial_wall_height is None:
            self.initial_wall_height = np.zeros(self.N, dtype=np.float32)
        else:
            self.initial_wall_height = np.asarray(initial_wall_height, dtype=np.float32)
            assert self.initial_wall_height.shape == (self.N,), "initial_wall_height must have length N"

        self.height_levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
        self.cost_prefactor = (self.alpha * self.u0 * np.sqrt(self.area.astype(np.float32))).astype(np.float32)

        # Multi-discrete action: pick a height increment option for each component
        action_sizes = np.full(self.N, len(self.height_levels), dtype=np.int64)
        self.action_space = spaces.MultiDiscrete(action_sizes)

        # Observation: [wall_height (N), current_year (1), econ_impact (1), social_impact (1)]
        # Reference scale for normalisation: maximum selectable increment
        self.h_ref = np.full(self.N, self.height_levels[-1], dtype=np.float32)
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
        self._last_positive_mask = np.zeros(self.N, dtype=bool)

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
        self.improvement_height = np.zeros(self.N, dtype=np.float32)
        self._base_heights: Optional[np.ndarray] = None
        self._base_durations: Optional[np.ndarray] = None
        self._prev_metrics: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._prev_loss: float = 0.0

    def _set_weight_index(self, index: int) -> None:
        """Update the active weight vector according to the schedule."""
        if self._weight_schedule.shape[0] == 0:
            raise ValueError("Weight schedule is empty.")
        max_idx = self._weight_schedule.shape[0] - 1
        safe_index = int(index)
        if safe_index < 0:
            safe_index = 0
        elif safe_index > max_idx:
            safe_index = max_idx
        self._current_weight_index = safe_index
        self.weights = self._weight_schedule[safe_index]

    # ------------------------
    # Helper computations
    # ------------------------
    def _compute_costs(self, height_deltas: np.ndarray) -> np.ndarray:
        """Return per-component construction costs for the proposed height increments."""
        deltas = np.asarray(height_deltas, dtype=np.float32)
        non_negative = np.clip(deltas, a_min=0.0, a_max=None)
        costs = self.cost_prefactor * np.power(non_negative, self.beta)
        return costs.astype(np.float32)

    def _sample_truncated_normal(self, mu: float, sigma: float, lower: float, upper: float) -> np.float32:
        """Sample from a truncated normal via simple rejection sampling."""
        low = float(lower)
        high = float(upper)
        if low > high:
            low, high = high, low
        if not np.isfinite(mu):
            mu = 0.0
        if sigma <= 0.0 or not np.isfinite(sigma):
            return np.float32(np.clip(mu, low, high))
        for _ in range(64):
            draw = float(self.rng.normal(mu, sigma))
            if low <= draw <= high:
                return np.float32(draw)
        return np.float32(np.clip(mu, low, high))

    def _sample_sea_level_path(self) -> None:
        """Sample cumulative sea level offsets for the upcoming episode."""
        if self._sea_level_scenarios is None:
            self._sea_level_offsets = None
            self._sea_level_deltas = None
            self._active_climate_scenario = None
            return

        preference = self._climate_scenario_selected or "All"
        if preference == "All":
            scenario_name = str(self.rng.choice(list(self._sea_level_scenarios.keys())))
        else:
            scenario_name = preference

        params = self._sea_level_scenarios[scenario_name]
        mu = params["mu"]
        sigma = params["sigma"]
        lower = params["lower"]
        upper = params["upper"]
        n_points = mu.shape[0]
        levels = np.empty(n_points, dtype=np.float32)
        prev_value: Optional[float] = None
        for idx in range(n_points):
            low = float(lower[idx])
            high = float(upper[idx])
            if prev_value is not None:
                low = max(low, prev_value)
            if high < low:
                high = low
            sampled = float(
                self._sample_truncated_normal(
                    float(mu[idx]),
                    float(sigma[idx]),
                    low,
                    high,
                )
            )
            if prev_value is not None and sampled < prev_value:
                sampled = prev_value
            levels[idx] = np.float32(sampled)
            prev_value = sampled

        base_level = levels[0]
        offsets = (levels - base_level).astype(np.float32)
        if offsets.size <= 1:
            deltas = np.zeros(0, dtype=np.float32)
        else:
            deltas = np.diff(offsets).astype(np.float32)

        self._sea_level_deltas = deltas
        self._sea_level_offsets = offsets
        self._active_climate_scenario = scenario_name

    def _sea_level_offset_for_year(self, year: float) -> np.float32:
        """Return the cumulative sea level adjustment for the provided elapsed year."""
        if self._sea_level_offsets is None:
            return np.float32(year * self.rise_rate)
        if self.year_step <= 0:
            return np.float32(self._sea_level_offsets[-1])
        idx = int(np.floor(year / self.year_step))
        idx = max(0, min(idx, self._sea_level_offsets.shape[0] - 1))
        return np.float32(self._sea_level_offsets[idx])

    def _generate_hazard_samples(self) -> None:
        """Draw and cache the Monte Carlo flood samples reused across the episode."""
        seed_val = int(self.rng.integers(0, 2**31 - 1))
        df_cop = sample_flood(self.mc_samples, self.copula_theta, seed=seed_val)
        heights = df_cop["height"].to_numpy(dtype=np.float32)
        durations = df_cop["duration"].to_numpy(dtype=np.float32)

        resample_count = 0
        while True:
            mask = (heights > self.max_depth) | (durations > self.max_duration)
            if not np.any(mask) or resample_count > 10:
                break
            n_bad = int(mask.sum())
            resample_seed = int(self.rng.integers(0, 2**31 - 1))
            df_res = sample_flood(n_bad, self.copula_theta, seed=resample_seed)
            heights[mask] = df_res["height"].to_numpy(dtype=np.float32)
            durations[mask] = df_res["duration"].to_numpy(dtype=np.float32)
            resample_count += 1

        self._base_heights = heights
        self._base_durations = durations

    def _compute_metrics(self, improvement_height: np.ndarray, year: float) -> tuple[float, float, float, float]:
        if self._base_heights is None or self._base_durations is None:
            self._generate_hazard_samples()

        improvement_height = np.asarray(improvement_height, dtype=np.float32)
        offset = self._sea_level_offset_for_year(year)
        samples = self._base_heights + offset
        durations = self._base_durations

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
        PV_fragility = np.clip(fragility_PV(depth_PV,    improvement_height[0]), 0.0, 1.0)
        substation1_fragility = np.clip(fragility_substation(depth_sub1, improvement_height[1]), 0.0, 1.0)
        substation2_fragility = np.clip(fragility_substation(depth_sub2, improvement_height[2]), 0.0, 1.0)
        compressor1_fragility = np.clip(fragility_compressor(depth_comp1, improvement_height[3]), 0.0, 1.0)
        compressor2_fragility = np.clip(fragility_compressor(depth_comp2, improvement_height[4]), 0.0, 1.0)
        compressor3_fragility = np.clip(fragility_compressor(depth_comp3, improvement_height[5]), 0.0, 1.0)
        thermal_unit_fragility = np.clip(fragility_thermal_unit(depth_therm, improvement_height[6]), 0.0, 1.0)
        LNG_terminal_fragility = np.clip(fragility_LNG_terminal(depth_LNG, improvement_height[7]), 0.0, 1.0)

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

        # Sample repair times (exclude flood duration for sequential restoration logic)
        repair_PV    = np.where(~PV_b,    pv_rt_fn(depth_PV),    0.0).astype(np.float32)
        repair_sub1  = np.where(~sub1_b,  sub_rt_fn(depth_sub1), 0.0).astype(np.float32)
        repair_sub2  = np.where(~sub2_b,  sub_rt_fn(depth_sub2), 0.0).astype(np.float32)
        repair_comp1 = np.where(~comp1_b, comp_rt_fn(depth_comp1), 0.0).astype(np.float32)
        repair_comp2 = np.where(~comp2_b, comp_rt_fn(depth_comp2), 0.0).astype(np.float32)
        repair_comp3 = np.where(~comp3_b, comp_rt_fn(depth_comp3), 0.0).astype(np.float32)
        repair_therm = np.where(~therm_b, therm_rt_fn(depth_therm), 0.0).astype(np.float32)
        repair_LNG   = np.where(~LNG_b,   LNG_rt_fn(depth_LNG),   0.0).astype(np.float32)
        flood_duration = durations.astype(np.float32)

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

        max_event_time = np.float32(self.max_duration + self.maximum_repair_time)
        max_event_time = np.clip(max_event_time, a_min=np.float32(1e-6), a_max=None)

        gas_services = np.stack([comp1_service, comp2_service, comp3_service])
        gas_service_times = np.stack([
            np.where(
                ~industrial_gas_service,
                flood_duration + repair_LNG + repair_comp1 + repair_sub1 + repair_sub2 + repair_therm + repair_comp3,
                0.0,
            ),
            np.where(
                ~residential_gas_service,
                flood_duration + repair_LNG + repair_comp2 + repair_sub1 + repair_sub2 + repair_therm + repair_comp3,
                0.0,
            ),
            np.where(
                ~comp3_service,
                flood_duration + repair_LNG + repair_comp3 + repair_sub1 + repair_sub2 + repair_therm,
                0.0,
            ),
        ]).astype(np.float32)
        gas_time_ratio = np.clip(gas_service_times / max_event_time, a_min=0.0, a_max=1.0)
        gas_outage_fraction = np.logical_not(gas_services).astype(np.float32) * self.compressor_supply_fraction[:, None]
        gas_loss_samples = (gas_outage_fraction * gas_time_ratio).sum(axis=0)
        gas_social_fraction = np.logical_not(gas_services).astype(np.float32) * self.compressor_people_fraction[:, None]
        gas_social_samples = (gas_social_fraction * gas_time_ratio).sum(axis=0)

        power_services = np.stack([industrial_elec_service, residential_elec_service])
        power_service_times = np.stack([
            np.where(
                ~industrial_elec_service,
                flood_duration + repair_LNG + repair_comp3 + repair_therm + repair_sub1,
                0.0,
            ),
            np.where(
                ~residential_elec_service,
                flood_duration + repair_PV + repair_sub2,
                0.0,
            ),
        ]).astype(np.float32)
        power_time_ratio = np.clip(power_service_times / max_event_time, a_min=0.0, a_max=1.0)
        power_outage_fraction = np.logical_not(power_services).astype(np.float32) * self.substation_supply_fraction[:, None]
        electricity_loss_samples = (power_outage_fraction * power_time_ratio).sum(axis=0)
        power_social_fraction = np.logical_not(power_services).astype(np.float32) * self.substation_people_fraction[:, None]
        electricity_social_samples = (power_social_fraction * power_time_ratio).sum(axis=0)
        gas_loss_mean = float(gas_loss_samples.mean())
        elec_loss_mean = float(electricity_loss_samples.mean())
        gas_social_mean = float(gas_social_samples.mean())
        elec_social_mean = float(electricity_social_samples.mean())
        return (
            gas_loss_mean,
            elec_loss_mean,
            gas_social_mean,
            elec_social_mean,
        )

    def _compute_loss(self, metrics: tuple[float, float, float, float]) -> float:
        gas_loss, elec_loss, gas_social, elec_social = metrics
        (
            w_gas,
            w_electricity,
            w_gas_loss,
            w_gas_social,
            w_electricity_loss,
            w_electricity_social,
        ) = self.weights
        return (
            w_gas * (gas_loss * w_gas_loss + gas_social * w_gas_social)
            + w_electricity * (
                elec_loss * w_electricity_loss + elec_social * w_electricity_social
            )
        )

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
        self._set_weight_index(0)
        self._sample_sea_level_path()
        self._base_heights = None
        self._base_durations = None
        metrics = self._compute_metrics(self.improvement_height, year=self._year)
        (
            self.gas_loss_mean,
            self.elec_loss_mean,
            self.gas_soc_mean,
            self.elec_soc_mean,
        ) = metrics
        self._econ_impact = self.gas_loss_mean + self.elec_loss_mean
        self._social_impact = self.gas_soc_mean + self.elec_soc_mean
        self._last_positive_mask = np.zeros(self.N, dtype=bool)
        self._prev_metrics = metrics
        self._prev_loss = self._compute_loss(metrics)
        info = {}
        if self._active_climate_scenario is not None and self._sea_level_offsets is not None:
            info["climate_scenario"] = self._active_climate_scenario
            info["sea_level_offset"] = float(self._sea_level_offsets[0])
        return self._obs(), info

    def step(self, action):
        # Improvements are wall-height increments chosen from discrete levels per component.
        action_array = np.asarray(action)
        if action_array.size != self.N:
            raise ValueError(f"Expected action with {self.N} entries, received shape {action_array.shape}.")
        action_array = action_array.astype(np.int64).reshape(self.N)
        np.clip(action_array, 0, len(self.height_levels) - 1, out=action_array)

        intended_heights = self.height_levels[action_array]
        executed_heights = intended_heights.astype(np.float32).copy()
        prev_improvement = self.improvement_height.astype(np.float32).copy()

        # Penalise repeated upgrades on consecutive steps
        repeat_mask = self._last_positive_mask & (executed_heights > 0.0)
        repeat_count = int(repeat_mask.sum())
        repeat_penalty_total = 0.0
        if repeat_count:
            executed_heights[repeat_mask] = 0.0
            repeat_penalty_total = self.repeat_asset_penalty * repeat_count

        costs = self._compute_costs(executed_heights)
        total_cost = float(costs.sum())

        trimmed_assets: list[int] = []
        if total_cost > self.budget:
            # Drop the most expensive upgrades until affordable
            order = np.argsort(costs)[::-1]
            for idx in order:
                if executed_heights[idx] <= 0.0:
                    continue
                trimmed_assets.append(int(idx))
                total_cost -= float(costs[idx])
                executed_heights[idx] = 0.0
                if total_cost <= self.budget:
                    break
            if trimmed_assets:
                costs = self._compute_costs(executed_heights)
                total_cost = float(costs.sum())

        post_improvement = np.maximum(prev_improvement + executed_heights, 0.0)
        self._last_positive_mask = executed_heights > 0.0
        self._year += self.year_step
        current_year = self._year
        sea_level_step_index = current_year // self.year_step if self.year_step > 0 else 0
        if self._weight_years is not None:
            target_year = self._weight_base_year + current_year
            idx = int(np.searchsorted(self._weight_years, target_year, side="right") - 1)
        else:
            idx = current_year // self.year_step
        self._set_weight_index(idx)
        prev_loss = self._compute_loss(self._prev_metrics)

        rng_state = self.rng.bit_generator.state
        base_metrics = self._compute_metrics(prev_improvement, year=current_year)
        self.rng.bit_generator.state = rng_state
        new_metrics = self._compute_metrics(post_improvement, year=current_year)

        (
            base_gas_loss,
            base_elec_loss,
            base_gas_soc,
            base_elec_soc,
        ) = base_metrics
        (
            new_gas_loss,
            new_elec_loss,
            new_gas_soc,
            new_elec_soc,
        ) = new_metrics

        base_loss = self._compute_loss(base_metrics)
        new_loss = self._compute_loss(new_metrics)
        climate_drift = base_loss - prev_loss
        action_gain = base_loss - new_loss
        reward_delta = prev_loss - new_loss

        self.improvement_height = post_improvement.astype(np.float32)
        self.wall_height = self.improvement_height + self.initial_wall_height
        self.gas_loss_mean = new_gas_loss
        self.elec_loss_mean = new_elec_loss
        self.gas_soc_mean = new_gas_soc
        self.elec_soc_mean = new_elec_soc
        self._econ_impact = self.gas_loss_mean + self.elec_loss_mean
        self._social_impact = self.gas_soc_mean + self.elec_soc_mean
        self._prev_metrics = new_metrics
        self._prev_loss = new_loss

        reward_delta *= self.reward_scale
        if self.budget > 0.0:
            normalized_cost = total_cost / self.budget
        else:
            normalized_cost = total_cost
        cost_penalty = self.cost_weight * normalized_cost
        reward = float(reward_delta - cost_penalty + repeat_penalty_total)

        terminated = bool(self._year >= self.T)
        truncated = False
        info = {
            "econ_impact": self._econ_impact,
            "social_impact": self._social_impact,
            "intended_heights": intended_heights.astype(np.float32),
            "executed_heights": executed_heights.astype(np.float32),
            "costs": costs.astype(np.float32),
            "total_cost": float(total_cost),
            "unused_budget": float(max(self.budget - total_cost, 0.0)),
            "normalized_cost": float(normalized_cost),
            "cost_penalty": float(cost_penalty),
            "prev_loss": float(prev_loss),
            "base_loss": float(base_loss),
            "new_loss": float(new_loss),
            "climate_drift": float(climate_drift),
            "action_gain": float(action_gain),
        }
        if self._active_climate_scenario is not None:
            info["climate_scenario"] = self._active_climate_scenario
            info["sea_level_offset"] = float(self._sea_level_offset_for_year(current_year))
            if self._sea_level_deltas is not None:
                delta_idx = sea_level_step_index - 1
                if 0 <= delta_idx < self._sea_level_deltas.shape[0]:
                    info["sea_level_delta"] = float(self._sea_level_deltas[delta_idx])
        if repeat_count:
            info["repeat_penalty"] = float(repeat_penalty_total)
            info["repeat_penalty_assets"] = [int(i) for i in np.nonzero(repeat_mask)[0]]
        if trimmed_assets:
            info["trimmed_assets"] = trimmed_assets
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        print(
            f"Year {self._year}/{self.T} | EconImpact={self._econ_impact:.3f} | SocialImpact={self._social_impact:.3f}"
        )

    def close(self):
        pass
