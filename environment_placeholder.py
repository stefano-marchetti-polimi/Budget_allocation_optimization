import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional


class TrialEnv(gym.Env):
    """
    Custom environment for sequential (yearly) investment over a network of nodes.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the system network.
    years : int, default=50
        Number of decision epochs.
    weight_econ : float, default=0.5
        Weight of the (economic) utility in the reward.
    weight_soc : float, default=0.5
        Weight of the (social) utility in the reward.
    util_econ : Callable[[float], float], default=lambda x: -x
        Utility function applied to expected economic impact (lower is better by default).
    util_soc : Callable[[float], float], default=lambda x: -x
        Utility function applied to expected social impact (lower is better by default).
    alpha : float, default=0.15
        Effectiveness coefficient mapping expenses to improvements (diminishing returns).
    decay : float, default=0.02
        Yearly degradation rate of node condition.
    action_max : float, default=1.0
        Upper bound for per-node yearly expense in the action space.

    Notes
    -----
    - State keeps a per-node "condition" in [0, 1], where 1 is best.
    - Expected impacts are (for now) simple linear functions of (1 - condition).
    - Reward is weight_econ * util_econ(econ_impact) + weight_soc * util_soc(social_impact).
    - The episode lasts `years` steps. Actions are vectors of expenses, one per node.
    """

    def __init__(
        self,
        num_nodes: int,
        years: int = 50,
        weight_econ: float = 0.5,
        weight_soc: float = 0.5,
        util_econ: Callable[[float], float] = lambda x: -x,
        util_soc: Callable[[float], float] = lambda x: -x,
        alpha: float = 0.15,
        decay: float = 0.02,
        budget: float = 100000.0,
        year_step: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert num_nodes >= 1, "num_nodes must be >= 1"
        assert years >= 1, "years must be >= 1"
        self.N = int(num_nodes)
        self.T = int(years)
        self.budget = float(budget)
        self.year_step = int(year_step)

        # Weights and utilities
        self.w_e = float(weight_econ)
        self.w_s = float(weight_soc)
        self.util_econ = util_econ
        self.util_soc = util_soc

        # Per-node baseline impact coefficients (>=0)
        self.base_econ = (
            np.ones(self.N, dtype=float)
        )
        self.base_social = (
            np.ones(self.N, dtype=float)
        )

        # Dynamics coefficients
        self.alpha = float(alpha)
        self.decay = float(decay)

        # Action space: per-node yearly expenses as a fraction of the budget (0..1 per node). No sum constraint enforced.
        self.action_max = float(1)
        self.action_space = spaces.Box(
            low=0.0, high=self.action_max, shape=(self.N,), dtype=np.float32
        )

        # Observation: [conditions (N), current_year (1), econ_impact (1), social_impact (1)]
        # Conditions are in [0, 1], impacts are >= 0, year in [0, T]
        high_obs = np.array([1.0] * self.N + [float(self.T)] + [np.finfo(np.float32).max] * 2, dtype=np.float32)
        low_obs = np.zeros(self.N + 3, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # RNG seed / state
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Internal state
        self._conditions = np.zeros(self.N, dtype=np.float32)
        self._year = 0
        self._econ_impact = 0.0
        self._social_impact = 0.0

    # ------------------------
    # Helper computations
    # ------------------------
    def _compute_impacts(self) -> tuple[float, float]:
        # Impacts decrease with node condition; linear for now
        # econ = sum_i base_econ[i] * (1 - condition_i)
        lack = 1.0 - self._conditions.astype(float)
        econ = float(np.dot(self.base_econ, lack))
        soc = float(np.dot(self.base_social, lack))
        return econ, soc

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [self._conditions.astype(np.float32), np.array([self._year, self._econ_impact, self._social_impact], dtype=np.float32)]
        )

    # ------------------------
    # Gym API
    # ------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Start from new
        self._conditions = np.ones(self.N, dtype=np.float32)
        self._year = 0
        self._econ_impact, self._social_impact = self._compute_impacts()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        # Validate and clip to action space
        action = np.asarray(action, dtype=np.float32).reshape(self.N)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Improvement with diminishing returns: alpha * a / (1 + a)
        improvement = self.alpha * (action / (1.0 + action))

        # Degradation and improvement update: c' = clip(c*(1-decay) + improvement, 0, 1)
        self._conditions = np.clip(
            self._conditions * (1.0 - self.decay) + improvement.astype(np.float32), 0.0, 1.0
        )

        # Advance time
        self._year += self.year_step
        self._econ_impact, self._social_impact = self._compute_impacts()

        # Reward: weighted sum of utilities of impacts
        reward = (
            self.w_e * float(self.util_econ(self._econ_impact))
            + self.w_s * float(self.util_soc(self._social_impact))
        )

        terminated = bool(self._year >= self.T)
        truncated = False
        info = {
            "econ_impact": self._econ_impact,
            "social_impact": self._social_impact,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        print(
            f"Year {self._year}/{self.T} | EconImpact={self._econ_impact:.3f} | SocialImpact={self._social_impact:.3f} | MeanCondition={self._conditions.mean():.3f}"
        )

    def close(self):
        pass