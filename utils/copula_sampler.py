# pip install copulas numpy pandas

import numpy as np
import pandas as pd

# ---- Direct Gumbel copula sampler (θ ≥ 1) via conditional method ------------
def _sample_gumbel_uv(n: int, theta: float, rng: np.random.Generator):
    """Sample (U1, U2) from a Gumbel copula with parameter theta >= 1.
    Uses the conditional distribution method to avoid numerical root-finding.
    """
    theta = float(theta)
    if theta < 1.0:
        theta = 1.0  # independence limit
    eps = 1e-12
    u1 = rng.uniform(eps, 1.0 - eps, size=n)
    v = rng.uniform(eps, 1.0 - eps, size=n)
    # a = (-ln u1)^θ
    a = (-np.log(u1)) ** theta
    # target = (-ln(u1*v))^θ - a  ≥ 0
    target = (-np.log(u1 * v)) ** theta - a
    target = np.maximum(target, 0.0)
    b = target ** (1.0 / theta)
    u2 = np.exp(-b)
    # Clip away from exact 0/1 for downstream transforms
    u1 = np.clip(u1, eps, 1.0 - eps)
    u2 = np.clip(u2, eps, 1.0 - eps)
    return u1, u2

# ---- Single parameter -------------------------------------------------------
theta = 3.816289  # Gumbel copula parameter (θ >= 1)

# ---- Your empirical marginals ----------------------------------------------
height_emp = np.array([
    1.732, 2.428, 2.191, 2.427, 1.437, 1.434, 1.431, 1.430,
    2.146, 1.992, 1.862, 3.716, 1.550, 1.597, 1.406, 1.409,
    1.461, 1.925, 1.524, 2.319, 1.411, 1.529, 1.486, 3.203
], dtype=float)

duration_emp = np.array([
    11, 37, 8, 17, 2, 1, 2, 2, 13, 13, 8, 104, 2, 5, 1, 1,
    8, 17, 6, 8, 1, 5, 4, 15
], dtype=float)

# ---- Generalized Pareto quantile (inverse CDF) ------------------------------
def gpd_ppf(u, k, sigma):
    """Vectorized GPD quantile for u in (0,1), with threshold (theta) = 0.
    If k == 0, falls back to the exponential limit case.
    """
    u = np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12)
    k = float(k)
    sigma = float(sigma)
    if k == 0.0:
        return -sigma * np.log(1.0 - u)
    return sigma / k * ((1.0 - u) ** (-k) - 1.0)

# ---- Many draws -------------------------------------------------------------
def sample_flood(n, theta, seed=None):
    """Sample n events; returns a DataFrame with U1,U2,height,duration."""
    # RNG
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    # Draw from Gumbel copula directly (no root-finding or tau checks)
    U1, U2 = _sample_gumbel_uv(n, theta, rng)
    k_depth = 0.8019
    k_duration = 0.3929
    sigma_depth = 0.1959
    sigma_duration = 7.2699

    # Map U1, U2 through GPD quantiles (threshold/location theta=0)
    heights_cont = gpd_ppf(U1, k_depth, sigma_depth)
    durations_cont = gpd_ppf(U2, k_duration, sigma_duration)

    # Duration in integer hours (at least 1 hour)
    durations = np.maximum(1, np.rint(durations_cont).astype(int))
    heights = heights_cont.astype(float)

    return pd.DataFrame({
        "U1": U1,
        "U2": U2,
        "height": heights,
        "duration": durations
    })