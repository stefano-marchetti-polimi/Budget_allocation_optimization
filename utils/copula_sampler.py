# pip install copulas numpy pandas

import numpy as np
import pandas as pd
from copulas.bivariate import Gumbel

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
    # Gumbel copula requires theta >= 1.  Treat theta <= 1 as independence (theta=1).
    cop = Gumbel()
    theta = float(theta)
    theta_eff = 1.0 if theta <= 1.0 else theta
    cop.theta = theta_eff
    # Ensure Kendall's tau is defined for sampling (the library checks this)
    cop.tau = 0.0 if theta_eff == 1.0 else 1.0 - 1.0 / theta_eff
    # Some versions expect stored params / fitted flag
    try:
        cop._params = {"theta": theta_eff}
        cop.is_fitted = True
    except Exception:
        pass
    if seed is not None:
        try:
            cop.random_state = np.random.RandomState(seed)
        except AttributeError:
            np.random.seed(seed)
    sample_uv = cop.sample(n)
    U = sample_uv.to_numpy() if hasattr(sample_uv, "to_numpy") else np.asarray(sample_uv)
    U1, U2 = U[:, 0], U[:, 1]
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