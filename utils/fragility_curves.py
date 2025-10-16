import numpy as np
from numpy import errstate

# the approach for fragility curves is based on Asaridis, P., Molinari, D., Di Maio, F., Ballio, F., & Zio, E. (2025).
# A probabilistic modeling and simulation framework for power grid flood risk assessment. 
# International Journal of Disaster Risk Reduction, 120, 105353. https://doi.org/10.1016/j.ijdrr.2025.105353

# the sigmas are based on the complexity and uncertainty of the assset

_INV_SQRT2 = np.float32(1.0 / np.sqrt(2.0))


def _erf(x):
    """Error function approximation (Abramowitz & Stegun 7.1.26)."""
    x = np.asarray(x, dtype=np.float32)
    sign = np.sign(x)
    abs_x = np.abs(x)
    t = np.float32(1.0) / (np.float32(1.0) + np.float32(0.3275911) * abs_x)
    a1 = np.float32(0.254829592)
    a2 = np.float32(-0.284496736)
    a3 = np.float32(1.421413741)
    a4 = np.float32(-1.453152027)
    a5 = np.float32(1.061405429)
    poly = (((a5 * t + a4) * t + a3) * t + a2) * t + a1
    exp_term = np.exp(-abs_x * abs_x)
    y = np.float32(1.0) - poly * exp_term
    return sign * y


def _lognorm_cdf(depth, sigma: float, scale: float):
    """Compute log-normal CDF using numpy primitives (no SciPy dependency)."""
    depth_arr = np.asarray(depth, dtype=np.float32)
    result = np.zeros_like(depth_arr, dtype=np.float32)
    positive_mask = depth_arr > 0.0
    if not np.any(positive_mask):
        return result

    sigma_val = np.float32(max(sigma, 1e-6))
    scale_val = np.float32(max(scale, 1e-6))

    with errstate(divide='ignore'):
        log_depth = np.log(depth_arr[positive_mask])
    log_scale = np.log(scale_val)
    z = (log_depth - log_scale) * (_INV_SQRT2 / sigma_val)
    result[positive_mask] = 0.5 * (1.0 + _erf(z))
    return np.clip(result, 0.0, 1.0)


# lognormal fragility curve for PV systems
def fragility_PV(depth, improvement = 0):
    """
    Calculate the probability of failure of a PV system based on the depth of the water.
    """
    scale = 1.0 + improvement
    return _lognorm_cdf(depth, sigma=0.05, scale=scale)

# dima paper
def fragility_substation(depth, improvement = 0):
    """
    Calculate the probability of failure of a substation based on the depth of the water.
    """
    scale = 3.0 + improvement
    return _lognorm_cdf(depth, sigma=0.2, scale=scale)

def fragility_compressor(depth, improvement = 0):
    """
    Calculate the probability of failure of a compressor based on the depth of the water.
    """
    scale = 3.0 + improvement
    return _lognorm_cdf(depth, sigma=0.2, scale=scale)

def fragility_thermal_unit(depth, improvement = 0):
    """
    Calculate the probability of failure of a thermal unit based on the depth of the water.
    """
    scale = 5.0 + improvement
    return _lognorm_cdf(depth, sigma=0.3, scale=scale)

def fragility_LNG_terminal(depth, improvement = 0):
    """
    Calculate the probability of failure of a thermal unit based on the depth of the water.
    """
    scale = 8.0 + improvement
    return _lognorm_cdf(depth, sigma=0.3, scale=scale)
