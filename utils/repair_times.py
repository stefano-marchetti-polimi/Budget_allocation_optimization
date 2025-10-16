import numpy as np
from typing import Optional, Union

# Normal distribution for each depth interval with 10% of the mean as the standard deviation (assumption)
# compressor repair times much longer than substation repair times?

RNGType = Union[np.random.Generator, np.random.RandomState]

_DEPTH_BREAKS = np.array([0.5, 0.6, 0.8, 0.9, 1.1], dtype=np.float32)
_SUBSTATION_MEANS = np.array([12.3, 14.1, 17.7, 19.4, 23.1, 25.0], dtype=np.float32)
_COMPRESSOR_MEANS = _SUBSTATION_MEANS.copy()
_THERMAL_MEANS = _SUBSTATION_MEANS * 1.5
_LNG_MEANS = _THERMAL_MEANS
_PV_MEANS = _SUBSTATION_MEANS * 0.8


def _as_array(data):
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def _standard_normal(rng: Optional[RNGType], size: int) -> np.ndarray:
    if rng is None:
        return np.random.standard_normal(size)
    if isinstance(rng, np.random.Generator):
        return rng.standard_normal(size)
    return rng.normal(loc=0.0, scale=1.0, size=size)


def _sample_array(depths: np.ndarray, mean_table: np.ndarray, rng: Optional[RNGType], normals: Optional[np.ndarray]) -> np.ndarray:
    result = np.zeros_like(depths, dtype=np.float32)
    positive_mask = depths > 0.0
    if not np.any(positive_mask):
        return result

    active_depths = depths[positive_mask]
    idx = np.digitize(active_depths, _DEPTH_BREAKS, right=False)
    means = mean_table[idx]
    scales = 0.1 * means

    if normals is None:
        draws = _standard_normal(rng, means.shape[0])
    else:
        draws = np.asarray(normals, dtype=np.float32)
        if draws.shape != depths.shape:
            raise ValueError("normals must match depths shape")
        draws = draws[positive_mask]

    samples = means + scales * draws
    cap = means * 1.3  # mean + 3 * (0.1 * mean)
    samples = np.clip(samples, 0.0, cap)
    result[positive_mask] = samples.astype(np.float32)
    return result


def _finalize(depth_input, samples: np.ndarray):
    if np.isscalar(depth_input):
        return float(samples[0])
    return samples.reshape(np.asarray(depth_input, dtype=np.float32).shape)


def compressor_repair_time(depth, rng: Optional[RNGType] = None, normals: Optional[np.ndarray] = None):
    """
    Calculate the time to repair (in hours) of a compressor based on the depth of the water.
    Accepts scalars or numpy arrays; providing `normals` reuses pre-generated standard normal draws.
    """
    depth_arr = _as_array(depth)
    samples = _sample_array(depth_arr, _COMPRESSOR_MEANS, rng, normals)
    return _finalize(depth, samples)


def substation_repair_time(depth, rng: Optional[RNGType] = None, normals: Optional[np.ndarray] = None):
    """
    Calculate the time to repair (in hours) of a substation based on the depth of the water.
    Source: M. Movahednia, A. Kargarian, “Flood-aware Optimal Power Flow for Proactive Day-ahead
    Transmission Substation Hardening,” CoRR, vol. abs/2201.03162, Jan. 2022
    """
    depth_arr = _as_array(depth)
    samples = _sample_array(depth_arr, _SUBSTATION_MEANS, rng, normals)
    return _finalize(depth, samples)


def thermal_unit_repair_time(depth, rng: Optional[RNGType] = None, normals: Optional[np.ndarray] = None):
    """
    Calculate the time to repair (in hours) of a thermal unit (assumed larger than a substation).
    """
    depth_arr = _as_array(depth)
    samples = _sample_array(depth_arr, _THERMAL_MEANS, rng, normals)
    return _finalize(depth, samples)


def LNG_repair_time(depth, rng: Optional[RNGType] = None, normals: Optional[np.ndarray] = None):
    """
    Calculate the time to repair (in hours) of the LNG terminal (assumed larger than a substation).
    """
    depth_arr = _as_array(depth)
    samples = _sample_array(depth_arr, _LNG_MEANS, rng, normals)
    return _finalize(depth, samples)


def pv_repair_time(depth, rng: Optional[RNGType] = None, normals: Optional[np.ndarray] = None):
    """
    Calculate the time to repair (in hours) of a photovoltaic unit (assumed smaller than a substation).
    """
    depth_arr = _as_array(depth)
    samples = _sample_array(depth_arr, _PV_MEANS, rng, normals)
    return _finalize(depth, samples)
