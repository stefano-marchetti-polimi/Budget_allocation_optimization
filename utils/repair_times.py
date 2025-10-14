import numpy as np
from typing import Optional, Union

# normal distribution for each depth interval with 10% of the mean as the standard deviation (assumption)
# compressor repair times much loger than substation repair times???? 

RNGType = Union[np.random.Generator, np.random.RandomState]

def _sample_repair_time(mean: float, rng: Optional[RNGType]) -> float:
    """Draw a repair time using a Normal(mean, 0.1*mean) and cap at mean+3σ (floor at 0).
    This ensures a maximum repair time per asset equal to mean + 3 * (0.1*mean) = 1.3*mean.
    """
    scale = 0.1 * mean
    if rng is None:
        sample = float(np.random.normal(mean, scale))
    else:
        sample = float(rng.normal(loc=mean, scale=scale))

    # Cap to at most mean + 3*scale and prevent negative times
    cap_upper = mean + 3.0 * scale
    if sample < 0.0:
        return 0.0
    return min(sample, cap_upper)


def compressor_repair_time(depth, rng: Optional[RNGType] = None):
    """
    Calculate the time to repair (in hours) of a compressor based on the depth of the water
    """
    if depth <= 0:
        return 0.0
    if depth < 0.5:
        mean = 12.3
        return _sample_repair_time(mean, rng)
    elif depth < 0.6:
        mean = 14.1
        return _sample_repair_time(mean, rng)
    elif depth < 0.8:
        mean = 17.7
        return _sample_repair_time(mean, rng)
    elif depth < 0.9:
        mean = 19.4
        return _sample_repair_time(mean, rng)
    elif depth < 1.1:
        mean = 23.1
        return _sample_repair_time(mean, rng)
    else:
        mean = 25.0
        return _sample_repair_time(mean, rng)

def substation_repair_time(depth, rng: Optional[RNGType] = None):
    """
    Calculate the time to repair (in hours) of a substationbased on the depth of the water
    from "M. Movahednia, A. Kargarian, “Flood-aware Optimal Power Flow for Proactive Day-ahead Transmission Substation Hardening,” 
    CoRR, vol. abs/2201.03162, Jan. 2022"
    """
    if depth <= 0:
        return 0.0
    if depth < 0.5:
        mean = 12.3
        return _sample_repair_time(mean, rng)
    elif depth < 0.6:
        mean = 14.1
        return _sample_repair_time(mean, rng)
    elif depth < 0.8:
        mean = 17.7
        return _sample_repair_time(mean, rng)
    elif depth < 0.9:
        mean = 19.4
        return _sample_repair_time(mean, rng)
    elif depth < 1.1:
        mean = 23.1
        return _sample_repair_time(mean, rng)
    else:
        mean = 25.0
        return _sample_repair_time(mean, rng)

def thermal_unit_repair_time(depth, rng: Optional[RNGType] = None):
    """
    Calculate the time to repair (in hours) of a thermal unit (assumed larger than substation)
    """
    if depth <= 0:
        return 0.0
    if depth < 0.5:
        mean = 12.3*1.5
        return _sample_repair_time(mean, rng)
    elif depth < 0.6:
        mean = 14.1*1.5
        return _sample_repair_time(mean, rng)
    elif depth < 0.8:
        mean = 17.7*1.5
        return _sample_repair_time(mean, rng)
    elif depth < 0.9:
        mean = 19.4*1.5
        return _sample_repair_time(mean, rng)
    elif depth < 1.1:
        mean = 23.1*1.5
        return _sample_repair_time(mean, rng)
    else:
        mean = 25.0*1.5
        return _sample_repair_time(mean, rng)

def LNG_repair_time(depth, rng: Optional[RNGType] = None):
    """
    Calculate the time to repair (in hours) of the LNG terminal (assumed larger than substation)
    """
    if depth <= 0:
        return 0.0
    if depth < 0.5:
        mean = 12.3*1.5
        return _sample_repair_time(mean, rng)
    elif depth < 0.6:
        mean = 14.1*1.5
        return _sample_repair_time(mean, rng)
    elif depth < 0.8:
        mean = 17.7*1.5
        return _sample_repair_time(mean, rng)
    elif depth < 0.9:
        mean = 19.4*1.5
        return _sample_repair_time(mean, rng)
    elif depth < 1.1:
        mean = 23.1*1.5
        return _sample_repair_time(mean, rng)
    else:
        mean = 25.0*1.5
        return _sample_repair_time(mean, rng)

def pv_repair_time(depth, rng: Optional[RNGType] = None):
    """
    Calculate the time to repair (in hours) of a photovoltaic unit (assumed smaller than substation)
    """
    if depth <= 0:
        return 0.0
    if depth < 0.5:
        mean = 12.3*0.8
        return _sample_repair_time(mean, rng)
    elif depth < 0.6:
        mean = 14.1*0.8
        return _sample_repair_time(mean, rng)
    elif depth < 0.8:
        mean = 17.7*0.8
        return _sample_repair_time(mean, rng)
    elif depth < 0.9:
        mean = 19.4*0.8
        return _sample_repair_time(mean, rng)
    elif depth < 1.1:
        mean = 23.1*0.8
        return _sample_repair_time(mean, rng)
    else:
        mean = 25.0*0.8
        return _sample_repair_time(mean, rng)
