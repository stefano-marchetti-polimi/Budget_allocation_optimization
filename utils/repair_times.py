import numpy as np
from typing import Optional, Union

# normal distribution for each depth interval with 10% of the mean as the standard deviation (assumption)
# compressor repair times much loger than substation repair times???? 

RNGType = Union[np.random.Generator, np.random.RandomState]


def _sample_repair_time(mean: float, rng: Optional[RNGType]) -> float:
    """Draw a repair time using the provided RNG (falls back to global np.random)."""
    scale = 0.1 * mean
    if rng is None:
        return float(np.random.normal(mean, scale))
    return float(rng.normal(loc=mean, scale=scale))


def compressor_repair_time(depth, rng: Optional[RNGType] = None):
    """
    Calculate the time to repair (in hours) of a compressor based on the depth of the water
    from "Fioravanti, A., De Simone, G., Carpignano, A., Ruzzone, A., Mortarino, G. & Piccini, M. (2020). 
    Compressor Station Facility Failure Modes: Causes, Taxonomy and Effects (EUR 30265 EN). 
    Luxembourg: Publications Office of the European Union. doi:10.2760/67609"
    """
    if depth == 0:
        return 0.0  
    elif 0 < depth < 0.5:
        mean = 48
        return _sample_repair_time(mean, rng)
    elif 0.5 <= depth < 1:
        mean = 168
        return _sample_repair_time(mean, rng)
    elif 1 <= depth < 1.5:
        mean = 504
        return _sample_repair_time(mean, rng)
    elif depth >= 1.5:
        mean = 1008
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
