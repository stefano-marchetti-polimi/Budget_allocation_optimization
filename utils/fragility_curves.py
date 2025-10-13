import imp
import numpy as np
from scipy import stats

# the approach for fragility curves is based on Asaridis, P., Molinari, D., Di Maio, F., Ballio, F., & Zio, E. (2025). 
# A probabilistic modeling and simulation framework for power grid flood risk assessment. 
# International Journal of Disaster Risk Reduction, 120, 105353. https://doi.org/10.1016/j.ijdrr.2025.105353

# the sigmas are based on the complexity and uncertainty of the assset

# lognormal fragility curve for PV systems
def fragility_PV(depth, improvement = 0):
    """
    Calculate the probability of failure of a PV system based on the depth of the water.
    """
    return stats.lognorm.cdf(depth, s=0.05, scale=1+improvement)

# dima paper
def fragility_substation(depth, improvement = 0):
    """
    Calculate the probability of failure of a substation based on the depth of the water.
    """
    return stats.lognorm.cdf(depth, s=0.2, scale=3+improvement)

def fragility_compressor(depth, improvement = 0):
    """
    Calculate the probability of failure of a compressor based on the depth of the water.
    """
    return stats.lognorm.cdf(depth, s=0.2, scale=3+improvement)

def fragility_thermal_unit(depth, improvement = 0):
    """
    Calculate the probability of failure of a thermal unit based on the depth of the water.
    """
    return stats.lognorm.cdf(depth, s=0.3, scale=5+improvement)

def fragility_LNG_terminal(depth, improvement = 0):
    """
    Calculate the probability of failure of a thermal unit based on the depth of the water.
    """
    return stats.lognorm.cdf(depth, s=0.3, scale=8+improvement)