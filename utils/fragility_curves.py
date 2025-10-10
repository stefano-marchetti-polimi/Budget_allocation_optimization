import numpy as np
from scipy import stats

# the approach for fragility curves is based on Asaridis, P., Molinari, D., Di Maio, F., Ballio, F., & Zio, E. (2025). 
# A probabilistic modeling and simulation framework for power grid flood risk assessment. 
# International Journal of Disaster Risk Reduction, 120, 105353. https://doi.org/10.1016/j.ijdrr.2025.105353

# the sigmas are based on the complexity and uncertainty of the assset

# lognormal fragility curve for PV systems
def fragility_PV(depth):
    """
    Calculate the probability of failure of a PV system based on the depth of the water.
    """
    return stats.lognorm.cdf(depth, s=0.05, scale=np.exp(0))

# dima paper
def fragility_substation(depth):
    """
    Calculate the probability of failure of a substation based on the depth of the water.
    """
    return stats.lognorm.cdf(depth, s=0.2, scale=np.exp(1.09))

def fragility_compressor(depth):
    """
    Calculate the probability of failure of a compressor based on the depth of the water.
    """
    return stats.lognorm.cdf(depth, s=0.2, scale=np.exp(1.09))

def fragility_thermal_unit(depth):
    """
    Calculate the probability of failure of a thermal unit based on the depth of the water.
    """
    return stats.lognorm.cdf(depth, s=0.3, scale=np.exp(1.609))

def fragility_LNG_terminal(depth):
    """
    Calculate the probability of failure of a thermal unit based on the depth of the water.
    """
    return stats.lognorm.cdf(depth, s=0.35, scale=np.exp(2.08))