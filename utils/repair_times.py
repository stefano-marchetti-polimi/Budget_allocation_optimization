import numpy as np

# normal distribution for each depth interval with 10% of the mean as the standard deviation (assumption)
# compressor repair times much loger than substation repair times???? 

def compressor_repair_time(depth):
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
        return np.random.normal(mean, 0.1 * mean)
    elif 0.5 <= depth < 1:
        mean = 168
        return np.random.normal(mean, 0.1 * mean)
    elif 1 <= depth < 1.5:
        mean = 504
        return np.random.normal(mean, 0.1 * mean)
    elif depth >= 1.5:
        mean = 1008
        return np.random.normal(mean, 0.1 * mean)

def substation_repair_time(depth):
    """
    Calculate the time to repair (in hours) of a substationbased on the depth of the water
    from "M. Movahednia, A. Kargarian, “Flood-aware Optimal Power Flow for Proactive Day-ahead Transmission Substation Hardening,” 
    CoRR, vol. abs/2201.03162, Jan. 2022"
    """
    if depth <= 0:
        return 0.0
    if depth < 0.5:
        mean = 12.3
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 0.6:
        mean = 14.1
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 0.8:
        mean = 17.7
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 0.9:
        mean = 19.4
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 1.1:
        mean = 23.1
        return np.random.normal(mean, 0.1 * mean)
    else:
        mean = 25.0
        return np.random.normal(mean, 0.1 * mean)

def thermal_unit_repair_time(depth):
    """
    Calculate the time to repair (in hours) of a thermal unit (assumed equal to substation)
    """
    if depth <= 0:
        return 0.0
    if depth < 0.5:
        mean = 12.3
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 0.6:
        mean = 14.1
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 0.8:
        mean = 17.7
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 0.9:
        mean = 19.4
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 1.1:
        mean = 23.1
        return np.random.normal(mean, 0.1 * mean)
    else:
        mean = 25.0
        return np.random.normal(mean, 0.1 * mean)

def pv_repair_time(depth):
    """
    Calculate the time to repair (in hours) of a photovoltaic unit (assumed equal to substation)
    """
    if depth <= 0:
        return 0.0
    if depth < 0.5:
        mean = 12.3
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 0.6:
        mean = 14.1
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 0.8:
        mean = 17.7
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 0.9:
        mean = 19.4
        return np.random.normal(mean, 0.1 * mean)
    elif depth < 1.1:
        mean = 23.1
        return np.random.normal(mean, 0.1 * mean)
    else:
        mean = 25.0
        return np.random.normal(mean, 0.1 * mean)