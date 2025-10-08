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
        return 48
    elif 0.5 <= depth < 1:
        return 168  
    elif 1 <= depth < 1.5:
        return 504
    elif depth >= 1.5:
        return 1008 

def substation_repair_time(depth):
    return 1

def pv_repair_time(depth):
    return 1

def thermal_unit_repair_time(depth):
    return 1