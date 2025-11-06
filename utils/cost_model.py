"""
wall_cost.py
------------
Importable helper to compute wall construction cost.

Formula:
    cost_MEUR = unit_cost_MEUR_per_km_m * length_km * height_m

Usage:
    from wall_cost import calculate_wall_cost, DEFAULT_ASSET_COSTS, list_asset_types

    cost = calculate_wall_cost("concrete retaining wall", 2.5, 3.0)
    print(cost)  # in M€

    # Or with your own cost map:
    my_costs = {"seawall A": 1.8, "seawall B": 1.2}
    cost = calculate_wall_cost("seawall A", 1.0, 4.0, asset_costs=my_costs)
"""

from typing import Dict, Iterable

# Cost of flood defense systems (M€ per (km*m))
# Aerts, J. C., Botzen, W. W., de Moel, H., & Bowman, M. (2013). Cost estimates for flood resilience and protection strategies in New York City. Annals of the New York Academy of Sciences, 1294(1), 1-104.

# from Table 3.1 (2008)
#  Label, Type of floodwall, cost in (M€ per (km*m))
#  "W76", 7-FootHighL-Wallwith6-FootWideMonoliths, 3.7
#  "W88", 8-FootHighT-Wallwith8-FootWideMonoliths, 3.43
#  "W108", 10-FootHighT-Wallwith8-FootWideMonoliths, 3.31
#  "W1211", 12-FootHighT-Wallwith11-FootWideMonoliths, 3.43
#  "W1411", 14-FootHighL-Wallwith11-FootWideMonoliths, 3.63
#  "W1611", 16-FootHighL-Wallwith11-FootWideMonoliths, 3.53
#  "W1813", 18-FootHighL-Wallwith13-FootWideMonoliths, 3.72
#  "W2014", 20-FootHighT-Wallwith14-FootWideMonoliths, 3.99
#  "W2216", 22-FootHighT-Wallwith16-FootWideMonoliths, 3.96
#  "W2417", 24-FootHighT-Wallwith17-FootWideMonoliths, 4.1
#  "W266", 26-FootHighL-Wallwith6-FootWideMonoliths, 4.54
#  "W286", 28-FootHighL-Wallwith6-FootWideMonoliths, 4.47
#  "W306", 30-FootHighL-Wallwith6-FootWideMonoliths, 4.52

# from Table 3.2
#  Label, Type of floodwall, cost in (M€ per (km*m)) converted accordint to €/$=1.25 (2012)
#  "WT24", Flood T-wall (24ft), 5.625
#  "WL30", Flood L-wall (30ft), 5.875
#  "D", Dike, 10
#  "HD", Hurricane Dike, 11.25
#  "RBH", Retrofit bulkhead, 10
#  "ML", Mixed levee, 10
#  "FPLD", Flood protection low density, 5
#  "HL", Hidden levee + nourishment, 10


DEFAULT_ASSET_COSTS: Dict[str, float] = {
    "W76": 3.7,
    "W88": 3.43,
    "W108": 3.31,
    "W1211": 3.43,
    "W1411": 3.63,
    "W1611": 3.53,
    "W1813": 3.72,
    "W2014": 3.99,
    "W2216": 3.96,
    "W2417": 4.1,
    "W266": 4.54,
    "W286": 4.47,
    "W306": 4.52,
    "WT24": 5.625,
    "WL30": 5.875,
    "D": 10,
    "HD": 11.25,
    "RBH": 10,
    "ML": 10,
    "FPLD": 5,
    "HL": 10,
}

def list_asset_types(asset_costs: Dict[str, float] | None = None) -> Iterable[str]:
    """
    Return available asset types.
    """
    return (asset_costs or DEFAULT_ASSET_COSTS).keys()

def calculate_wall_cost(
    asset_type: str,
    length_km: float,
    height_m: float,
    asset_costs: Dict[str, float] | None = None,
) -> float:
    """
    Compute total construction cost in million euros (M€).

    Parameters
    ----------
    asset_type : str
        Must exist in the asset_costs mapping.
    length_km : float
        Wall length in kilometers (>= 0).
    height_m : float
        Wall height in meters (>= 0).
    asset_costs : dict[str, float], optional
        Mapping of asset type -> unit cost in M€ per (km*m).
        If not provided, DEFAULT_ASSET_COSTS is used.

    Returns
    -------
    float
        Total cost in M€.

    Raises
    ------
    KeyError
        If asset_type is not present in asset_costs.
    ValueError
        If length_km or height_m is negative.
    """
    costs = asset_costs or DEFAULT_ASSET_COSTS
    if asset_type not in costs:
        raise KeyError(f"Unknown asset type: {asset_type!r}. Available: {', '.join(costs)}")
    if length_km < 0 or height_m < 0:
        raise ValueError("length_km and height_m must be non-negative.")
    unit = costs[asset_type]
    return unit * length_km * height_m
