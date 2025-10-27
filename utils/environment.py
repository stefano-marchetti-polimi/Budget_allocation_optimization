import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from utils.fragility_curves import (
    fragility_PV,
    fragility_substation,
    fragility_compressor,
    fragility_thermal_unit,
    fragility_LNG_terminal,
)
from utils.repair_times import (
    compressor_repair_time,
    substation_repair_time,
    thermal_unit_repair_time,
    pv_repair_time,
    LNG_repair_time,
)
from utils.copula_sampler import sample_flood

EXCEL_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
EXCEL_NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
ASSET_COORD_CRS = "EPSG:32615"

FRAGILITY_FUNCTIONS = {
    "renewable": fragility_PV,
    "thermal": fragility_thermal_unit,
    "compressor": fragility_compressor,
    "substation": fragility_substation,
    "lng": fragility_LNG_terminal,
}

REPAIR_FUNCTIONS = {
    "renewable": pv_repair_time,
    "thermal": thermal_unit_repair_time,
    "compressor": compressor_repair_time,
    "substation": substation_repair_time,
    "lng": LNG_repair_time,
}


@dataclass(frozen=True)
class ComponentConfig:
    name: str
    category: str
    hazard_key: str
    area: float
    dependencies: Tuple[str, ...] = ()
    generator_sources: Tuple[str, ...] = ()
    capacity: float = 0.0
    population: float = 0.0
    coordinate: Optional[Tuple[float, float]] = None
    upgradable: bool = True
    can_fail: bool = True
    industrial_load: float = 0.0


@dataclass(frozen=True)
class NetworkConfig:
    components: Tuple[ComponentConfig, ...]
    compressor_supply: Dict[str, float]
    compressor_people: Dict[str, float]
    substation_supply: Dict[str, float]
    substation_people: Dict[str, float]
    compressor_industrial: Dict[str, float]
    substation_industrial: Dict[str, float]


def _format_component_name(raw: str) -> str:
    cleaned = " ".join(str(raw).strip().split())
    for token in ["'", ".", ","]:
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace("-", " ").replace("/", " ")
    parts = [part for part in cleaned.split() if part]
    return "_".join(parts)


def _parse_cell_value(cell, shared_strings: Dict[int, str]):
    value_tag = cell.find(f"{{{EXCEL_NS_MAIN}}}v")
    if value_tag is None:
        return None
    raw = value_tag.text
    if cell.attrib.get("t") == "s":
        try:
            return shared_strings.get(int(raw), "")
        except (TypeError, ValueError):
            return ""
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return raw
    if val.is_integer():
        return int(val)
    return val


def _read_excel_sheet(path: Path, sheet_name: str) -> List[Dict[str, object]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    with zipfile.ZipFile(path) as z:
        shared_strings: Dict[int, str] = {}
        shared_path = "xl/sharedStrings.xml"
        if shared_path in z.namelist():
            root = ET.fromstring(z.read(shared_path))
            strings: List[str] = []
            for si in root.findall(f"{{{EXCEL_NS_MAIN}}}si"):
                text_fragments = [t.text or "" for t in si.findall(f".//{{{EXCEL_NS_MAIN}}}t")]
                strings.append("".join(text_fragments))
            shared_strings = {idx: text for idx, text in enumerate(strings)}

        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{EXCEL_NS_REL}}}Relationship")
        }

        target_rel = None
        for sheet in workbook.findall(f"{{{EXCEL_NS_MAIN}}}sheets/{{{EXCEL_NS_MAIN}}}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
                target_rel = rel_map.get(rel_id)
                break
        if target_rel is None:
            available = [sheet.attrib.get("name") for sheet in workbook.findall(f"{{{EXCEL_NS_MAIN}}}sheets/{{{EXCEL_NS_MAIN}}}sheet")]
            raise ValueError(f"Sheet '{sheet_name}' not found in {path.name}. Available sheets: {available}")

        sheet_xml = ET.fromstring(z.read(f"xl/{target_rel}"))

    rows: List[Dict[str, object]] = []
    header_map: Dict[str, str] = {}
    for row in sheet_xml.findall(f"{{{EXCEL_NS_MAIN}}}sheetData/{{{EXCEL_NS_MAIN}}}row"):
        row_idx = int(row.attrib.get("r", "0"))
        cells = {}
        for cell in row.findall(f"{{{EXCEL_NS_MAIN}}}c"):
            ref = cell.attrib.get("r", "")
            column = "".join([ch for ch in ref if ch.isalpha()])
            if not column:
                continue
            value = _parse_cell_value(cell, shared_strings)
            if value is None:
                continue
            cells[column] = value
        if not cells:
            continue
        if row_idx == 1:
            header_map = {
                column: str(value).strip()
                for column, value in cells.items()
                if str(value).strip()
            }
            continue
        if not header_map:
            continue
        row_data: Dict[str, object] = {}
        for column, header in header_map.items():
            if column not in cells:
                continue
            row_data[header] = cells[column]
        if row_data:
            rows.append(row_data)
    return rows


def _estimate_area(category: str, capacity: float = 0.0, population: float = 0.0) -> float:
    base_defaults = {
        "lng": 320.0,
        "compressor": 180.0,
        "thermal": 260.0,
        "renewable": 210.0,
        "substation": 190.0,
    }
    scale_defaults = {
        "lng": 4.5,
        "compressor": 3.0,
        "thermal": 5.0,
        "renewable": 4.0,
        "substation": 3.5,
    }
    base = base_defaults.get(category, 200.0)
    scale = scale_defaults.get(category, 3.0)
    proxy = capacity if category not in {"substation", "compressor"} else max(population, capacity)
    return float(base + scale * np.sqrt(max(proxy, 1.0)))


@lru_cache(maxsize=1)
def _build_default_network(data_dir: Path) -> NetworkConfig:
    data_dir = Path(data_dir)
    asset_path = data_dir / "Assets Catalog filtered.xlsx"
    population_path = data_dir / "County Population.xlsx"

    electric_rows = _read_excel_sheet(asset_path, "Electric Network")
    gas_rows = _read_excel_sheet(asset_path, "Gas Network")
    population_rows = _read_excel_sheet(population_path, "Sheet1")

    electric_map: Dict[str, Dict[str, object]] = {}
    for row in electric_rows:
        name = row.get("Name")
        power = row.get("Power (MWe)")
        if not name or power in (None, ""):
            continue
        electric_map[_format_component_name(name)] = row

    gas_map: Dict[str, Dict[str, object]] = {}
    for row in gas_rows:
        name = row.get("Name")
        power = row.get("Power (MWe)")
        if not name or power in (None, ""):
            continue
        gas_map[_format_component_name(name)] = row

    population_map: Dict[str, float] = {}
    for row in population_rows:
        county = row.get("County")
        pop = row.get("Population Size")
        if not county or not isinstance(pop, (int, float)):
            continue
        population_map[_format_component_name(county)] = float(pop)

    industrial_intensity_base = {
        "Harris": 1.0,
        "Jefferson": 0.85,
        "Orange": 0.75,
        "Brazoria": 0.7,
        "Galveston": 0.6,
        "Chambers": 0.65,
        "Liberty": 0.4,
        "Hardin": 0.35,
        "Fort Bend": 0.45,
        "Brazos": 0.35,
    }
    intensity_default = 0.3
    industrial_intensity = {
        _format_component_name(name): float(value)
        for name, value in industrial_intensity_base.items()
    }

    def _row_coordinate(row: Mapping[str, object]) -> Tuple[float, float]:
        east = row.get("Easting (m)")
        north = row.get("Northing (m)")
        if east is None or north is None:
            raise KeyError("Easting/Northing columns missing in asset catalog row.")
        return float(east), float(north)

    def capacity_from_row(row: Dict[str, object]) -> float:
        value = row.get("Power (MWe)")
        if value is None:
            return 0.0
        return float(value)

    components: List[ComponentConfig] = []
    capacity_map: Dict[str, float] = {}
    coordinate_map: Dict[str, Tuple[float, float]] = {}

    def _weighted_centroid(asset_names: Sequence[str]) -> Optional[Tuple[float, float]]:
        coords: List[Tuple[float, float]] = []
        weights: List[float] = []
        for asset in asset_names:
            coord = coordinate_map.get(asset)
            if coord is None:
                continue
            coords.append(coord)
            weights.append(float(capacity_map.get(asset, 1.0)))
        if not coords:
            return None
        total_weight = float(sum(weights))
        if total_weight <= 0.0:
            total_weight = float(len(coords))
            weights = [1.0] * len(coords)
        centroid_x = sum(c[0] * w for c, w in zip(coords, weights)) / total_weight
        centroid_y = sum(c[1] * w for c, w in zip(coords, weights)) / total_weight
        return (centroid_x, centroid_y)

    # LNG terminal (treated as source for gas network)
    lng_row = gas_map.get("Calcasieu_Pass_2")
    if lng_row is None:
        raise KeyError("'Calcasieu Pass 2' entry missing from gas asset catalog")
    lng_capacity = capacity_from_row(lng_row)
    lng_coord = _row_coordinate(lng_row)
    components.append(
        ComponentConfig(
            name="Calcasieu_Pass_LNG",
            category="lng",
            hazard_key="P8",
            area=_estimate_area("lng", lng_capacity),
            capacity=lng_capacity,
            coordinate=lng_coord,
        )
    )
    capacity_map["Calcasieu_Pass_LNG"] = lng_capacity
    coordinate_map["Calcasieu_Pass_LNG"] = lng_coord

    # Renewable and non-gas generation units
    generation_defs = [
        {"name": "Roy_S_Nelson_Coal", "source": ("electric", "Roy_S_Nelson"), "category": "thermal", "hazard": "P7", "dependencies": ()},
        {"name": "Cottonwood_Bayou_Solar", "source": ("electric", "Cottonwood_Bayou"), "category": "renewable", "hazard": "P1"},
        {"name": "Galveston_1_Wind", "source": ("electric", "Galveston_1"), "category": "renewable", "hazard": "P1"},
        {"name": "Galveston_2_Wind", "source": ("electric", "Galveston_2"), "category": "renewable", "hazard": "P1"},
        {"name": "Lake_Charles_Wind", "source": ("electric", "Lake_Charles"), "category": "renewable", "hazard": "P1"},
    ]

    thermal_defs = [
        {"name": "Sabine_CCGT", "source": ("gas", "Sabine"), "category": "thermal", "hazard": "P7", "dependencies": ("Comp_Sabine",)},
        {"name": "Port_Arthur_CCGT", "source": ("gas", "Porth_Arthur"), "category": "thermal", "hazard": "P7", "dependencies": ("Comp_Sabine",)},
        {"name": "Lake_Charles_CCGT", "source": ("gas", "Lake_Charles"), "category": "thermal", "hazard": "P7", "dependencies": ("Comp_Calcasieu",)},
        {"name": "Cottonwood_Gas", "source": ("gas", "Cottonwood"), "category": "thermal", "hazard": "P7", "dependencies": ("Comp_Calcasieu",)},
        {"name": "Cedar_Bayou_CCGT", "source": ("gas", "Cedar_Bayou"), "category": "thermal", "hazard": "P7", "dependencies": ("Comp_CedarBayou",)},
        {"name": "Channelview_CCGT", "source": ("gas", "Channelview"), "category": "thermal", "hazard": "P7", "dependencies": ("Comp_CedarBayou",)},
        {"name": "Bacliff_CCGT", "source": ("gas", "Bacliff"), "category": "thermal", "hazard": "P7", "dependencies": ("Comp_Freeport",)},
        {"name": "Galveston_CCGT", "source": ("gas", "Galveston"), "category": "thermal", "hazard": "P7", "dependencies": ("Comp_Freeport",)},
        {"name": "Freeport_CCGT", "source": ("gas", "Free_Port"), "category": "thermal", "hazard": "P7", "dependencies": ("Comp_Freeport",)},
    ]

    offshore_wind_assets = {"Galveston_1_Wind", "Galveston_2_Wind", "Lake_Charles_Wind"}

    for spec in generation_defs + thermal_defs:
        source_type, source_key = spec["source"]
        data_map = electric_map if source_type == "electric" else gas_map
        row = data_map.get(source_key)
        if row is None:
            raise KeyError(f"Asset '{source_key}' not found in {source_type} catalog")
        capacity = capacity_from_row(row)
        area = _estimate_area(spec["category"], capacity)
        coord = _row_coordinate(row)
        is_offshore_wind = spec["name"] in offshore_wind_assets
        components.append(
            ComponentConfig(
                name=spec["name"],
                category=spec["category"],
                hazard_key=spec.get("hazard", "P7"),
                area=area,
                dependencies=tuple(spec.get("dependencies", ())),
                capacity=capacity,
                coordinate=coord,
                upgradable=not is_offshore_wind,
                can_fail=not is_offshore_wind,
            )
        )
        capacity_map[spec["name"]] = capacity
        coordinate_map[spec["name"]] = coord

    compressor_feeds = {
        "Comp_Sabine": ("Sabine_CCGT", "Port_Arthur_CCGT"),
        "Comp_Calcasieu": ("Lake_Charles_CCGT", "Cottonwood_Gas"),
        "Comp_CedarBayou": ("Cedar_Bayou_CCGT", "Channelview_CCGT"),
        "Comp_Freeport": ("Freeport_CCGT", "Bacliff_CCGT", "Galveston_CCGT"),
    }

    compressor_counties = {
        "Comp_Sabine": ("Jefferson", "Orange"),
        "Comp_Calcasieu": ("Liberty", "Chambers", "Hardin"),
        "Comp_CedarBayou": ("Harris",),
        "Comp_Freeport": ("Brazoria", "Galveston", "Fort_Bend"),
    }

    # Compressor stations (dependencies handled later)
    compressor_defs = {
        "Comp_Sabine": ("P4", ("Calcasieu_Pass_LNG",)),
        "Comp_Calcasieu": ("P5", ("Calcasieu_Pass_LNG",)),
        "Comp_CedarBayou": ("P6", ("Calcasieu_Pass_LNG",)),
        "Comp_Freeport": ("P5", ("Calcasieu_Pass_LNG",)),
    }
    compressor_industrial_load: Dict[str, float] = {}
    for name, (hazard, deps) in compressor_defs.items():
        feed_assets = compressor_feeds.get(name, ())
        supply_capacity = sum(capacity_map.get(asset, 0.0) for asset in feed_assets)
        centroid = _weighted_centroid(feed_assets)
        if centroid is None:
            centroid = coordinate_map.get("Calcasieu_Pass_LNG")
        counties = compressor_counties.get(name, ())
        industrial_load = 0.0
        for county in counties:
            key = _format_component_name(county)
            intensity = industrial_intensity.get(key, intensity_default)
            industrial_load += population_map.get(key, 0.0) * intensity
        compressor_industrial_load[name] = industrial_load
        components.append(
            ComponentConfig(
                name=name,
                category="compressor",
                hazard_key=hazard,
                area=_estimate_area("compressor", supply_capacity),
                dependencies=tuple(deps),
                capacity=supply_capacity,
                coordinate=centroid,
                upgradable=True,
                can_fail=True,
                industrial_load=industrial_load,
            )
        )
        if centroid is not None:
            coordinate_map[name] = centroid
        capacity_map[name] = supply_capacity

    substation_generators: Dict[str, Tuple[str, ...]] = {
        "Sub_Harris_ShipChannel": ("Channelview_CCGT", "Cedar_Bayou_CCGT", "Galveston_1_Wind", "Galveston_2_Wind"),
        "Sub_FortBend_Expansion": ("Freeport_CCGT", "Cottonwood_Bayou_Solar"),
        "Sub_Galveston_Island": ("Galveston_1_Wind", "Galveston_2_Wind", "Galveston_CCGT", "Bacliff_CCGT"),
        "Sub_Brazoria_Gulf": ("Freeport_CCGT", "Cottonwood_Bayou_Solar", "Bacliff_CCGT"),
        "Sub_Jefferson_Orange": ("Roy_S_Nelson_Coal", "Sabine_CCGT", "Port_Arthur_CCGT", "Lake_Charles_Wind", "Lake_Charles_CCGT"),
        "Sub_Liberty_Chambers": ("Cedar_Bayou_CCGT", "Channelview_CCGT", "Cottonwood_Gas"),
    }

    substation_hazards = {
        "Sub_Harris_ShipChannel": "P2",
        "Sub_FortBend_Expansion": "P2",
        "Sub_Galveston_Island": "P3",
        "Sub_Brazoria_Gulf": "P2",
        "Sub_Jefferson_Orange": "P3",
        "Sub_Liberty_Chambers": "P2",
    }

    substation_counties = {
        "Sub_Harris_ShipChannel": ("Harris",),
        "Sub_FortBend_Expansion": ("Fort_Bend",),
        "Sub_Galveston_Island": ("Galveston",),
        "Sub_Brazoria_Gulf": ("Brazoria",),
        "Sub_Jefferson_Orange": ("Jefferson", "Orange"),
        "Sub_Liberty_Chambers": ("Liberty", "Chambers", "Hardin"),
    }

    substation_people: Dict[str, float] = {}
    for name, counties in substation_counties.items():
        total = sum(population_map.get(_format_component_name(county), 0.0) for county in counties)
        substation_people[name] = total

    substation_industrial_load: Dict[str, float] = {}
    for name, generators in substation_generators.items():
        capacity = sum(capacity_map.get(gen, 0.0) for gen in generators)
        population = substation_people.get(name, 0.0)
        centroid = _weighted_centroid(generators)
        if centroid is None:
            centroid = coordinate_map.get("Calcasieu_Pass_LNG")
        counties = substation_counties.get(name, ())
        industrial_load = 0.0
        for county in counties:
            key = _format_component_name(county)
            intensity = industrial_intensity.get(key, intensity_default)
            industrial_load += population_map.get(key, 0.0) * intensity
        substation_industrial_load[name] = industrial_load
        components.append(
            ComponentConfig(
                name=name,
                category="substation",
                hazard_key=substation_hazards.get(name, "P2"),
                area=_estimate_area("substation", capacity, population),
                generator_sources=generators,
                population=population,
                capacity=capacity,
                coordinate=centroid,
                industrial_load=industrial_load,
            )
        )
        if centroid is not None:
            coordinate_map[name] = centroid
        capacity_map[name] = capacity

    compressor_supply: Dict[str, float] = {}
    for name, plants in compressor_feeds.items():
        compressor_supply[name] = sum(capacity_map.get(plant, 0.0) for plant in plants)

    compressor_people: Dict[str, float] = {}
    for name, counties in compressor_counties.items():
        compressor_people[name] = sum(population_map.get(_format_component_name(county), 0.0) for county in counties)

    substation_supply = {name: capacity_map.get(name, 0.0) for name in substation_generators}

    return NetworkConfig(
        components=tuple(components),
        compressor_supply=compressor_supply,
        compressor_people=compressor_people,
        substation_supply=substation_supply,
        substation_people=substation_people,
        compressor_industrial=compressor_industrial_load,
        substation_industrial=substation_industrial_load,
    )

class TrialEnv(gym.Env):
    """
    Custom environment for sequential (yearly) investment over a network of nodes.

    At each decision step the agent chooses wall-height increments for every component from
    the discrete set {0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0} meters. Positive choices incur
    component-specific construction costs; if their sum exceeds the yearly budget, the most
    expensive upgrades are dropped until the action is feasible. Reapplying an upgrade to the
    same component on consecutive steps is cancelled and penalised. Observations can optionally
    be normalised to lie in [0, 1]. The reward is shaped as the reduction in expected losses due
    to the action minus a normalized cost term, so inaction while sea level rises naturally
    incurs a penalty through worsening impacts.
    """

    def __init__(
        self,
        num_nodes: int,
        years: int = 50,
        weights: Sequence[float] = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        weight_years: Optional[Sequence[int]] = None,
        cost_per_meter: float = 10000.0,
        budget: float = 100000.0,
        year_step: int = 1,
        initial_wall_height = None,
        area = None,
        seed: Optional[int] = None,
        mc_samples: int = 500_000,
        csv_path: str = 'outputs/coastal_inundation_samples.csv',
        copula_theta: float = 3.816289,
        max_depth: float = 8.0,
        max_duration: float = 100,
        threshold_depth: float = 0.5,
        rise_rate: float = 0.02,
        sea_level_scenarios: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
        climate_scenario: str = "All",
        normalize_observations: bool = True,
        maximum_repair_time: float = 40.75,
        repeat_asset_penalty: float = -0.1,
        reward_scale: float = 60000.0,
        cost_weight: float = 0.08,
        industrial_weight: float = 0.6,
    ):
        super().__init__()
        assert num_nodes >= 1, "num_nodes must be >= 1"
        assert years >= 1, "years must be >= 1"
        self.N = int(num_nodes)
        self.T = int(years)
        self.budget = float(budget)
        self.year_step = int(year_step)
        weights_array = np.asarray(weights, dtype=np.float32)
        if weights_array.ndim == 1:
            if weights_array.shape[0] != 6:
                raise ValueError(f"weights must contain 6 entries, received shape {weights_array.shape}.")
            weights_array = weights_array.reshape(1, -1)
        elif weights_array.ndim == 2:
            if weights_array.shape[1] != 6:
                raise ValueError(
                    f"Weight schedule must provide 6 entries per step, received shape {weights_array.shape}."
                )
        else:
            raise ValueError("weights must be a 1D sequence of length 6 or a 2D schedule with 6 columns.")

        if np.isnan(weights_array).any():
            raise ValueError("Weight schedule contains NaN values.")

        required_points = ((self.T + self.year_step - 1) // self.year_step) + 1
        if weights_array.shape[0] < required_points:
            raise ValueError(
                f"Weight schedule length {weights_array.shape[0]} insufficient for "
                f"{self.T} years with step {self.year_step}."
            )

        self._weight_schedule = weights_array.astype(np.float32)
        self._current_weight_index = 0
        self.weights = self._weight_schedule[0].copy()

        self._weight_years = None
        self._weight_base_year = 0
        if weight_years is not None:
            weight_years_array = np.asarray(list(weight_years), dtype=np.int64)
            if weight_years_array.shape[0] != self._weight_schedule.shape[0]:
                raise ValueError(
                    f"weight_years length {weight_years_array.shape[0]} "
                    f"does not match weight schedule length {self._weight_schedule.shape[0]}."
                )
            if weight_years_array.shape[0] > 1 and np.any(np.diff(weight_years_array) < 0):
                raise ValueError("weight_years must be sorted in non-decreasing order.")
            self._weight_years = weight_years_array
            self._weight_base_year = int(weight_years_array[0])

        self.rise_rate = float(rise_rate)
        self._decision_points = required_points
        self._num_decision_steps = max(self._decision_points - 1, 0)
        self._sea_level_offsets: Optional[np.ndarray] = None
        self._sea_level_deltas: Optional[np.ndarray] = None
        self._active_climate_scenario: Optional[str] = None
        self._climate_scenario_selected: Optional[str] = None
        if sea_level_scenarios is not None:
            if not isinstance(sea_level_scenarios, Mapping) or not sea_level_scenarios:
                raise ValueError("sea_level_scenarios must be a non-empty mapping when provided.")
            expected_len = self._decision_points
            scenario_map: dict[str, dict[str, np.ndarray]] = {}
            for name, params in sea_level_scenarios.items():
                if not isinstance(params, Mapping):
                    raise ValueError(f"Sea level parameters for scenario '{name}' must be a mapping.")
                processed: dict[str, np.ndarray] = {}
                for key in ("mu", "sigma", "lower", "upper"):
                    if key not in params:
                        raise ValueError(f"Scenario '{name}' missing '{key}' entries.")
                    arr = np.asarray(params[key], dtype=np.float32).reshape(-1)
                    if arr.size != expected_len:
                        raise ValueError(
                            f"Scenario '{name}' expected {expected_len} entries for '{key}', received {arr.size}."
                        )
                    if key == "sigma" and np.any(arr < 0.0):
                        raise ValueError(f"Scenario '{name}' contains negative sigma values.")
                    arr_copy = arr.copy()
                    arr_copy.setflags(write=False)
                    processed[key] = arr_copy
                scenario_map[str(name)] = processed
            if not scenario_map:
                raise ValueError("sea_level_scenarios mapping is empty.")
            preference = str(climate_scenario).strip()
            if not preference:
                raise ValueError("climate_scenario must be a non-empty string.")
            if preference != "All" and preference not in scenario_map:
                available = sorted(scenario_map.keys())
                raise ValueError(
                    f"climate_scenario '{preference}' not found in available scenarios: {available}"
                )
            self._sea_level_scenarios = scenario_map
            self._climate_scenario_selected = preference
        else:
            self._sea_level_scenarios = None
            self._climate_scenario_selected = None
        self.maximum_repair_time = float(maximum_repair_time)
        self.repeat_asset_penalty = float(repeat_asset_penalty)
        self.max_duration = float(max_duration)
        self.reward_scale = float(reward_scale)
        self.cost_weight = float(cost_weight)
        self.industrial_weight = float(np.clip(industrial_weight, 0.0, 1.0))

        # Load enlarged network configuration from asset and population data
        project_root = Path(__file__).resolve().parents[1]
        network_config = _build_default_network(project_root / "data")
        self.component_specs = list(network_config.components)
        expected_nodes = len(self.component_specs)
        if int(num_nodes) != expected_nodes:
            raise ValueError(
                f"num_nodes must be {expected_nodes} for the enlarged case study; received {num_nodes}."
            )
        self.N = expected_nodes

        self.component_names = [cfg.name for cfg in self.component_specs]
        self.name_to_index = {name: idx for idx, name in enumerate(self.component_names)}
        self.component_categories = {cfg.name: cfg.category for cfg in self.component_specs}
        self.component_dependencies = {cfg.name: tuple(cfg.dependencies) for cfg in self.component_specs}
        self.substation_generators = {
            cfg.name: tuple(cfg.generator_sources)
            for cfg in self.component_specs
            if cfg.category == "substation"
        }

        # Compute dependency closures for repair-time aggregation
        dependency_closure: Dict[str, Tuple[str, ...]] = {}

        def _closure(name: str) -> Tuple[str, ...]:
            if name in dependency_closure:
                return dependency_closure[name]
            closure = {name}
            for dep in self.component_dependencies.get(name, ()):  # tuples of component names
                closure.update(_closure(dep))
            ordered = tuple(sorted(closure))
            dependency_closure[name] = ordered
            return ordered

        for component_name in self.component_names:
            _closure(component_name)
        self.component_dependency_closure = dependency_closure
        self.component_upgradable = np.array([cfg.upgradable for cfg in self.component_specs], dtype=bool)
        self.component_can_fail = np.array([cfg.can_fail for cfg in self.component_specs], dtype=bool)
        self.component_coordinates = {
            cfg.name: tuple(cfg.coordinate)
            for cfg in self.component_specs
            if cfg.coordinate is not None
        }

        # Store configuration
        self.mc_samples = int(mc_samples)
        self.csv_path = csv_path
        self.copula_theta = float(copula_theta)
        self.max_depth = float(max_depth)
        self.threshold_depth = float(threshold_depth)

        # Constants for cost-to-height conversion (used in step and obs normalization)
        self.alpha = 4.0   # shape factor (3.5–4.6)
        self.u0 = 1800.0   # €/m cost at 1 m wall height
        self.beta = 1.2    # cost-height exponent

        # Normalization toggle
        self.normalize_observations = bool(normalize_observations)

        default_area = np.array([cfg.area for cfg in self.component_specs], dtype=np.float32)
        if area is None:
            area_array = default_area
        else:
            area_array = np.asarray(area, dtype=np.float32)
            assert area_array.shape == (self.N,), "area must have length N"
        self.area = area_array

        # Compressor and substation supply/population distributions derived from catalog data
        self.compressor_order = [
            "Comp_Sabine",
            "Comp_Calcasieu",
            "Comp_CedarBayou",
            "Comp_Freeport",
        ]
        self.substation_order = [
            "Sub_Harris_ShipChannel",
            "Sub_FortBend_Expansion",
            "Sub_Galveston_Island",
            "Sub_Brazoria_Gulf",
            "Sub_Jefferson_Orange",
            "Sub_Liberty_Chambers",
        ]

        def _array_from_mapping(order: Sequence[str], mapping: Dict[str, float], label: str) -> np.ndarray:
            values = []
            for key in order:
                if key not in mapping:
                    raise KeyError(f"'{key}' missing from {label} mapping when building network configuration")
                values.append(float(mapping[key]))
            return np.array(values, dtype=np.float32)

        self.compressor_gas_supply = _array_from_mapping(
            self.compressor_order,
            network_config.compressor_supply,
            "compressor supply",
        )
        self.compressor_people = _array_from_mapping(
            self.compressor_order,
            network_config.compressor_people,
            "compressor population",
        )
        self.substation_power_supply = _array_from_mapping(
            self.substation_order,
            network_config.substation_supply,
            "substation supply",
        )
        self.substation_people = _array_from_mapping(
            self.substation_order,
            network_config.substation_people,
            "substation population",
        )

        self.compressor_industrial = _array_from_mapping(
            self.compressor_order,
            network_config.compressor_industrial,
            "compressor industrial",
        )
        self.substation_industrial = _array_from_mapping(
            self.substation_order,
            network_config.substation_industrial,
            "substation industrial",
        )

        self.compressor_names = list(self.compressor_order)
        self.substation_names = list(self.substation_order)
        self.lng_names = [cfg.name for cfg in self.component_specs if cfg.category == "lng"]
        self.generation_names = [
            cfg.name for cfg in self.component_specs if cfg.category in {"renewable", "thermal"}
        ]
        self.compressor_indices = [self.name_to_index[name] for name in self.compressor_order]
        self.substation_indices = [self.name_to_index[name] for name in self.substation_order]

        self.total_gas_supply = float(self.compressor_gas_supply.sum())
        self.total_gas_people = float(self.compressor_people.sum())
        self.total_power_supply = float(self.substation_power_supply.sum())
        self.total_power_people = float(self.substation_people.sum())
        self.total_gas_industrial = float(self.compressor_industrial.sum())
        self.total_power_industrial = float(self.substation_industrial.sum())

        if self.total_gas_supply > 0.0:
            self.compressor_supply_fraction = (self.compressor_gas_supply / self.total_gas_supply).astype(np.float32)
        else:
            self.compressor_supply_fraction = np.zeros_like(self.compressor_gas_supply, dtype=np.float32)

        if self.total_gas_people > 0.0:
            self.compressor_people_fraction = (self.compressor_people / self.total_gas_people).astype(np.float32)
        else:
            self.compressor_people_fraction = np.zeros_like(self.compressor_people, dtype=np.float32)

        if self.total_power_supply > 0.0:
            self.substation_supply_fraction = (self.substation_power_supply / self.total_power_supply).astype(np.float32)
        else:
            self.substation_supply_fraction = np.zeros_like(self.substation_power_supply, dtype=np.float32)

        if self.total_power_people > 0.0:
            self.substation_people_fraction = (self.substation_people / self.total_power_people).astype(np.float32)
        else:
            self.substation_people_fraction = np.zeros_like(self.substation_people, dtype=np.float32)

        if self.total_gas_industrial > 0.0:
            self.compressor_industrial_fraction = (self.compressor_industrial / self.total_gas_industrial).astype(np.float32)
        else:
            self.compressor_industrial_fraction = np.zeros_like(self.compressor_industrial, dtype=np.float32)

        if self.total_power_industrial > 0.0:
            self.substation_industrial_fraction = (self.substation_industrial / self.total_power_industrial).astype(np.float32)
        else:
            self.substation_industrial_fraction = np.zeros_like(self.substation_industrial, dtype=np.float32)

        if initial_wall_height is None:
            self.initial_wall_height = np.zeros(self.N, dtype=np.float32)
        else:
            self.initial_wall_height = np.asarray(initial_wall_height, dtype=np.float32)
            assert self.initial_wall_height.shape == (self.N,), "initial_wall_height must have length N"

        self.height_levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
        self.cost_prefactor = (self.alpha * self.u0 * np.sqrt(self.area.astype(np.float32))).astype(np.float32)

        # Multi-discrete action: pick a height increment option for each component
        action_sizes = np.full(self.N, len(self.height_levels), dtype=np.int64)
        self.action_space = spaces.MultiDiscrete(action_sizes)

        # Observation: [wall_height (N), current_year (1), sea_level_offset (1), econ_impact (1), social_impact (1)]
        # Reference scale for normalisation: maximum selectable increment
        self.h_ref = np.full(self.N, self.height_levels[-1], dtype=np.float32)
        if self.normalize_observations:
            # All features scaled to [0,1]
            low_obs = np.zeros(self.N + 4, dtype=np.float32)
            high_obs = np.ones(self.N + 4, dtype=np.float32)
        else:
            # Raw ranges (year in [0, T]; impacts nonnegative; wall heights unbounded above)
            high_obs = np.array(
                [1] * self.N + [float(self.T)] + [np.finfo(np.float32).max] * 3,
                dtype=np.float32,
            )
            low_obs = np.zeros(self.N + 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # RNG seed / state
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.rng = self.np_random

        # Internal state
        self._year = 0
        self._econ_impact = 0.0
        self._social_impact = 0.0

        # Track current and previous component-wise means for delta-based reward
        self.gas_loss_mean = 0.0
        self.elec_loss_mean = 0.0
        self.gas_soc_mean = 0.0
        self.elec_soc_mean = 0.0
        self._last_positive_mask = np.zeros(self.N, dtype=bool)
        self._latest_breakdown: Dict[str, float] = {}

        # --- Cache CSV and per-row values for fast nearest-neighbour lookup ---
        df_raw = pd.read_csv(self.csv_path, sep=None, engine='python', header=0, index_col=0)
        df_raw.columns = [float(c) for c in df_raw.columns]
        df = df_raw.apply(pd.to_numeric, errors='coerce')
        self.input_grid = np.array(sorted(df.columns), dtype=float)

        self.ROW_FOR_COMPONENT = {cfg.name: cfg.hazard_key for cfg in self.component_specs}
        required_rows = {cfg.hazard_key for cfg in self.component_specs}
        missing_rows = [r for r in required_rows if r not in df.index]
        if missing_rows:
            raise KeyError(f"Missing rows in CSV for points: {missing_rows}. Available rows: {list(df.index)}")

        # Precompute depth value arrays per component in the same input_grid order
        self.depth_vals = {
            name: df.loc[row].reindex(self.input_grid).to_numpy(dtype=np.float32)
            for name, row in self.ROW_FOR_COMPONENT.items()
        }
        self.improvement_height = np.zeros(self.N, dtype=np.float32)
        self._base_heights: Optional[np.ndarray] = None
        self._base_durations: Optional[np.ndarray] = None
        self._prev_metrics: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._prev_loss: float = 0.0

    def _set_weight_index(self, index: int) -> None:
        """Update the active weight vector according to the schedule."""
        if self._weight_schedule.shape[0] == 0:
            raise ValueError("Weight schedule is empty.")
        max_idx = self._weight_schedule.shape[0] - 1
        safe_index = int(index)
        if safe_index < 0:
            safe_index = 0
        elif safe_index > max_idx:
            safe_index = max_idx
        self._current_weight_index = safe_index
        self.weights = self._weight_schedule[safe_index]

    # ------------------------
    # Helper computations
    # ------------------------
    def _compute_costs(self, height_deltas: np.ndarray) -> np.ndarray:
        """Return per-component construction costs for the proposed height increments."""
        deltas = np.asarray(height_deltas, dtype=np.float32)
        non_negative = np.clip(deltas, a_min=0.0, a_max=None)
        costs = self.cost_prefactor * np.power(non_negative, self.beta)
        return costs.astype(np.float32)

    def _sample_truncated_normal(self, mu: float, sigma: float, lower: float, upper: float) -> np.float32:
        """Sample from a truncated normal via simple rejection sampling."""
        low = float(lower)
        high = float(upper)
        if low > high:
            low, high = high, low
        if not np.isfinite(mu):
            mu = 0.0
        if sigma <= 0.0 or not np.isfinite(sigma):
            return np.float32(np.clip(mu, low, high))
        for _ in range(64):
            draw = float(self.rng.normal(mu, sigma))
            if low <= draw <= high:
                return np.float32(draw)
        return np.float32(np.clip(mu, low, high))

    def _sample_sea_level_path(self) -> None:
        """Sample cumulative sea level offsets for the upcoming episode."""
        if self._sea_level_scenarios is None:
            self._sea_level_offsets = None
            self._sea_level_deltas = None
            self._active_climate_scenario = None
            return

        preference = self._climate_scenario_selected or "All"
        if preference == "All":
            scenario_name = str(self.rng.choice(list(self._sea_level_scenarios.keys())))
        else:
            scenario_name = preference

        params = self._sea_level_scenarios[scenario_name]
        mu = params["mu"]
        sigma = params["sigma"]
        lower = params["lower"]
        upper = params["upper"]
        n_points = mu.shape[0]
        levels = np.empty(n_points, dtype=np.float32)
        prev_value: Optional[float] = None
        for idx in range(n_points):
            low = float(lower[idx])
            high = float(upper[idx])
            if prev_value is not None:
                low = max(low, prev_value+0.01)
            if high < low:
                high = low
            sampled = float(
                self._sample_truncated_normal(
                    float(mu[idx]),
                    float(sigma[idx]),
                    low,
                    high,
                )
            )
            if prev_value is not None and sampled < prev_value:
                sampled = prev_value
            levels[idx] = np.float32(sampled)
            prev_value = sampled

        base_level = levels[0]
        offsets = (levels - base_level).astype(np.float32)
        if offsets.size <= 1:
            deltas = np.zeros(0, dtype=np.float32)
        else:
            deltas = np.diff(offsets).astype(np.float32)

        self._sea_level_deltas = deltas
        self._sea_level_offsets = offsets
        self._active_climate_scenario = scenario_name

    def _sea_level_offset_for_year(self, year: float) -> np.float32:
        """Return the cumulative sea level adjustment for the provided elapsed year."""
        if self._sea_level_offsets is None:
            return np.float32(year * self.rise_rate)
        if self.year_step <= 0:
            return np.float32(self._sea_level_offsets[-1])
        idx = int(np.floor(year / self.year_step))
        idx = max(0, min(idx, self._sea_level_offsets.shape[0] - 1))
        return np.float32(self._sea_level_offsets[idx])

    def _generate_hazard_samples(self) -> None:
        """Draw and cache the Monte Carlo flood samples reused across the episode."""
        seed_val = int(self.rng.integers(0, 2**31 - 1))
        df_cop = sample_flood(self.mc_samples, self.copula_theta, seed=seed_val)
        heights = df_cop["height"].to_numpy(dtype=np.float32)
        durations = df_cop["duration"].to_numpy(dtype=np.float32)

        resample_count = 0
        while True:
            mask = (heights > self.max_depth) | (durations > self.max_duration)
            if not np.any(mask) or resample_count > 10:
                break
            n_bad = int(mask.sum())
            resample_seed = int(self.rng.integers(0, 2**31 - 1))
            df_res = sample_flood(n_bad, self.copula_theta, seed=resample_seed)
            heights[mask] = df_res["height"].to_numpy(dtype=np.float32)
            durations[mask] = df_res["duration"].to_numpy(dtype=np.float32)
            resample_count += 1

        self._base_heights = heights
        self._base_durations = durations

    def _generate_random_streams(self) -> dict[str, np.ndarray]:
        """Pre-draw random numbers reused across metric evaluations."""
        component_uniform = self.rng.random((self.N, self.mc_samples)).astype(np.float32, copy=False)
        repair_normals = self.rng.standard_normal((self.N, self.mc_samples)).astype(np.float32, copy=False)
        return {
            "component_uniform": component_uniform,
            "repair_normals": repair_normals,
        }

    def _compute_metrics(
        self,
        improvement_height: np.ndarray,
        year: float,
        random_cache: Optional[dict[str, np.ndarray]] = None,
        *,
        return_cache: bool = False,
        return_details: bool = False,
    ):
        if self._base_heights is None or self._base_durations is None:
            self._generate_hazard_samples()

        improvement_height = np.asarray(improvement_height, dtype=np.float32)
        offset = self._sea_level_offset_for_year(year)
        samples = self._base_heights + offset
        durations = self._base_durations

        grid = self.input_grid
        idx = np.searchsorted(grid, samples, side="left")
        idx_right = np.clip(idx, 0, grid.size - 1)
        idx_left = np.clip(idx - 1, 0, grid.size - 1)
        choose_left = (idx <= 0) | (np.abs(samples - grid[idx_left]) <= np.abs(samples - grid[idx_right]))
        nearest_idx = np.where(choose_left, idx_left, idx_right)

        depth_samples = {
            name: self.depth_vals[name][nearest_idx]
            for name in self.component_names
        }

        cache = random_cache if random_cache is not None else self._generate_random_streams()
        U = cache["component_uniform"]
        repair_normals = cache["repair_normals"]

        functional: Dict[str, np.ndarray] = {}
        repair_times: Dict[str, np.ndarray] = {}

        for idx_comp, name in enumerate(self.component_names):
            category = self.component_categories[name]
            frag_fn = FRAGILITY_FUNCTIONS[category]
            repair_fn = REPAIR_FUNCTIONS[category]
            depths = depth_samples[name]
            if not self.component_can_fail[idx_comp]:
                status = np.ones(depths.shape, dtype=bool)
                functional[name] = status
                repair_times[name] = np.zeros(depths.shape, dtype=np.float32)
                continue
            fragility = np.clip(frag_fn(depths, improvement_height[idx_comp]), 0.0, 1.0)
            status = U[idx_comp] >= fragility
            functional[name] = status
            repair_draws = repair_fn(depths, rng=self.rng, normals=repair_normals[idx_comp])
            repair_times[name] = np.where(status, 0.0, repair_draws).astype(np.float32)

        availability: Dict[str, np.ndarray] = {}

        def resolve_availability(component: str) -> np.ndarray:
            if component in availability:
                return availability[component]
            base = functional[component].copy()
            for dep in self.component_dependencies.get(component, ()):  # dependencies follow logical AND
                base = np.logical_and(base, resolve_availability(dep))
            availability[component] = base
            return base

        for name in self.component_names:
            if self.component_categories[name] != "substation":
                resolve_availability(name)

        for name in self.substation_names:
            base = functional[name].copy()
            for dep in self.component_dependencies.get(name, ()):  # kept for completeness
                base = np.logical_and(base, resolve_availability(dep))
            generators = self.substation_generators.get(name, ())
            if generators:
                supply = np.zeros_like(base, dtype=bool)
                for gen in generators:
                    supply = np.logical_or(supply, resolve_availability(gen))
            else:
                supply = np.ones_like(base, dtype=bool)
            availability[name] = np.logical_and(base, supply)

        flood_duration = durations.astype(np.float32)
        component_downtime: Dict[str, np.ndarray] = {}

        for name in self.component_names:
            if self.component_categories[name] == "substation":
                continue
            repair_sum = np.zeros_like(flood_duration, dtype=np.float32)
            for dep in self.component_dependency_closure[name]:
                repair_sum += repair_times[dep]
            component_downtime[name] = np.where(
                ~availability[name],
                flood_duration + repair_sum,
                0.0,
            ).astype(np.float32)

        for name in self.substation_names:
            base_failure = np.where(~functional[name], flood_duration + repair_times[name], 0.0).astype(np.float32)
            dependency_issue = np.zeros_like(base_failure)
            for dep in self.component_dependencies.get(name, ()):  # typically empty
                dependency_issue = np.maximum(
                    dependency_issue,
                    component_downtime.get(dep, np.zeros_like(base_failure)),
                )
            generator_issue = np.zeros_like(base_failure)
            for gen in self.substation_generators.get(name, ()):  # generators feeding this node
                generator_issue = np.maximum(
                    generator_issue,
                    component_downtime.get(gen, np.zeros_like(base_failure)),
                )
            total_issue = np.maximum(base_failure, dependency_issue)
            total_issue = np.maximum(total_issue, generator_issue)
            component_downtime[name] = np.where(~availability[name], total_issue, 0.0).astype(np.float32)

        max_event_time = np.float32(self.max_duration + self.maximum_repair_time)
        max_event_time = np.clip(max_event_time, a_min=np.float32(1e-6), a_max=None)

        gas_services = np.stack([availability[name] for name in self.compressor_order])
        gas_downtime = np.stack([component_downtime[name] for name in self.compressor_order])
        gas_time_ratio = np.clip(gas_downtime / max_event_time, a_min=0.0, a_max=1.0)
        gas_unavailable = 1.0 - gas_services.astype(np.float32)
        gas_supply_fraction = gas_unavailable * self.compressor_supply_fraction[:, None]
        gas_supply_loss_samples = (gas_supply_fraction * gas_time_ratio).sum(axis=0)
        gas_industrial_fraction = gas_unavailable * self.compressor_industrial_fraction[:, None]
        gas_industrial_loss_samples = (gas_industrial_fraction * gas_time_ratio).sum(axis=0)
        gas_loss_samples = (
            (1.0 - self.industrial_weight) * gas_supply_loss_samples
            + self.industrial_weight * gas_industrial_loss_samples
        )
        gas_social_fraction = gas_unavailable * self.compressor_people_fraction[:, None]
        gas_social_samples = (gas_social_fraction * gas_time_ratio).sum(axis=0)

        power_services = np.stack([availability[name] for name in self.substation_order])
        power_downtime = np.stack([component_downtime[name] for name in self.substation_order])
        power_time_ratio = np.clip(power_downtime / max_event_time, a_min=0.0, a_max=1.0)
        power_unavailable = 1.0 - power_services.astype(np.float32)
        power_supply_fraction = power_unavailable * self.substation_supply_fraction[:, None]
        power_supply_loss_samples = (power_supply_fraction * power_time_ratio).sum(axis=0)
        power_industrial_fraction = power_unavailable * self.substation_industrial_fraction[:, None]
        power_industrial_loss_samples = (power_industrial_fraction * power_time_ratio).sum(axis=0)
        electricity_loss_samples = (
            (1.0 - self.industrial_weight) * power_supply_loss_samples
            + self.industrial_weight * power_industrial_loss_samples
        )
        power_social_fraction = power_unavailable * self.substation_people_fraction[:, None]
        electricity_social_samples = (power_social_fraction * power_time_ratio).sum(axis=0)

        gas_supply_mean = float(gas_supply_loss_samples.mean())
        gas_industrial_mean = float(gas_industrial_loss_samples.mean())
        gas_loss_mean = float(gas_loss_samples.mean())
        elec_supply_mean = float(power_supply_loss_samples.mean())
        elec_industrial_mean = float(power_industrial_loss_samples.mean())
        elec_loss_mean = float(electricity_loss_samples.mean())
        gas_social_mean = float(gas_social_samples.mean())
        elec_social_mean = float(electricity_social_samples.mean())

        self._latest_breakdown = {
            "gas_supply_mean": gas_supply_mean,
            "gas_industrial_mean": gas_industrial_mean,
            "gas_total_mean": gas_loss_mean,
            "elec_supply_mean": elec_supply_mean,
            "elec_industrial_mean": elec_industrial_mean,
            "elec_total_mean": elec_loss_mean,
            "gas_social_mean": gas_social_mean,
            "elec_social_mean": elec_social_mean,
        }

        depth_means = {
            name: float(depth_samples[name].mean() + offset)
            for name in self.component_names
        }
        depth_p95 = {
            name: float(np.percentile(depth_samples[name], 95) + offset)
            for name in self.component_names
        }
        depth_max = {
            name: float(depth_samples[name].max() + offset)
            for name in self.component_names
        }

        metrics = (
            gas_loss_mean,
            elec_loss_mean,
            gas_social_mean,
            elec_social_mean,
        )
        if return_cache and return_details:
            return metrics, cache, (depth_means, depth_p95, depth_max)
        if return_cache:
            return metrics, cache
        if return_details:
            return metrics, (depth_means, depth_p95, depth_max)
        return metrics

    def _compute_loss(self, metrics: tuple[float, float, float, float]) -> float:
        gas_loss, elec_loss, gas_social, elec_social = metrics
        (
            w_gas,
            w_electricity,
            w_gas_loss,
            w_gas_social,
            w_electricity_loss,
            w_electricity_social,
        ) = self.weights
        return (
            w_gas * (gas_loss * w_gas_loss + gas_social * w_gas_social)
            + w_electricity * (
                elec_loss * w_electricity_loss + elec_social * w_electricity_social
            )
        )

    def _obs(self) -> np.ndarray:
        sea_level_offset = float(self._sea_level_offset_for_year(self._year))
        if self.normalize_observations:
            # Normalize wall heights by (h_ref * T), clip to [0,1]
            denom = (self.h_ref * self.T).astype(np.float32)
            denom = np.where(denom <= 0.0, 1.0, denom)
            wh = np.clip(self.wall_height / denom, 0.0, 1.0).astype(np.float32)
            yr = np.float32(self._year / self.T)
            if self.max_depth > 0.0:
                sea = np.float32(np.clip(sea_level_offset / self.max_depth, 0.0, 1.0))
            else:
                sea = np.float32(0.0)
            econ = np.float32(min(self._econ_impact / 2.0, 1.0))
            soc = np.float32(min(self._social_impact / 2.0, 1.0))
            tail = np.array([yr, sea, econ, soc], dtype=np.float32)
            return np.concatenate([wh, tail])
        else:
            return np.concatenate([
                self.wall_height.astype(np.float32),
                np.array([self._year, sea_level_offset, self._econ_impact, self._social_impact], dtype=np.float32),
            ])

    # ------------------------
    # Gym API
    # ------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            self.rng = self.np_random
        # Start from new
        self.wall_height = self.initial_wall_height.copy()
        self.improvement_height = np.zeros(self.N, dtype=np.float32)
        self._year = 0
        self._set_weight_index(0)
        self._sample_sea_level_path()
        self._base_heights = None
        self._base_durations = None
        metrics = self._compute_metrics(self.improvement_height, year=self._year)
        (
            self.gas_loss_mean,
            self.elec_loss_mean,
            self.gas_soc_mean,
            self.elec_soc_mean,
        ) = metrics
        self._econ_impact = self.gas_loss_mean + self.elec_loss_mean
        self._social_impact = self.gas_soc_mean + self.elec_soc_mean
        self._last_positive_mask = np.zeros(self.N, dtype=bool)
        self._prev_metrics = metrics
        self._prev_loss = self._compute_loss(metrics)
        info = {}
        if self._active_climate_scenario is not None and self._sea_level_offsets is not None:
            info["climate_scenario"] = self._active_climate_scenario
            info["sea_level_offset"] = float(self._sea_level_offsets[0])
        return self._obs(), info

    def step(self, action):
        # Improvements are wall-height increments chosen from discrete levels per component.
        action_array = np.asarray(action)
        if action_array.size != self.N:
            raise ValueError(f"Expected action with {self.N} entries, received shape {action_array.shape}.")
        action_array = action_array.astype(np.int64).reshape(self.N)
        np.clip(action_array, 0, len(self.height_levels) - 1, out=action_array)

        intended_heights = self.height_levels[action_array]
        non_upgradable_mask = ~self.component_upgradable
        if np.any(non_upgradable_mask):
            intended_heights = intended_heights.astype(np.float32)
            intended_heights[non_upgradable_mask] = 0.0
        executed_heights = intended_heights.astype(np.float32).copy()
        prev_improvement = self.improvement_height.astype(np.float32).copy()

        # Penalise repeated upgrades on consecutive steps
        repeat_mask = self._last_positive_mask & (executed_heights > 0.0) & self.component_upgradable
        repeat_count = int(repeat_mask.sum())
        repeat_penalty_total = 0.0
        if repeat_count:
            executed_heights[repeat_mask] = 0.0
            repeat_penalty_total = self.repeat_asset_penalty * repeat_count

        costs = self._compute_costs(executed_heights)
        total_cost = float(costs.sum())
        over_budget_penalty_total = 0.0
        over_budget_amount = 0.0
        trimmed_assets: list[int] = []
        if total_cost > self.budget:
            # Drop the most expensive upgrades until affordable
            order = np.argsort(costs)[::-1]
            for idx in order:
                if executed_heights[idx] <= 0.0:
                    continue
                over_budget_amount += float(costs[idx])
                trimmed_assets.append(int(idx))
                total_cost -= float(costs[idx])
                executed_heights[idx] = 0.0
                if total_cost <= self.budget:
                    break
            if trimmed_assets:
                costs = self._compute_costs(executed_heights)
                total_cost = float(costs.sum())
                over_budget_penalty_total = self.cost_weight * (over_budget_amount / self.budget if self.budget > 0.0 else over_budget_amount)

        post_improvement = np.maximum(prev_improvement + executed_heights, 0.0)
        self._last_positive_mask = (executed_heights > 0.0) & self.component_upgradable
        self._year += self.year_step
        current_year = self._year
        sea_level_step_index = current_year // self.year_step if self.year_step > 0 else 0
        if self._weight_years is not None:
            target_year = self._weight_base_year + current_year
            idx = int(np.searchsorted(self._weight_years, target_year, side="right") - 1)
        else:
            idx = current_year // self.year_step
        self._set_weight_index(idx)
        prev_loss = self._compute_loss(self._prev_metrics)

        base_metrics, random_cache, depth_details_base = self._compute_metrics(
            prev_improvement,
            year=current_year,
            return_cache=True,
            return_details=True,
        )
        base_depth_means, base_depth_p95, base_depth_max = depth_details_base

        new_metrics, depth_details_new = self._compute_metrics(
            post_improvement,
            year=current_year,
            random_cache=random_cache,
            return_details=True,
        )
        new_depth_means, new_depth_p95, new_depth_max = depth_details_new

        (
            base_gas_loss,
            base_elec_loss,
            base_gas_soc,
            base_elec_soc,
        ) = base_metrics
        (
            new_gas_loss,
            new_elec_loss,
            new_gas_soc,
            new_elec_soc,
        ) = new_metrics

        base_loss = self._compute_loss(base_metrics)
        new_loss = self._compute_loss(new_metrics)
        climate_drift = base_loss - prev_loss
        action_gain = base_loss - new_loss
        reward_delta = prev_loss - new_loss

        self.improvement_height = post_improvement.astype(np.float32)
        self.wall_height = self.improvement_height + self.initial_wall_height
        self.gas_loss_mean = new_gas_loss
        self.elec_loss_mean = new_elec_loss
        self.gas_soc_mean = new_gas_soc
        self.elec_soc_mean = new_elec_soc
        self._econ_impact = self.gas_loss_mean + self.elec_loss_mean
        self._social_impact = self.gas_soc_mean + self.elec_soc_mean
        self._prev_metrics = new_metrics
        self._prev_loss = new_loss

        reward_delta *= self.reward_scale
        unused_budget = float(max(self.budget - total_cost, 0.0))
        if self.budget > 0.0:
            normalized_cost = total_cost / self.budget
            normalized_unused = unused_budget / self.budget
        else:
            normalized_cost = total_cost
            normalized_unused = 0.0
        unused_budget_penalty = self.cost_weight * normalized_unused
        reward = float(
            reward_delta
            - unused_budget_penalty
            + repeat_penalty_total
            - over_budget_penalty_total
        )

        terminated = bool(self._year >= self.T)
        truncated = False
        info = {
            "econ_impact": self._econ_impact,
            "social_impact": self._social_impact,
            "intended_heights": intended_heights.astype(np.float32),
            "executed_heights": executed_heights.astype(np.float32),
            "costs": costs.astype(np.float32),
            "total_cost": float(total_cost),
            "unused_budget": float(max(self.budget - total_cost, 0.0)),
            "normalized_cost": float(normalized_cost),
            "normalized_unused_budget": float(normalized_unused),
            "unused_budget_penalty": float(unused_budget_penalty),
            "unused_budget_penalty_signed": float(-unused_budget_penalty),
            "over_budget_penalty": float(over_budget_penalty_total),
            "over_budget_penalty_signed": float(-over_budget_penalty_total),
            "over_budget_amount": float(over_budget_amount),
            "reward_delta": float(reward_delta),
            "prev_loss": float(prev_loss),
            "base_loss": float(base_loss),
            "new_loss": float(new_loss),
            "climate_drift": float(climate_drift),
            "action_gain": float(action_gain),
        }
        breakdown = getattr(self, "_latest_breakdown", None)
        if breakdown:
            info["gas_supply_loss_mean"] = float(breakdown.get("gas_supply_mean", 0.0))
            info["gas_industrial_loss_mean"] = float(breakdown.get("gas_industrial_mean", 0.0))
            info["elec_supply_loss_mean"] = float(breakdown.get("elec_supply_mean", 0.0))
            info["elec_industrial_loss_mean"] = float(breakdown.get("elec_industrial_mean", 0.0))
            info["gas_social_loss_mean"] = float(breakdown.get("gas_social_mean", 0.0))
            info["elec_social_loss_mean"] = float(breakdown.get("elec_social_mean", 0.0))
        info["repeat_penalty"] = float(repeat_penalty_total)
        if self._active_climate_scenario is not None:
            info["climate_scenario"] = self._active_climate_scenario
            info["sea_level_offset"] = float(self._sea_level_offset_for_year(current_year))
            if self._sea_level_deltas is not None:
                delta_idx = sea_level_step_index - 1
                if 0 <= delta_idx < self._sea_level_deltas.shape[0]:
                    info["sea_level_delta"] = float(self._sea_level_deltas[delta_idx])
        if repeat_count:
            info["repeat_penalty_assets"] = [int(i) for i in np.nonzero(repeat_mask)[0]]
        if trimmed_assets:
            info["trimmed_assets"] = trimmed_assets
        for asset_name, depth_val in new_depth_means.items():
            info[f"expected_depth_mean_{asset_name}"] = float(depth_val)
        for asset_name, depth_val in new_depth_p95.items():
            info[f"expected_depth_p95_{asset_name}"] = float(depth_val)
        for asset_name, depth_val in new_depth_max.items():
            info[f"expected_depth_max_{asset_name}"] = float(depth_val)
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        print(
            f"Year {self._year}/{self.T} | EconImpact={self._econ_impact:.3f} | SocialImpact={self._social_impact:.3f}"
        )

    def close(self):
        pass
