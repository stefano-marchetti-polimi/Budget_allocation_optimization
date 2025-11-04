from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import rasterio as rio
from pygeoflood import pyGeoFlood

# Namespace constants reused from utils.environment
EXCEL_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
EXCEL_NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

# Default paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_DEM_PATH = DEFAULT_DATA_DIR / "houston_example_DEM_30m.tif"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "coastal_inundation_samples.csv"

# Representative offshore coordinate for boundary forcing
DEFAULT_OCEAN_COORD = (317_540.0, 3_272_260.0)


def _format_component_name(raw: object) -> str:
    text = " ".join(str(raw).strip().split())
    for token in ["'", ".", ","]:
        text = text.replace(token, "")
    text = text.replace("-", " ").replace("/", " ")
    parts = [part for part in text.split(" ") if part]
    return "_".join(parts)


def _parse_cell_value(cell: ET.Element, shared_strings: Mapping[int, str]) -> Optional[object]:
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

    with zipfile.ZipFile(path) as zf:
        shared_strings: Dict[int, str] = {}
        shared_path = "xl/sharedStrings.xml"
        if shared_path in zf.namelist():
            root = ET.fromstring(zf.read(shared_path))
            strings: List[str] = []
            for si in root.findall(f"{{{EXCEL_NS_MAIN}}}si"):
                texts = [t.text or "" for t in si.findall(f".//{{{EXCEL_NS_MAIN}}}t")]
                strings.append("".join(texts))
            shared_strings = {idx: text for idx, text in enumerate(strings)}

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall(f"{{{EXCEL_NS_REL}}}Relationship")
        }

        rel_target: Optional[str] = None
        for sheet in workbook.findall(f"{{{EXCEL_NS_MAIN}}}sheets/{{{EXCEL_NS_MAIN}}}sheet"):
            if sheet.attrib.get("name") != sheet_name:
                continue
            rel_id = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            rel_target = rel_map.get(rel_id)
            break
        if rel_target is None:
            available = [
                sheet.attrib.get("name")
                for sheet in workbook.findall(f"{{{EXCEL_NS_MAIN}}}sheets/{{{EXCEL_NS_MAIN}}}sheet")
            ]
            raise ValueError(f"Sheet '{sheet_name}' not found. Available: {available}")

        sheet_xml = ET.fromstring(zf.read(f"xl/{rel_target}"))

    rows: List[Dict[str, object]] = []
    header: Dict[str, str] = {}
    for row in sheet_xml.findall(f"{{{EXCEL_NS_MAIN}}}sheetData/{{{EXCEL_NS_MAIN}}}row"):
        row_idx = int(row.attrib.get("r", "0"))
        cells: Dict[str, object] = {}
        for cell in row.findall(f"{{{EXCEL_NS_MAIN}}}c"):
            ref = cell.attrib.get("r", "")
            column = "".join(ch for ch in ref if ch.isalpha())
            if not column:
                continue
            value = _parse_cell_value(cell, shared_strings)
            if value is None:
                continue
            cells[column] = value
        if not cells:
            continue
        if row_idx == 1:
            header = {
                column: str(value).strip()
                for column, value in cells.items()
                if str(value).strip()
            }
            continue
        if not header:
            continue
        record: Dict[str, object] = {}
        for column, name in header.items():
            if column in cells:
                record[name] = cells[column]
        if record:
            rows.append(record)
    return rows


def _row_coordinate(row: Mapping[str, object]) -> Tuple[float, float]:
    east = row.get("Easting (m)")
    north = row.get("Northing (m)")
    if east is None or north is None:
        raise KeyError("Missing 'Easting (m)'/'Northing (m)' in asset catalog row.")
    return float(east), float(north)


def _capacity_from_row(row: Mapping[str, object]) -> float:
    value = row.get("Power (MWe)")
    if value is None:
        return 0.0
    return float(value)


def _weighted_centroid(
    asset_names: Iterable[str],
    coordinate_map: Mapping[str, np.ndarray],
    capacity_map: Mapping[str, float],
) -> Optional[np.ndarray]:
    coords: List[np.ndarray] = []
    weights: List[float] = []
    for asset in asset_names:
        coord = coordinate_map.get(asset)
        if coord is None:
            continue
        weight = float(capacity_map.get(asset, 0.0))
        if not np.isfinite(weight) or weight <= 0.0:
            weight = 1.0
        coords.append(coord.astype(float, copy=False))
        weights.append(weight)
    if not coords:
        return None
    weight_arr = np.asarray(weights, dtype=float)
    coord_arr = np.stack(coords, axis=0)
    centroid = np.sum(coord_arr * weight_arr[:, None], axis=0) / np.sum(weight_arr)
    return centroid


@dataclass(frozen=True)
class AssetSpec:
    name: str
    source_type: str
    source_key: str


GENERATION_ASSETS: Tuple[AssetSpec, ...] = (
    AssetSpec("Roy_S_Nelson_Coal", "electric", "Roy_S_Nelson"),
    AssetSpec("Cottonwood_Bayou_Solar", "electric", "Cottonwood_Bayou"),
    AssetSpec("Galveston_1_Wind", "electric", "Galveston_1"),
    AssetSpec("Galveston_2_Wind", "electric", "Galveston_2"),
    AssetSpec("Lake_Charles_Wind", "electric", "Lake_Charles"),
    AssetSpec("Liberty_1_Solar", "electric", "Liberty_1"),
    AssetSpec("Trinity_River_Solar", "electric", "Trinity_river_solar"),
    AssetSpec("Myrtle_Solar", "electric", "Myrtle_solar"),
    AssetSpec("Red_Bluff_Road_Solar", "electric", "Red_Bluff_Road_Solar"),
    AssetSpec("Brazoria_West_Solar", "electric", "Brazoria_West"),
)

THERMAL_ASSETS: Tuple[AssetSpec, ...] = (
    AssetSpec("Sabine_CCGT", "gas", "Sabine"),
    AssetSpec("Port_Arthur_CCGT", "gas", "Porth_Arthur"),
    AssetSpec("Lake_Charles_CCGT", "gas", "Lake_Charles"),
    AssetSpec("Cottonwood_Gas", "gas", "Cottonwood"),
    AssetSpec("Cedar_Bayou_CCGT", "gas", "Cedar_Bayou"),
    AssetSpec("Channelview_CCGT", "gas", "Channelview"),
    AssetSpec("Bacliff_CCGT", "gas", "Bacliff"),
    AssetSpec("Galveston_CCGT", "gas", "Galveston"),
    AssetSpec("Freeport_CCGT", "gas", "Free_Port"),
)

COMPRESSOR_FEEDS: Mapping[str, Tuple[str, ...]] = {
    "Comp_Sabine": ("Sabine_CCGT", "Port_Arthur_CCGT"),
    "Comp_Calcasieu": ("Lake_Charles_CCGT", "Cottonwood_Gas"),
    "Comp_CedarBayou": ("Cedar_Bayou_CCGT", "Channelview_CCGT"),
    "Comp_Freeport": ("Freeport_CCGT", "Bacliff_CCGT", "Galveston_CCGT"),
}

SUBSTATION_GENERATORS: Mapping[str, Tuple[str, ...]] = {
    "Sub_Harris_ShipChannel": (
        "Channelview_CCGT",
        "Cedar_Bayou_CCGT",
        "Galveston_1_Wind",
        "Galveston_2_Wind",
        "Red_Bluff_Road_Solar",
    ),
    "Sub_FortBend_Expansion": (
        "Freeport_CCGT",
        "Cottonwood_Bayou_Solar",
        "Myrtle_Solar",
    ),
    "Sub_Galveston_Island": (
        "Galveston_1_Wind",
        "Galveston_2_Wind",
        "Galveston_CCGT",
        "Bacliff_CCGT",
    ),
    "Sub_Brazoria_Gulf": (
        "Freeport_CCGT",
        "Cottonwood_Bayou_Solar",
        "Bacliff_CCGT",
        "Brazoria_West_Solar",
    ),
    "Sub_Jefferson_Orange": (
        "Roy_S_Nelson_Coal",
        "Sabine_CCGT",
        "Port_Arthur_CCGT",
        "Lake_Charles_Wind",
        "Lake_Charles_CCGT",
    ),
    "Sub_Liberty_Chambers": (
        "Cedar_Bayou_CCGT",
        "Channelview_CCGT",
        "Cottonwood_Gas",
        "Liberty_1_Solar",
        "Trinity_River_Solar",
    ),
}

def _load_asset_catalog(data_dir: Path) -> Tuple[MutableMapping[str, Mapping[str, object]], MutableMapping[str, Mapping[str, object]]]:
    asset_path = data_dir / "Assets Catalog filtered.xlsx"
    electric_rows = _read_excel_sheet(asset_path, "Electric Network")
    gas_rows = _read_excel_sheet(asset_path, "Gas Network")

    electric_map: Dict[str, Mapping[str, object]] = {}
    for row in electric_rows:
        name = row.get("Name")
        if not name:
            continue
        key = _format_component_name(name)
        electric_map[key] = row

    gas_map: Dict[str, Mapping[str, object]] = {}
    for row in gas_rows:
        name = row.get("Name")
        if not name:
            continue
        key = _format_component_name(name)
        gas_map[key] = row
    return electric_map, gas_map


def _build_hazard_coordinates(data_dir: Path) -> Dict[str, Tuple[float, float]]:
    electric_map, gas_map = _load_asset_catalog(data_dir)
    hazard_coords: Dict[str, Tuple[float, float]] = {}
    coordinate_map: Dict[str, np.ndarray] = {}
    capacity_map: Dict[str, float] = {}

    def register(component: str, coord: np.ndarray, weight: float) -> None:
        coordinate_map[component] = coord
        capacity_map[component] = float(weight)
        hazard_coords[component] = (float(coord[0]), float(coord[1]))

    def extract(spec: AssetSpec) -> Tuple[np.ndarray, float]:
        if spec.source_type == "electric":
            row = electric_map.get(spec.source_key)
        else:
            row = gas_map.get(spec.source_key)
        if row is None:
            raise KeyError(f"Asset '{spec.source_key}' not found in {spec.source_type} catalog.")
        coord = np.asarray(_row_coordinate(row), dtype=float)
        capacity = _capacity_from_row(row)
        return coord, capacity

    # Renewable + thermal generation assets
    for spec in GENERATION_ASSETS + THERMAL_ASSETS:
        coord, capacity = extract(spec)
        register(spec.name, coord, capacity)

    # LNG terminal (fuel supply)
    lng_row = gas_map.get("Calcasieu_Pass_2")
    if lng_row is None:
        raise KeyError("'Calcasieu_Pass_2' entry missing from gas asset catalog.")
    lng_coord = np.asarray(_row_coordinate(lng_row), dtype=float)
    lng_capacity = _capacity_from_row(lng_row)
    register("Calcasieu_Pass_LNG", lng_coord, lng_capacity)

    # Compressors (one coordinate derived from fed generators)
    for name, feed_assets in COMPRESSOR_FEEDS.items():
        centroid = _weighted_centroid(feed_assets, coordinate_map, capacity_map)
        if centroid is None:
            centroid = lng_coord.copy()
        supply_capacity = sum(capacity_map.get(asset, 0.0) for asset in feed_assets)
        register(name, centroid, supply_capacity)

    # Substations (centroid of contributing generation assets)
    for name, generators in SUBSTATION_GENERATORS.items():
        centroid = _weighted_centroid(generators, coordinate_map, capacity_map)
        if centroid is None:
            centroid = lng_coord.copy()
        capacity = sum(capacity_map.get(gen, 0.0) for gen in generators)
        register(name, centroid, capacity)

    if not hazard_coords:
        raise RuntimeError("No hazard points derived from asset catalog.")
    return hazard_coords


def _sample_depths(
    dem_path: Path,
    ocean_coords: Tuple[float, float],
    hazard_coords: Mapping[str, Tuple[float, float]],
    levels: Sequence[float],
) -> Dict[str, List[float]]:
    pgf = pyGeoFlood(dem_path=str(dem_path))
    results: Dict[str, List[float]] = {key: [] for key in hazard_coords}

    for level in levels:
        pgf.c_hand(ocean_coords=ocean_coords, gage_el=float(level))
        with rio.open(pgf.coastal_inundation_path) as dataset:
            band = dataset.read(1, masked=True)
            height, width = band.shape
            for hazard, (east, north) in hazard_coords.items():
                value = float("nan")
                try:
                    row, col = dataset.index(float(east), float(north))
                    row = int(row)
                    col = int(col)
                except (IndexError, ValueError):
                    pass
                else:
                    if row < 0 or row >= height or col < 0 or col >= width:
                        pass
                    else:
                        sample = band[row, col]
                        if not np.ma.is_masked(sample):
                            value = float(np.asarray(sample))
                results[hazard].append(value)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hazard depth samples for network assets.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing asset catalog.")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM_PATH, help="DEM raster used by pyGeoFlood.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination CSV file.")
    parser.add_argument("--ocean-easting", type=float, default=DEFAULT_OCEAN_COORD[0], help="Easting of offshore forcing cell.")
    parser.add_argument("--ocean-northing", type=float, default=DEFAULT_OCEAN_COORD[1], help="Northing of offshore forcing cell.")
    parser.add_argument("--min-level", type=float, default=0.0, help="Minimum gauge elevation (m).")
    parser.add_argument("--max-level", type=float, default=8.0, help="Maximum gauge elevation (m).")
    parser.add_argument("--num-levels", type=int, default=100, help="Number of evenly spaced gauge levels.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_levels <= 1:
        raise ValueError("num-levels must be at least 2.")

    levels = np.linspace(float(args.min_level), float(args.max_level), num=int(args.num_levels))
    hazard_coords = _build_hazard_coordinates(Path(args.data_dir))
    results = _sample_depths(
        dem_path=Path(args.dem),
        ocean_coords=(float(args.ocean_easting), float(args.ocean_northing)),
        hazard_coords=hazard_coords,
        levels=levels,
    )


    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_hazards = sorted(results)
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + [f"{level:.6f}" for level in levels])
        for hazard in ordered_hazards:
            row_values: List[str] = []
            for value in results[hazard]:
                if value is None or not np.isfinite(value):
                    row_values.append("")
                else:
                    row_values.append(f"{value:.6f}")
            writer.writerow([hazard] + row_values)

    print("Resolved hazard points (UTM 15N):")
    for hazard in sorted(hazard_coords):
        east, north = hazard_coords[hazard]
        print(f"  {hazard}: ({east:.3f}, {north:.3f})")
    print(f"Saved samples to: {output_path}")


if __name__ == "__main__":
    main()
