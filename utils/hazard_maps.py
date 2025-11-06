from __future__ import annotations

import argparse
import sys
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio as rio
from pygeoflood import pyGeoFlood

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.environment import _build_default_network

# Default paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_DEM_PATH = DEFAULT_DATA_DIR / "houston_example_DEM_30m.tif"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "coastal_inundation_samples.csv"

# Representative offshore coordinate for boundary forcing
DEFAULT_OCEAN_COORD = (317_540.0, 3_272_260.0)


def _build_hazard_coordinates(data_dir: Path) -> Dict[str, Tuple[float, float]]:
    network = _build_default_network(data_dir)
    hazard_coords: Dict[str, Tuple[float, float]] = {}
    for component in network.components:
        coord = component.coordinate
        if coord is None:
            continue
        key = component.hazard_key
        candidate = (float(coord[0]), float(coord[1]))
        if key in hazard_coords:
            existing = hazard_coords[key]
            if not np.allclose(existing, candidate, atol=1e-3):
                raise ValueError(
                    f"Hazard key '{key}' has conflicting coordinates: "
                    f"{existing} vs {candidate}."
                )
            continue
        hazard_coords[key] = candidate
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
    parser.add_argument("--max-level", type=float, default=10.0, help="Maximum gauge elevation (m).")
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
