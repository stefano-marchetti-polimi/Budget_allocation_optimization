"""Plot the dependency network of assets in the TrialEnv environment.

The relationships mirror the logical dependencies implemented in
``utils.environment.TrialEnv._compute_metrics`` where service
availability propagates across components such as generation, compressors,
substations, and the LNG terminal.
"""

from __future__ import annotations

from itertools import filterfalse
import math
import sys
from functools import lru_cache
from pathlib import Path
from tkinter import FALSE
from typing import Dict, Mapping, Sequence, Tuple
import warnings

try:
    import contextily as cx
except ImportError:  # pragma: no cover - optional dependency
    cx = None
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import rasterio as rio
from matplotlib import patheffects
from matplotlib.lines import Line2D
from rasterio.errors import RasterioIOError
from rasterio.plot import plotting_extent

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.environment import ASSET_COORD_CRS, _build_default_network

__all__ = ["plot_asset_dependency_network"]

# Presentation metadata for component categories.
CATEGORY_LABELS: Mapping[str, str] = {
    "renewable": "Renewable Generation",
    "thermal": "Thermal Generation",
    "lng": "Fuel Supply",
    "compressor": "Gas Compression (per plant)",
    "substation": "Electrical Distribution",
}

CATEGORY_COLORS: Mapping[str, str] = {
    "renewable": "#5ab4ac",
    "thermal": "#b8860b",
    "lng": "#6c93ff",
    "compressor": "#80b1d3",
    "substation": "#c7e9c0",
    "unclassified": "#bdbdbd",
}

DEFAULT_CATEGORY_LABEL = "Unclassified"
DEFAULT_DEM_PATH = REPO_ROOT / "data" / "houston_example_DEM_30m.tif"
DEFAULT_BACKGROUND_IMAGE = REPO_ROOT / "data" / "houston_example_DEM_30m_coastal_inundation.tif"
DEFAULT_IMAGE_DIR = REPO_ROOT / "Images"
DEFAULT_PRIMARY_FILENAME = "asset_dependency_map.png"


@lru_cache(maxsize=1)
def _network_snapshot() -> Dict[str, object]:
    network = _build_default_network(REPO_ROOT / "data")
    dependencies = {
        cfg.name: tuple(cfg.dependencies)
        for cfg in network.components
        if cfg.dependencies
    }
    compressor_set = {cfg.name for cfg in network.components if cfg.category == "compressor"}
    compressor_assignments: Dict[str, str] = {}
    edges: set[Tuple[str, str]] = set()
    for cfg in network.components:
        for dep in cfg.dependencies:
            edges.add((dep, cfg.name))
            if cfg.category == "thermal" and dep in compressor_set:
                compressor_assignments[dep] = cfg.name
        for source in cfg.generator_sources:
            edges.add((source, cfg.name))
    category_map = {cfg.name: cfg.category for cfg in network.components}
    coordinates = {
        cfg.name: tuple(cfg.coordinate)
        for cfg in network.components
        if cfg.coordinate is not None
    }
    components = {cfg.name: cfg for cfg in network.components}
    return {
        "dependencies": dependencies,
        "edges": tuple(sorted(edges)),
        "category_map": category_map,
        "coordinates": coordinates,
        "components": components,
        "compressor_people": network.compressor_people,
        "compressor_industrial": network.compressor_industrial,
        "compressor_residential": network.compressor_residential,
        "compressor_supply": network.compressor_supply,
        "substation_people": network.substation_people,
        "substation_industrial": network.substation_industrial,
        "substation_residential": network.substation_residential,
        "substation_supply": network.substation_supply,
        "county_coordinates": network.county_coordinates,
        "county_population": network.county_population,
        "asset_counties": network.asset_counties,
        "county_display_names": network.county_display_names,
        "compressor_assignments": compressor_assignments,
    }


def _dependency_map() -> Dict[str, Sequence[str]]:
    snapshot = _network_snapshot()
    return {asset: deps[:] for asset, deps in snapshot["dependencies"].items()}


def _edge_labels() -> Dict[Tuple[str, str], str]:
    return {}


def _collect_assets(dependency_map: Mapping[str, Sequence[str]]) -> Sequence[str]:
    snapshot = _network_snapshot()
    nodes = sorted(snapshot["components"].keys())
    return nodes


def _edge_list() -> Sequence[Tuple[str, str]]:
    snapshot = _network_snapshot()
    return list(snapshot["edges"])


def _circular_layout(nodes: Sequence[str], radius: float = 2.5) -> Dict[str, Tuple[float, float]]:
    """Fallback layout that arranges nodes on a circle."""
    positions: Dict[str, Tuple[float, float]] = {}
    n = max(len(nodes), 1)
    for idx, node in enumerate(nodes):
        angle = 2.0 * math.pi * (idx / n)
        positions[node] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions


def _compute_positions(nodes: Sequence[str], coord_map: Mapping[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Derive 2-D positions from geographic coordinates for schematic plotting."""
    available = {node: coord_map[node] for node in nodes if node in coord_map}
    if not available:
        return _circular_layout(nodes)
    eastings = np.array([coord[0] for coord in available.values()], dtype=np.float32)
    northings = np.array([coord[1] for coord in available.values()], dtype=np.float32)
    min_e, max_e = float(eastings.min()), float(eastings.max())
    min_n, max_n = float(northings.min()), float(northings.max())
    width = max(max_e - min_e, 1.0)
    height = max(max_n - min_n, 1.0)
    positions: Dict[str, Tuple[float, float]] = {}
    for node in nodes:
        coord = coord_map.get(node)
        if coord is None:
            continue
        x_norm = ((coord[0] - min_e) / width) * 2.0 - 1.0
        y_norm = ((coord[1] - min_n) / height) * 2.0 - 1.0
        positions[node] = (x_norm, y_norm)
    missing = [node for node in nodes if node not in positions]
    if missing:
        positions.update(_circular_layout(missing, radius=1.3))
    return positions


def _asset_category(asset: str) -> str:
    snapshot = _network_snapshot()
    return snapshot["category_map"].get(asset, "unclassified")


def _categorize_asset(asset: str) -> str:
    """Return the display category label for the requested asset."""
    category = _asset_category(asset)
    return CATEGORY_LABELS.get(category, DEFAULT_CATEGORY_LABEL)


def _asset_color(asset: str) -> str:
    category = _asset_category(asset)
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["unclassified"])


def _asset_coordinate_map() -> Dict[str, Tuple[float, float]]:
    """Return a mapping of asset -> (Easting, Northing) coordinates."""
    snapshot = _network_snapshot()
    return dict(snapshot["coordinates"])


def _county_coordinate_map() -> Dict[str, Tuple[float, float]]:
    snapshot = _network_snapshot()
    return dict(snapshot.get("county_coordinates", {}))


def _asset_county_links() -> Dict[str, Tuple[str, ...]]:
    snapshot = _network_snapshot()
    return {asset: tuple(counties) for asset, counties in snapshot.get("asset_counties", {}).items()}


def _county_display_name(county_key: str) -> str:
    snapshot = _network_snapshot()
    display_map: Mapping[str, str] = snapshot.get("county_display_names", {})
    return display_map.get(county_key, county_key.replace("_", " "))


def _map_extent(coords: Mapping[str, Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Compute a padded map extent that comfortably frames the provided coordinates."""
    if not coords:
        raise ValueError("Cannot determine extent without asset coordinates.")
    eastings = [pt[0] for pt in coords.values()]
    northings = [pt[1] for pt in coords.values()]
    min_e, max_e = min(eastings), max(eastings)
    min_n, max_n = min(northings), max(northings)
    width = max(max_e - min_e, 1.0)
    height = max(max_n - min_n, 1.0)
    pad_e = max(width * 0.12, 15_000.0)
    pad_n = max(height * 0.12, 15_000.0)
    return (
        min_e - pad_e,
        max_e + pad_e,
        min_n - pad_n,
        max_n + pad_n,
    )


def _load_dem(dem_path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM raster not found at {dem_path}.")
    with rio.open(dem_path) as ds:
        dem = ds.read(1).astype(np.float32)
        profile = ds.profile
    nodata = profile.get("nodata")
    if nodata is not None:
        dem[dem == nodata] = np.nan
    return dem, profile


def _add_dem_basemap(
    ax: plt.Axes,
    coord_map: Mapping[str, Tuple[float, float]],
    *,
    dem_path: Path | None = DEFAULT_DEM_PATH,
    background_image: Path | None = DEFAULT_BACKGROUND_IMAGE,
) -> Tuple[Tuple[float, float, float, float], plt.AxesImage | None]:
    """Draw the requested background (image and/or DEM) and frame the axes."""
    extent = _map_extent(coord_map)
    xmin, xmax, ymin, ymax = extent
    crs_to_use = ASSET_COORD_CRS
    dem_image: plt.AxesImage | None = None
    background_drawn = False

    if background_image is not None:
        try:
            bg_extent, _, _ = _add_image_background(
                ax,
                coord_map,
                image_path=Path(background_image),
                cmap=None,
                alpha=0.95,
                mask_zero=False,
            )
            extent = bg_extent
            xmin, xmax, ymin, ymax = extent
            background_drawn = True
        except FileNotFoundError as exc:
            warnings.warn(
                f"Background image not found ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to load background image {background_image} ({exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
    if dem_path is not None:
        try:
            dem, profile = _load_dem(dem_path)
            dem_extent = plotting_extent(dem, profile["transform"])
            finite_vals = dem[np.isfinite(dem)]
            if finite_vals.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.nanpercentile(dem, 5))
                vmax = float(np.nanpercentile(dem, 95))
            dem_image = ax.imshow(
                dem,
                extent=dem_extent,
                cmap="terrain",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                alpha=0.8,
                zorder=1,
            )
            crs_val = profile.get("crs")
            if crs_val is not None:
                crs_to_use = crs_val
        except Exception as exc:
            warnings.warn(
                f"Failed to load DEM {dem_path} ({exc}). Proceeding with basemap only.",
                RuntimeWarning,
                stacklevel=2,
            )
    if not background_drawn and cx is not None:
        try:
            cx.add_basemap(
                ax,
                crs=str(crs_to_use),
                source=cx.providers.Esri.WorldImagery,
                attribution_size=6,
                zoom=None,
                zorder=0,
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to retrieve satellite basemap ({exc}). Using plain background instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            ax.set_facecolor("#dcdcdc")
    elif not background_drawn:
        warnings.warn(
            "contextily not available; using DEM-only background.",
            RuntimeWarning,
            stacklevel=2,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    return extent, dem_image


def _load_georeferenced_image(image_path: Path) -> Tuple[np.ndarray, Tuple[float, float, float, float], Dict[str, object]] | None:
    """Return pixel data and extent if the path references a georeferenced raster."""
    try:
        with rio.open(image_path) as dataset:
            data = dataset.read(out_dtype=np.float32)
            profile = dataset.profile
            transform = profile.get("transform")
            if transform is None:
                return None
    except (RasterioIOError, ValueError):
        return None

    # Move band axis to the end for RGB/A rasters
    processed: np.ndarray
    if data.ndim == 3:
        if data.shape[0] == 1:
            processed = data[0]
        elif data.shape[0] in (3, 4):
            processed = np.moveaxis(data, 0, -1)
        else:
            processed = data[0]
    else:
        processed = data

    nodata = profile.get("nodata")
    if nodata is not None:
        processed = np.where(processed == nodata, np.nan, processed)

    extent = plotting_extent(data[0], transform)
    return processed, extent, profile


def _add_image_background(
    ax: plt.Axes,
    coord_map: Mapping[str, Tuple[float, float]],
    *,
    image_path: Path,
    cmap: str | None = None,
    alpha: float = 0.95,
    mask_zero: bool = True,
) -> Tuple[Tuple[float, float, float, float], plt.AxesImage | None, Dict[str, object] | None]:
    """Use a pre-rendered image as the basemap background."""
    if not image_path.exists():
        raise FileNotFoundError(f"Background image not found at {image_path}.")

    georas = _load_georeferenced_image(image_path)
    if georas is not None:
        image_data, extent, profile = georas
        xmin, xmax, ymin, ymax = extent

        if mask_zero and image_data.ndim == 2:
            image_data = image_data.copy()
            image_data[np.isclose(image_data, 0.0)] = np.nan

        artist = ax.imshow(
            image_data,
            extent=extent,
            zorder=0,
            cmap=None if image_data.ndim == 3 else (cmap or "viridis"),
            alpha=alpha,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        return extent, artist, profile

    warnings.warn(
        f"Background image {image_path} does not contain geospatial metadata; aligning using asset extent."
        " The overlay may be misaligned. Prefer supplying a GeoTIFF.",
        RuntimeWarning,
        stacklevel=2,
    )

    extent = _map_extent(coord_map)
    img = mpimg.imread(image_path)
    xmin, xmax, ymin, ymax = extent
    artist = ax.imshow(img, extent=extent, zorder=0, alpha=alpha)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    return extent, artist, None


def _draw_geographic_network(
    ax: plt.Axes,
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    edge_labels: Mapping[Tuple[str, str], str],
    coord_map: Mapping[str, Tuple[float, float]],
    *,
    edge_color: str = "#4a4a4a",
    edge_linewidth: float = 1.4,
    edge_alpha: float = 1.0,
    label_facecolor: str = "white",
    label_alpha: float = 0.7,
    label_text_color: str = "black",
    node_label_color: str = "black",
    colored_arrows: bool = False,
    arrow_outline_color: str = "#1f1f1f",
    label_offset: float = 2000.0,
    label_fontsize: float = 7.5,
    node_size: float = 140.0,
) -> Dict[str, Line2D]:
    """Draw dependency edges and nodes using real-world coordinates."""
    missing_assets: set[str] = set()
    legend_handles: Dict[str, Line2D] = {}

    for start, end in edges:
        start_coord = coord_map.get(start)
        end_coord = coord_map.get(end)
        if start_coord is None or end_coord is None:
            if start_coord is None:
                missing_assets.add(start)
            if end_coord is None:
                missing_assets.add(end)
            continue
        start_color = edge_color
        if colored_arrows:
            arrow_kwargs = dict(
                arrowstyle="-|>",
                linewidth=edge_linewidth,
                shrinkA=4,
                shrinkB=10,
                mutation_scale=12,
                alpha=edge_alpha,
                color="#ffffff",
                joinstyle="miter",
            )
        else:
            arrow_kwargs = dict(
                arrowstyle="-|>",
                linewidth=edge_linewidth,
                shrinkA=4,
                shrinkB=10,
                mutation_scale=12,
                alpha=edge_alpha,
                color=edge_color,
                joinstyle="miter",
            )
        ax.annotate(
            "",
            xy=end_coord,
            xytext=start_coord,
            arrowprops=arrow_kwargs,
            zorder=2,
        )
        label = edge_labels.get((start, end))
        if label:
            mid_x = (start_coord[0] + end_coord[0]) / 2.0
            mid_y = (start_coord[1] + end_coord[1]) / 2.0
            dx = end_coord[0] - start_coord[0]
            dy = end_coord[1] - start_coord[1]
            length = math.hypot(dx, dy)
            if length > 0:
                offset_scale = 0.004 * length
                off_x = -dy / length * offset_scale
                off_y = dx / length * offset_scale
                mid_x += off_x
                mid_y += off_y
            ax.text(
                mid_x,
                mid_y,
                label,
                fontsize=8,
                ha="center",
                va="center",
                color=label_text_color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=label_facecolor, alpha=label_alpha),
                zorder=3,
            )

    for node in nodes:
        coord = coord_map.get(node)
        if coord is None:
            missing_assets.add(node)
            continue
        asset_type = _categorize_asset(node)
        color = _asset_color(node)
        ax.scatter(
            [coord[0]],
            [coord[1]],
            s=node_size,
            color=color,
            edgecolors="#0f0f0f",
            linewidths=1.1,
            zorder=4,
        )

        if label_offset > 0.0:
            angle = (abs(hash(node)) % 3600) / 3600.0 * 2.0 * math.pi
            dx = label_offset * math.cos(angle)
            dy = label_offset * math.sin(angle)
        else:
            dx = dy = 0.0
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        text = ax.text(
            coord[0] + dx,
            coord[1] + dy,
            node,
            fontsize=label_fontsize,
            ha=ha,
            va=va,
            color=node_label_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.78, linewidth=0.0),
            zorder=5,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=2.6, foreground="white")])
        if asset_type not in legend_handles:
            legend_handles[asset_type] = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="#1f1f1f",
                markersize=9,
                linewidth=0.0,
            )

    if missing_assets:
        warnings.warn(
            f"No geographic coordinates provided for assets: {sorted(missing_assets)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return legend_handles


def _draw_network_overlay(
    ax: plt.Axes,
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    edge_labels: Mapping[Tuple[str, str], str],
    coord_map: Mapping[str, Tuple[float, float]],
) -> set[str]:
    """Draw dependency edges, nodes, and labels onto an existing axis."""
    missing_assets: set[str] = set()

    for start, end in edges:
        start_coord = coord_map.get(start)
        end_coord = coord_map.get(end)
        if start_coord is None or end_coord is None:
            if start_coord is None:
                missing_assets.add(start)
            if end_coord is None:
                missing_assets.add(end)
            continue
        ax.annotate(
            "",
            xy=end_coord,
            xytext=start_coord,
            arrowprops=dict(
                color="#4a4a4a",
                linewidth=1.4,
                shrinkA=6,
                shrinkB=20,
                mutation_scale=4,
            ),
            zorder=2,
        )
        label = edge_labels.get((start, end))
        if label:
            mid_x = (start_coord[0] + end_coord[0]) / 2.0
            mid_y = (start_coord[1] + end_coord[1]) / 2.0
            dx = end_coord[0] - start_coord[0]
            dy = end_coord[1] - start_coord[1]
            length = math.hypot(dx, dy)
            if length > 0:
                offset_scale = 0.004 * length
                off_x = -dy / length * offset_scale
                off_y = dx / length * offset_scale
                mid_x += off_x
                mid_y += off_y
            ax.text(
                mid_x,
                mid_y,
                label,
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                zorder=3,
            )

    legend_handles: Dict[str, Line2D] = {}
    for node in nodes:
        coord = coord_map.get(node)
        if coord is None:
            missing_assets.add(node)
            continue
        asset_type = _categorize_asset(node)
        color = _asset_color(node)
        ax.scatter(
            [coord[0]],
            [coord[1]],
            s=120,
            color=color,
            edgecolors="#1f1f1f",
            linewidths=0.9,
            zorder=4,
        )
        text = ax.text(
            coord[0],
            coord[1],
            node,
            fontsize=9,
            ha="left",
            va="bottom",
            color="black",
            zorder=5,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=2.6, foreground="white")])
        if asset_type not in legend_handles:
            legend_handles[asset_type] = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="#1f1f1f",
                markersize=9,
                linewidth=0.0,
            )

    if legend_handles:
        ax.legend(
            legend_handles.values(),
            legend_handles.keys(),
            title="Asset Type",
            loc="upper left",
        )
    return missing_assets


def _draw_county_connections(
    ax: plt.Axes,
    nodes: Sequence[str],
    coord_map: Mapping[str, Tuple[float, float]],
    *,
    county_positions: Mapping[str, Tuple[float, float]] | None = None,
    connection_color: str = "#6a51a3",
    connection_linestyle: str = "-",
    connection_alpha: float = 0.55,
    connection_linewidth: float = 1.2,
    county_marker: str = "s",
    county_color: str = "#feb24c",
    county_edgecolor: str = "#6a51a3",
    county_marker_size: float = 88.0,
    county_label_fontsize: float = 7.0,
    county_label_color: str = "#202020",
    county_label_offset: float = 2400.0,
) -> Dict[str, Line2D]:
    """Overlay county centroids and draw connections to served assets."""
    county_coords = dict(county_positions) if county_positions is not None else _county_coordinate_map()
    asset_counties = _asset_county_links()
    connection_color = "#1f1f1f"
    if not county_coords or not asset_counties:
        return {}

    counties_to_plot: Dict[str, Tuple[float, float]] = {}
    for asset in nodes:
        for county_key in asset_counties.get(asset, ()):  # type: ignore[arg-type]
            coord = county_coords.get(county_key)
            if coord is not None:
                counties_to_plot[county_key] = coord

    if not counties_to_plot:
        return {}

    # Draw connections before markers so nodes appear on top
    for asset in nodes:
        asset_coord = coord_map.get(asset)
        if asset_coord is None:
            continue
        for county_key in asset_counties.get(asset, ()):  # type: ignore[arg-type]
            county_coord = counties_to_plot.get(county_key)
            if county_coord is None:
                continue
            ax.annotate(
                "",
                xy=county_coord,
                xytext=asset_coord,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=connection_color,
                    linewidth=connection_linewidth,
                    alpha=connection_alpha,
                    shrinkA=6,
                    shrinkB=14,
                    mutation_scale=10,
                    linestyle=connection_linestyle,
                ),
                zorder=2.2,
            )

    for county_key, coord in counties_to_plot.items():
        ax.scatter(
            [coord[0]],
            [coord[1]],
            marker=county_marker,
            s=county_marker_size,
            color=county_color,
            edgecolors=county_edgecolor,
            linewidths=0.9,
            zorder=3.6,
        )
        if county_label_offset > 0.0:
            angle = (abs(hash((county_key, "county"))) % 3600) / 3600.0 * 2.0 * math.pi
            dx = county_label_offset * math.cos(angle)
            dy = county_label_offset * math.sin(angle)
        else:
            dx = dy = 0.0
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        ax.text(
            coord[0] + dx,
            coord[1] + dy,
            _county_display_name(county_key),
            fontsize=county_label_fontsize,
            ha=ha,
            va=va,
            color=county_label_color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", alpha=0.72, linewidth=0.0),
            zorder=3.8,
        )

    return {
        "County": Line2D(
            [0],
            [0],
            marker=county_marker,
            color="w",
            markerfacecolor=county_color,
            markeredgecolor=county_edgecolor,
            markersize=6,
            linewidth=0.0,
        )
    }


def plot_asset_dependency_network(
    *,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    subplot_title: str | None = None,
    include_map: bool = True,
    dem_path: str | Path | None = DEFAULT_DEM_PATH,
    background_image: str | Path | None = DEFAULT_BACKGROUND_IMAGE,
    colored_arrows: bool = False,
    plot_loads: bool = True,
    population_heatmap: bool = False,
    industrial_heatmap: bool = False,
    background_cmap: str | None = "Blues",
    background_alpha: float = 0.9,
    background_colorbar_label: str | None = "Inundation depth [m]",
    background_zero_transparent: bool = True,
    show_counties: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot asset dependencies and optionally overlay geographic locations on a map.

    Parameters
    ----------
    background_image:
        Path to a background raster (GeoTIFF recommended) used for the secondary map figure.
        Images without geospatial metadata fall back to the asset extent and may appear misaligned.
    background_cmap:
        Colormap applied to single-band rasters when rendering ``background_image``.
    background_alpha:
        Opacity applied to the background raster.
    background_colorbar_label:
        Label for the background colorbar; set to ``None`` to disable.
    background_zero_transparent:
        Treat zero-valued pixels in single-band backgrounds as transparent to expose the basemap.
    show_counties:
        Draw county centroids and highlight their service connections to substations and compressors.
    """
    dependency_map = _dependency_map()
    nodes = _collect_assets(dependency_map)
    edges = list(_edge_list())
    edge_labels = _edge_labels()
    coord_map_full = _asset_coordinate_map()
    snapshot_full = _network_snapshot()

    def _create_load_figure() -> plt.Figure | None:
        components_info = snapshot_full["components"].values()
        compressor_components = [cfg for cfg in components_info if cfg.category == "compressor"]
        substation_components = [cfg for cfg in components_info if cfg.category == "substation"]
        if not compressor_components and not substation_components:
            return None

        fig_load, axes = plt.subplots(1, 2, figsize=(12, 5))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(1, -1)

        def _plot_panel(ax: plt.Axes, comps, supply_map, residential_map, industrial_map, title: str) -> None:
            if not comps:
                ax.axis("off")
                ax.set_title(f"{title}\n(no data)")
                return
            names = [cfg.name for cfg in comps]
            idx = np.arange(len(names))
            width = 0.35
            residential_vals = np.array([residential_map.get(name, 0.0) for name in names]) / 1e3
            industrial_vals = np.array([industrial_map.get(name, 0.0) for name in names]) / 1e3
            supply_vals = np.array([supply_map.get(name, 0.0) for name in names])
            ax.bar(idx - width / 2, residential_vals, width=width, label="Residential (×10³)", color="#74add1")
            ax.bar(idx + width / 2, industrial_vals, width=width, label="Industrial (×10³)", color="#fdae61")
            for i, supply in enumerate(supply_vals):
                ax.text(
                    idx[i],
                    max(residential_vals[i], industrial_vals[i]) + 0.05,
                    f"{supply:.0f} MW",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.set_xticks(idx)
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_ylabel("Weighted load (thousands)")
            ax.set_title(title)
            ax.legend(fontsize=8)

        _plot_panel(
            axes[0, 0],
            compressor_components,
            snapshot_full.get("compressor_supply", {}),
            snapshot_full.get("compressor_residential", {}),
            snapshot_full.get("compressor_industrial", {}),
            "Compressor loads",
        )
        _plot_panel(
            axes[0, 1],
            substation_components,
            snapshot_full.get("substation_supply", {}),
            snapshot_full.get("substation_residential", {}),
            snapshot_full.get("substation_industrial", {}),
            "Substation loads",
        )
        fig_load.tight_layout()
        return fig_load

    def _create_heatmap_figure(weight_attr: str, title: str, cmap: str) -> plt.Figure | None:
        coords = []
        weights = []
        for name, cfg in snapshot_full["components"].items():
            if cfg.category not in {"compressor", "substation"}:
                continue
            coord = coord_map_full.get(name)
            if coord is None:
                continue
            weight = getattr(cfg, weight_attr, 0.0)
            if not weight:
                continue
            coords.append(coord)
            weights.append(float(weight))
        if not coords:
            return None
        extent = _map_extent(coord_map_full)
        xmin, xmax, ymin, ymax = extent
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
        fig_heat, ax_heat = plt.subplots(figsize=(10, 10))
        hb = ax_heat.hexbin(
            xs,
            ys,
            C=np.array(weights, dtype=np.float64),
            gridsize=40,
            extent=extent,
            cmap=cmap,
            reduce_C_function=np.sum,
            mincnt=1,
            linewidths=0.0,
            alpha=0.85,
        )
        ax_heat.set_xlim(xmin, xmax)
        ax_heat.set_ylim(ymin, ymax)
        ax_heat.set_xlabel("Easting [m]")
        ax_heat.set_ylabel("Northing [m]")
        ax_heat.set_title(title)
        fig_heat.colorbar(hb, ax=ax_heat, label="Weighted density")
        _draw_geographic_network(
            ax_heat,
            nodes,
            edges,
            edge_labels,
            coord_map_full,
            colored_arrows=colored_arrows,
            edge_linewidth=1.0,
            edge_color="#2b2b2b",
            edge_alpha=0.6,
        )
        fig_heat.tight_layout()
        return fig_heat

    secondary_fig: plt.Figure | None = None

    if ax is None:
        if include_map:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    secondary_fig: plt.Figure | None = None
    load_fig: plt.Figure | None = None
    residential_fig: plt.Figure | None = None
    industrial_fig: plt.Figure | None = None

    if include_map:
        relevant_coords = {node: coord_map_full[node] for node in nodes if node in coord_map_full}
        county_coords_full = _county_coordinate_map()
        asset_counties = _asset_county_links()
        if show_counties:
            for asset in nodes:
                for county_key in asset_counties.get(asset, ()):  # type: ignore[arg-type]
                    county_coord = county_coords_full.get(county_key)
                    if county_coord is not None:
                        relevant_coords.setdefault(f"county::{county_key}", county_coord)
        if not relevant_coords:
            raise ValueError("No geographic coordinates available for the requested assets.")
        dem_param = Path(dem_path) if dem_path is not None else None
        _, dem_image = _add_dem_basemap(ax, relevant_coords, dem_path=dem_param)
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")
        if subplot_title:
            ax.set_title(subplot_title, fontsize=13)
        else:
            ax.set_title("Asset Dependency Map (UTM 15N)")
        if dem_image is not None:
            cbar = fig.colorbar(dem_image, ax=ax, shrink=0.78, label="Elevation [m]")
            cbar.ax.tick_params(labelsize=9)

        coord_map = coord_map_full
        if plot_loads:
            load_fig = _create_load_figure()
        if population_heatmap:
            residential_fig = _create_heatmap_figure("residential_load", "Residential load density", "YlGnBu")
        if industrial_heatmap:
            industrial_fig = _create_heatmap_figure("industrial_load", "Industrial load density", "YlOrRd")
        legend_handles = _draw_geographic_network(
            ax,
            nodes,
            edges,
            edge_labels,
            coord_map,
            colored_arrows=colored_arrows,
            arrow_outline_color="#101010",
        )

        if show_counties:
            county_handles = _draw_county_connections(
                ax,
                nodes,
                coord_map,
            )
            if county_handles:
                legend_handles.update(county_handles)

        if legend_handles:
            ax.legend(
                legend_handles.values(),
                legend_handles.keys(),
                title="Asset Type",
                loc="upper left",
            )

        if background_image is not None:
            image_param = Path(background_image)
            try:
                fig_img, ax_img = plt.subplots(figsize=(10, 10))
                _, img_artist, img_profile = _add_image_background(
                    ax_img,
                    relevant_coords,
                    image_path=image_param,
                    cmap=background_cmap,
                    alpha=background_alpha,
                    mask_zero=background_zero_transparent,
                )
                ax_img.set_xlabel("Easting [m]")
                ax_img.set_ylabel("Northing [m]")
                if subplot_title:
                    ax_img.set_title(f"{subplot_title} (image background)", fontsize=13)
                else:
                    ax_img.set_title("Asset Dependency Map (image background)")

                if (
                    background_colorbar_label
                    and img_artist is not None
                    and img_profile is not None
                    and getattr(img_artist.get_array(), "ndim", 0) == 2
                ):
                    fig_img.colorbar(
                        img_artist,
                        ax=ax_img,
                        shrink=0.8,
                        label=background_colorbar_label,
                    )

                legend_img = _draw_geographic_network(
                    ax_img,
                    nodes,
                    edges,
                    edge_labels,
                    coord_map,
                    edge_color="#f5f5f5",
                    edge_linewidth=1.8,
                    edge_alpha=0.98,
                    label_facecolor="#1f1f1f",
                    label_alpha=0.75,
                    label_text_color="white",
                    node_label_color="black",
                    colored_arrows=colored_arrows,
                    arrow_outline_color="#f5f5f5",
                )
                county_img_handles: Dict[str, Line2D] = {}
                if show_counties:
                    county_img_handles = _draw_county_connections(
                        ax_img,
                        nodes,
                        coord_map,
                    )
                combined_handles = dict(legend_img)
                if county_img_handles:
                    combined_handles.update(county_img_handles)
                if combined_handles:
                    ax_img.legend(
                        combined_handles.values(),
                        combined_handles.keys(),
                        title="Asset Type",
                        loc="upper left",
                    )
                fig_img.tight_layout()
                secondary_fig = fig_img
            except Exception as exc:
                warnings.warn(
                    f"Could not render image-background map ({exc}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                secondary_fig = None
    else:
        county_coords_full = _county_coordinate_map()
        asset_counties = _asset_county_links() if show_counties else {}
        county_nodes: set[str] = set()
        if show_counties:
            for asset in nodes:
                for county_key in asset_counties.get(asset, ()):  # type: ignore[arg-type]
                    if county_key in county_coords_full:
                        county_nodes.add(county_key)

        layout_nodes = list(nodes)
        layout_coord_map = dict(coord_map_full)
        if show_counties:
            for county_key in county_nodes:
                layout_nodes.append(county_key)
                layout_coord_map[county_key] = county_coords_full[county_key]

        positions = _compute_positions(layout_nodes, layout_coord_map)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0.2)
        if subplot_title:
            ax.set_title(subplot_title, fontsize=13)
        if plot_loads:
            load_fig = _create_load_figure()
        if population_heatmap:
            residential_fig = _create_heatmap_figure("residential_load", "Residential load density", "YlGnBu")
        if industrial_heatmap:
            industrial_fig = _create_heatmap_figure("industrial_load", "Industrial load density", "YlOrRd")

        for start, end in edges:
            x0, y0 = positions[start]
            x1, y1 = positions[end]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#4a4a4a",
                    linewidth=1.4,
                    shrinkA=18,
                    shrinkB=18,
                    mutation_scale=12,
                ),
            )

            label = edge_labels.get((start, end))
            if label:
                mid_x = (x0 + x1) / 2.0
                mid_y = (y0 + y1) / 2.0
                dx = x1 - x0
                dy = y1 - y0
                length = math.hypot(dx, dy)
                if length > 0:
                    offset_scale = 0.4
                    off_x = -dy / length * offset_scale
                    off_y = dx / length * offset_scale
                    mid_x += off_x
                    mid_y += off_y
                ax.text(
                    mid_x,
                    mid_y,
                    label,
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

        legend_handles = {}
        for node in nodes:
            x, y = positions[node]
            asset_type = _categorize_asset(node)
            color = _asset_color(node)
            ax.scatter(
                [x],
                [y],
                s=440,
                color=color,
                edgecolors="#1f1f1f",
                linewidths=1.2,
                zorder=3,
            )
            ax.text(
                x,
                y,
                node,
                fontsize=11,
                ha="center",
                va="center",
                color="black",
                zorder=4,
            )
            if asset_type not in legend_handles:
                legend_handles[asset_type] = Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="#1f1f1f",
                    markersize=9,
                    linewidth=0.0,
                )
        if show_counties and county_nodes:
            asset_position_map = {name: positions[name] for name in nodes}
            county_position_map = {name: positions[name] for name in county_nodes if name in positions}
            county_handles = _draw_county_connections(
                ax,
                nodes,
                asset_position_map,
                county_positions=county_position_map,
            )
            if county_handles:
                legend_handles.update(county_handles)
        if legend_handles:
            ax.legend(
                legend_handles.values(),
                legend_handles.keys(),
                title="Asset Type",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )

    fig.tight_layout()
    if include_map:
        fig._secondary_map_figure = secondary_fig
        if secondary_fig is not None:
            secondary_fig._primary_map_figure = fig
    if plot_loads:
        fig._load_distribution_figure = load_fig
        if load_fig is not None:
            load_fig._primary_network_figure = fig
    if population_heatmap:
        fig._residential_heatmap_figure = residential_fig
        fig._population_heatmap_figure = residential_fig
        if residential_fig is not None:
            residential_fig._primary_network_figure = fig
    if industrial_heatmap:
        fig._industrial_heatmap_figure = industrial_fig
        if industrial_fig is not None:
            industrial_fig._primary_network_figure = fig
    if save_path is not None:
        save_path_obj = Path(save_path)
    else:
        save_path_obj = DEFAULT_IMAGE_DIR / DEFAULT_PRIMARY_FILENAME

    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    arrow_suffix = "_colored" if colored_arrows else ""
    if save_path_obj.suffix:
        primary_path = save_path_obj.with_name(f"{save_path_obj.stem}{arrow_suffix}{save_path_obj.suffix}")
    else:
        primary_path = Path(f"{save_path_obj}{arrow_suffix}")

    fig.savefig(primary_path, bbox_inches="tight")
    secondary_path: Path | None = None
    if include_map and secondary_fig is not None:
        if primary_path.suffix:
            secondary_path = primary_path.with_name(f"{primary_path.stem}_image{primary_path.suffix}")
        else:
            secondary_path = Path(f"{primary_path}_image")
        secondary_fig.savefig(secondary_path, bbox_inches="tight")
    if plot_loads and load_fig is not None:
        if primary_path.suffix:
            loads_path = primary_path.with_name(f"{primary_path.stem}_loads{primary_path.suffix}")
        else:
            loads_path = Path(f"{primary_path}_loads")
        load_fig.savefig(loads_path, bbox_inches="tight")
    if population_heatmap and residential_fig is not None:
        if primary_path.suffix:
            pop_path = primary_path.with_name(f"{primary_path.stem}_residential{primary_path.suffix}")
        else:
            pop_path = Path(f"{primary_path}_residential")
        residential_fig.savefig(pop_path, bbox_inches="tight")
    if industrial_heatmap and industrial_fig is not None:
        if primary_path.suffix:
            ind_path = primary_path.with_name(f"{primary_path.stem}_industrial{primary_path.suffix}")
        else:
            ind_path = Path(f"{primary_path}_industrial")
        industrial_fig.savefig(ind_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    plot_asset_dependency_network(
    include_map=True,
    population_heatmap=False,
    industrial_heatmap=False,
    save_path="Images/network.png",
)
