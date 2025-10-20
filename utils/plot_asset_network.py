"""Plot the dependency network of assets in the TrialEnv environment.

The relationships mirror the logical dependencies implemented in
``utils.environment_placeholder.TrialEnv._compute_metrics`` where service
availability propagates across assets such as PV, substations, compressors,
the thermal unit, and the LNG terminal.
"""

from __future__ import annotations

import math
from pathlib import Path
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
from rasterio.plot import plotting_extent

__all__ = ["plot_asset_dependency_network"]

# Directed dependencies: each asset lists the upstream components it requires
# to be operational. Edges are drawn from dependency -> dependent.
BASE_ASSET_DEPENDENCIES: Mapping[str, Sequence[str]] = {
    "Substation2": ("PV",),
    "Compressor1": ("LNG", "Substation1", "Substation2"),
    "Compressor2": ("LNG", "Substation1", "Substation2"),
    "Compressor3": ("LNG", "Substation1", "Substation2"),
    "ThermalUnit": ("Compressor3",),
    "Substation1": ("ThermalUnit",),
}

# Asset categories drive node styling on the plot.
ASSET_TYPE_MAP: Mapping[str, str] = {
    "PV": "Renewable Generation",
    "ThermalUnit": "Thermal Generation",
    "LNG": "Fuel Supply",
    "Substation1": "Electrical Distribution",
    "Substation2": "Electrical Distribution",
    "Compressor1": "Gas Compression",
    "Compressor2": "Gas Compression",
    "Compressor3": "Gas Compression",
}

ASSET_TYPE_COLORS: Mapping[str, str] = {
    "Renewable Generation": "#5ab4ac",
    "Thermal Generation": "#d8b365",
    "Fuel Supply": "#fdb863",
    "Electrical Distribution": "#c7e9c0",
    "Gas Compression": "#80b1d3",
    "Unclassified": "#bdbdbd",
}

# Optional edge labels to highlight the role of key connections.
BASE_EDGE_LABELS: Dict[Tuple[str, str], str] = {}

# Manually chosen coordinates to keep the diagram easy to read.
BASE_MANUAL_POSITIONS: Mapping[str, Tuple[float, float]] = {
    # Anchor sources far apart so downstream edges do not stack.
    "PV": (-10.0, 4.5),
    "LNG": (12.0, 6.0),
    # Downstream assets arranged to follow the logical flow left-to-right.
    "Substation2": (-5.0, 3.2),
    "Compressor1": (1.0, 7.5),
    "Compressor2": (1.0, 2.2),
    "Compressor3": (1.0, -3.0),
    "ThermalUnit": (3.5, -5.5),
    "Substation1": (10.0, 0.0),
}

# Real-world coordinates for assets as used in the hazard sampling workflow.
BASE_ASSET_COORDINATES: Mapping[str, Tuple[float, float]] = {
    "PV": (415337.819, 3321791.508),
    "Substation1": (471235.458, 3349033.547),
    "Substation2": (313798.678, 3292636.851),
    "Compressor1": (429265.789, 3347691.776),
    "Compressor2": (295005.562, 3302607.527),
    "Compressor3": (307580.921, 3264174.781),
    "ThermalUnit": (312418.912, 3251429.175),
    "LNG": (265455.097, 3209409.708),
}

ASSET_COORD_CRS = "EPSG:32615"
DEFAULT_DEM_PATH = Path("data/houston_example_DEM_30m.tif")
DEFAULT_BACKGROUND_IMAGE = Path("data/inun_zero_depth.png")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = REPO_ROOT / "Images"
DEFAULT_PRIMARY_FILENAME = "asset_dependency_map.png"


def _dependency_map() -> Dict[str, Sequence[str]]:
    return dict(BASE_ASSET_DEPENDENCIES)


def _edge_labels() -> Dict[Tuple[str, str], str]:
    return dict(BASE_EDGE_LABELS)


def _manual_positions() -> Dict[str, Tuple[float, float]]:
    return dict(BASE_MANUAL_POSITIONS)


def _collect_assets(dependency_map: Mapping[str, Sequence[str]]) -> Sequence[str]:
    """Gather the unique set of assets present in the dependency map."""
    downstream = set(dependency_map.keys())
    upstream = {asset for deps in dependency_map.values() for asset in deps}
    ordered = list(dict.fromkeys(list(upstream) + list(downstream)))
    return sorted(ordered)


def _circular_layout(nodes: Sequence[str], radius: float = 2.5) -> Dict[str, Tuple[float, float]]:
    """Fallback layout that arranges nodes on a circle."""
    positions: Dict[str, Tuple[float, float]] = {}
    n = max(len(nodes), 1)
    for idx, node in enumerate(nodes):
        angle = 2.0 * math.pi * (idx / n)
        positions[node] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions


def _compute_positions(nodes: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """Get plotting coordinates based on predefined manual positions."""
    manual_positions = _manual_positions()
    positions = {node: manual_positions[node] for node in nodes if node in manual_positions}
    missing = [node for node in nodes if node not in positions]
    if missing:
        positions.update(_circular_layout(missing))
    return positions


def _categorize_asset(asset: str) -> str:
    """Return the display category for the requested asset."""
    return ASSET_TYPE_MAP.get(asset, "Unclassified")


def _asset_coordinate_map() -> Dict[str, Tuple[float, float]]:
    """Return a mapping of asset -> (Easting, Northing) coordinates."""
    return dict(BASE_ASSET_COORDINATES)


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
) -> Tuple[Tuple[float, float, float, float], plt.AxesImage | None]:
    """Add a DEM background plus satellite imagery to mirror the c_hand example."""
    extent = _map_extent(coord_map)
    xmin, xmax, ymin, ymax = extent
    crs_to_use = ASSET_COORD_CRS
    dem_image: plt.AxesImage | None = None

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
    if cx is not None:
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
    else:
        warnings.warn(
            "contextily not available; using DEM-only background.",
            RuntimeWarning,
            stacklevel=2,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    return extent, dem_image


def _add_image_background(
    ax: plt.Axes,
    coord_map: Mapping[str, Tuple[float, float]],
    *,
    image_path: Path,
) -> Tuple[float, float, float, float]:
    """Use a pre-rendered image as the basemap background."""
    extent = _map_extent(coord_map)
    if not image_path.exists():
        raise FileNotFoundError(f"Background image not found at {image_path}.")
    img = mpimg.imread(image_path)
    xmin, xmax, ymin, ymax = extent
    img_artist = ax.imshow(img, extent=extent, zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    return extent, img_artist


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
        ax.annotate(
            "",
            xy=end_coord,
            xytext=start_coord,
            arrowprops=dict(
                arrowstyle="->",
                color=edge_color,
                linewidth=edge_linewidth,
                alpha=edge_alpha,
                shrinkA=8,
                shrinkB=8,
                mutation_scale=10,
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
        color = ASSET_TYPE_COLORS.get(asset_type, ASSET_TYPE_COLORS["Unclassified"])
        ax.scatter(
            [coord[0]],
            [coord[1]],
            s=160,
            color=color,
            edgecolors="#0f0f0f",
            linewidths=1.1,
            zorder=4,
        )
        label_x, label_y = coord
        if node == "Compressor1":
            label_y += 2500.0
        text = ax.text(
            label_x,
            label_y,
            node,
            fontsize=9,
            ha="left",
            va="bottom",
            color=node_label_color,
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
                arrowstyle="->",
                color="#4a4a4a",
                linewidth=1.4,
                shrinkA=8,
                shrinkB=8,
                mutation_scale=10,
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
        color = ASSET_TYPE_COLORS.get(asset_type, ASSET_TYPE_COLORS["Unclassified"])
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


def plot_asset_dependency_network(
    *,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    subplot_title: str | None = None,
    include_map: bool = True,
    dem_path: str | Path | None = DEFAULT_DEM_PATH,
    background_image: str | Path | None = DEFAULT_BACKGROUND_IMAGE,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot asset dependencies and optionally overlay geographic locations on a map."""
    dependency_map = _dependency_map()
    nodes = _collect_assets(dependency_map)
    edges = [
        (dependency, asset)
        for asset, dependencies in dependency_map.items()
        for dependency in dependencies
    ]
    edge_labels = _edge_labels()

    secondary_fig: plt.Figure | None = None

    if ax is None:
        if include_map:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    if include_map:
        coord_map_full = _asset_coordinate_map()
        relevant_coords = {node: coord_map_full[node] for node in nodes if node in coord_map_full}
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
        legend_handles = _draw_geographic_network(
            ax,
            nodes,
            edges,
            edge_labels,
            coord_map,
        )

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
                _add_image_background(ax_img, relevant_coords, image_path=image_param)
                ax_img.set_xlabel("Easting [m]")
                ax_img.set_ylabel("Northing [m]")
                if subplot_title:
                    ax_img.set_title(f"{subplot_title} (image background)", fontsize=13)
                else:
                    ax_img.set_title("Asset Dependency Map (image background)")
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
                )
                if legend_img:
                    ax_img.legend(
                        legend_img.values(),
                        legend_img.keys(),
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
        positions = _compute_positions(nodes)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0.2)
        if subplot_title:
            ax.set_title(subplot_title, fontsize=13)

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
            color = ASSET_TYPE_COLORS.get(asset_type, ASSET_TYPE_COLORS["Unclassified"])
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
    if save_path is not None:
        save_path_obj = Path(save_path)
    else:
        save_path_obj = DEFAULT_IMAGE_DIR / DEFAULT_PRIMARY_FILENAME

    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path_obj, bbox_inches="tight")
    secondary_path: Path | None = None
    if include_map and secondary_fig is not None:
        if save_path_obj.suffix:
            secondary_path = save_path_obj.with_name(f"{save_path_obj.stem}_image{save_path_obj.suffix}")
        else:
            secondary_path = Path(str(save_path_obj) + "_image")
        secondary_fig.savefig(secondary_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    plot_asset_dependency_network()
