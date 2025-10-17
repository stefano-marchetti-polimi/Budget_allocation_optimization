"""Plot the dependency network of assets in the TrialEnv environment.

The relationships mirror the logical dependencies implemented in
``utils.environment_placeholder.TrialEnv._compute_metrics`` where service
availability propagates across assets such as PV, substations, compressors,
the thermal unit, and the LNG terminal.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt

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

EXTENDED_ASSET_DEPENDENCIES: Mapping[str, Sequence[str]] = {
    "Substation3": ("ThermalUnit",),
    "Compressor4": ("LNG_Backup", "Substation1", "Substation3"),
    "ThermalUnit2": ("Compressor4",),
    "Substation4": ("PV_Expansion", "ThermalUnit2"),
    "Compressor5": ("LNG_Backup", "Substation3", "Substation4"),
}

# Optional edge labels to highlight the role of key connections.
BASE_EDGE_LABELS: Dict[Tuple[str, str], str] = {
    ("Substation1", "Compressor1"): "Feeder S1-C1",
    ("Substation1", "Compressor2"): "Feeder S1-C2",
    ("Substation1", "Compressor3"): "Feeder S1-C3",
    ("Substation2", "Compressor1"): "Feeder S2-C1",
    ("Substation2", "Compressor2"): "Feeder S2-C2",
    ("Substation2", "Compressor3"): "Feeder S2-C3",
}

EXTENDED_EDGE_LABELS: Dict[Tuple[str, str], str] = {
    ("Substation1", "Compressor4"): "Feeder S1-C4",
    ("Substation3", "Compressor4"): "Feeder S3-C4",
    ("Substation3", "Compressor5"): "Feeder S3-C5",
    ("Substation4", "Compressor5"): "Feeder S4-C5",
}

# Manually chosen coordinates to keep the diagram easy to read.
BASE_MANUAL_POSITIONS: Mapping[str, Tuple[float, float]] = {
    # Anchor sources far apart so downstream edges do not stack.
    "PV": (-11.0, 4.5),
    "LNG": (11.0, 6.0),
    # Downstream assets arranged to follow the logical flow left-to-right.
    "Substation2": (-6.0, 3.2),
    "Compressor1": (-2.0, 7.5),
    "Compressor2": (-2.0, 2.2),
    "Compressor3": (-2.0, -3.0),
    "ThermalUnit": (3.5, -5.5),
    "Substation1": (7.0, -2.0),
}

EXTENDED_MANUAL_POSITIONS: Mapping[str, Tuple[float, float]] = {
    "PV_Expansion": (-11.0, 8.5),
    "LNG_Backup": (13.0, 8.0),
    "Substation3": (3.5, 0.5),
    "Compressor4": (1.0, 4.5),
    "ThermalUnit2": (6.0, -6.5),
    "Substation4": (9.0, 2.0),
    "Compressor5": (5.0, 5.5),
}


def _dependency_map(include_extensions: bool) -> Dict[str, Sequence[str]]:
    data: Dict[str, Sequence[str]] = dict(BASE_ASSET_DEPENDENCIES)
    if include_extensions:
        data.update(EXTENDED_ASSET_DEPENDENCIES)
    return data


def _edge_labels(include_extensions: bool) -> Dict[Tuple[str, str], str]:
    labels: Dict[Tuple[str, str], str] = dict(BASE_EDGE_LABELS)
    if include_extensions:
        labels.update(EXTENDED_EDGE_LABELS)
    return labels


def _manual_positions(include_extensions: bool) -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = dict(BASE_MANUAL_POSITIONS)
    if include_extensions:
        positions.update(EXTENDED_MANUAL_POSITIONS)
    return positions


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


def _compute_positions(
    nodes: Sequence[str],
    include_extensions: bool,
) -> Dict[str, Tuple[float, float]]:
    """Get plotting coordinates based on predefined manual positions."""
    manual_positions = _manual_positions(include_extensions)
    positions = {node: manual_positions[node] for node in nodes if node in manual_positions}
    missing = [node for node in nodes if node not in positions]
    if missing:
        positions.update(_circular_layout(missing))
    return positions


def plot_asset_dependency_network(
    *,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    include_extensions: bool = False,
    subplot_title: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the asset dependency graph described by TrialEnv."""
    dependency_map = _dependency_map(include_extensions)
    nodes = _collect_assets(dependency_map)
    positions = _compute_positions(
        nodes,
        include_extensions=include_extensions,
    )
    edges = [
        (dependency, asset)
        for asset, dependencies in dependency_map.items()
        for dependency in dependencies
    ]
    edge_labels = _edge_labels(include_extensions)

    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.2)
    if subplot_title:
        ax.set_title(subplot_title, fontsize=13)

    # Draw edges first so node markers appear on top.
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
                # Offset label slightly perpendicular to the edge to avoid overlap.
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

    # Draw nodes with a consistent style.
    for node in nodes:
        x, y = positions[node]
        ax.scatter(
            [x],
            [y],
            s=440,
            color="#2b83ba" if node not in dependency_map else "#abdda4",
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

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    plot_asset_dependency_network(include_extensions=False)
