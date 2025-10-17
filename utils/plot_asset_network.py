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
ASSET_DEPENDENCIES: Mapping[str, Sequence[str]] = {
    "Substation2": ("PV",),
    "Compressor1": ("LNG", "Substation1", "Substation2"),
    "Compressor2": ("LNG", "Substation1", "Substation2"),
    "Compressor3": ("LNG", "Substation1", "Substation2"),
    "ThermalUnit": ("Compressor3",),
    "Substation1": ("ThermalUnit",),
}

# Optional edge labels to highlight the role of key connections.
EDGE_LABELS: Dict[Tuple[str, str], str] = {
    ("Substation1", "Compressor1"): "Feeder S1-C1",
    ("Substation1", "Compressor2"): "Feeder S1-C2",
    ("Substation1", "Compressor3"): "Feeder S1-C3",
    ("Substation2", "Compressor1"): "Feeder S2-C1",
    ("Substation2", "Compressor2"): "Feeder S2-C2",
    ("Substation2", "Compressor3"): "Feeder S2-C3",
}

# Manually chosen coordinates to keep the diagram easy to read.
MANUAL_POSITIONS: Mapping[str, Tuple[float, float]] = {
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


def _collect_assets() -> Sequence[str]:
    """Gather the unique set of assets present in the dependency map."""
    downstream = set(ASSET_DEPENDENCIES.keys())
    upstream = {asset for deps in ASSET_DEPENDENCIES.values() for asset in deps}
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


def _compute_positions(nodes: Sequence[str], use_networkx: bool) -> Dict[str, Tuple[float, float]]:
    """Get plotting coordinates, optionally using networkx spring layout."""
    manual = {node: MANUAL_POSITIONS[node] for node in nodes if node in MANUAL_POSITIONS}
    missing = [node for node in nodes if node not in manual]

    if use_networkx:
        try:
            import networkx as nx  # type: ignore
        except ImportError:
            pass
        else:
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(
                (dependency, asset)
                for asset, dependencies in ASSET_DEPENDENCIES.items()
                for dependency in dependencies
            )
            manual = nx.spring_layout(
                graph,
                seed=42,
                k=2.0,
                scale=8.0,
                iterations=300,
            )  # type: ignore[assignment]
            return dict(manual)

    if missing:
        manual.update(_circular_layout(missing))
    return manual


def plot_asset_dependency_network(
    *,
    ax: plt.Axes | None = None,
    show: bool = True,
    save_path: str | None = None,
    use_networkx: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the asset dependency graph described by TrialEnv."""
    nodes = _collect_assets()
    positions = _compute_positions(nodes, use_networkx=use_networkx)
    edges = [
        (dependency, asset)
        for asset, dependencies in ASSET_DEPENDENCIES.items()
        for dependency in dependencies
    ]

    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.2)

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

        label = EDGE_LABELS.get((start, end))
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
            color="#2b83ba" if node not in ASSET_DEPENDENCIES else "#abdda4",
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
    plot_asset_dependency_network()
