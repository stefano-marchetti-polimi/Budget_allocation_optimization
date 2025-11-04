#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
from pathlib import Path
import warnings
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio.v3 as iio

from optimization_parallel import TrialEnv, env_kwargs, year_step

TEST_ENV_KWARGS = dict(env_kwargs)
TEST_ENV_KWARGS["mc_samples"] = 1000
from utils.plot_asset_network import (
    DEFAULT_BACKGROUND_IMAGE,
    DEFAULT_DEM_PATH,
    _network_snapshot,
    _map_extent,
    _add_dem_basemap,
    _draw_geographic_network,
    _edge_labels,
)


def _calendar_year_range(base_year: int, horizon: int, step: int) -> list[int]:
    """Return calendar years from base_year to base_year + horizon inclusive."""
    if step <= 0:
        step = 1
    end_year = base_year + horizon
    return list(range(base_year, end_year + 1, step))


def _snap_calendar_years(
    years: Iterable[float],
    *,
    base_year: int,
    horizon: int,
    step: int,
) -> list[int]:
    """Snap arbitrary calendar years to the simulation grid."""
    end_year = base_year + horizon
    snapped: list[int] = []
    if step <= 0:
        step = 1
    for value in years:
        offset = round((float(value) - base_year) / step) * step
        snapped_year = base_year + int(offset)
        snapped_year = max(base_year, min(snapped_year, end_year))
        if snapped_year not in snapped:
            snapped.append(snapped_year)
    snapped.sort()
    return snapped


def _compute_state_payloads(
    env: TrialEnv,
    years: Sequence[int],
    *,
    reuse_samples: bool = True,
) -> tuple[list[dict[str, dict[str, np.ndarray]]], dict[str, np.ndarray] | None]:
    """Evaluate availability states for the requested years."""
    if not years:
        return [], None
    improvement = env.improvement_height.copy()
    payloads: list[dict[str, dict[str, np.ndarray]]] = []
    cache: dict[str, np.ndarray] | None = None
    for idx, year in enumerate(years):
        kwargs = dict(year=float(year), return_states=True)
        if reuse_samples and cache is not None:
            kwargs["random_cache"] = cache
            _, state_payload = env._compute_metrics(improvement, **kwargs)  # type: ignore[arg-type]
        else:
            kwargs["return_cache"] = reuse_samples
            result = env._compute_metrics(improvement, **kwargs)  # type: ignore[arg-type]
            if reuse_samples:
                _, cache, state_payload = result  # type: ignore[misc]
            else:
                _, state_payload = result  # type: ignore[misc]
        if state_payload is None:
            raise RuntimeError("Environment did not return state payloads; ensure return_states=True is supported.")
        payloads.append(state_payload)
    return payloads, cache


def _summarise_probabilities(
    env: TrialEnv,
    env_years: Sequence[int],
    payloads: Sequence[dict[str, dict[str, np.ndarray]]],
    *,
    base_year: int = 0,
) -> pd.DataFrame:
    if len(env_years) != len(payloads):
        raise ValueError("Mismatch between requested years and computed payloads.")
    order = {name: idx for idx, name in enumerate(env.component_names)}
    records: list[dict[str, object]] = []
    for env_year, state in zip(env_years, payloads):
        functional = state["functional"]
        availability = state["availability"]
        calendar_year = base_year + env_year
        for name in env.component_names:
            func = functional[name]
            avail = availability[name]
            records.append(
                {
                    "env_year": env_year,
                    "year": calendar_year,
                    "asset": name,
                    "category": env.component_categories[name],
                    "failure_probability": float(1.0 - func.mean()),
                    "outage_probability": float(1.0 - avail.mean()),
                }
            )
    df = pd.DataFrame.from_records(records)
    df["asset_order"] = df["asset"].map(order)
    df = df.sort_values(["env_year", "asset_order"]).reset_index(drop=True)
    return df


def _plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Path | None,
    vmax: float | None = None,
) -> plt.Figure:
    ordered_assets = list(df.sort_values("asset_order")["asset"].unique())
    ordered_years = list(sorted(df["year"].unique()))
    pivot = (
        df.pivot(index="asset", columns="year", values=value_col)
        .reindex(index=ordered_assets, columns=ordered_years)
        .fillna(0.0)
    )
    data = pivot.to_numpy(dtype=np.float32)

    height = max(3.0, 0.35 * len(ordered_assets))
    width = max(4.5, 0.6 * len(ordered_years))
    fig, ax = plt.subplots(figsize=(width, height))
    data_max = float(np.nanmax(data)) if data.size else 0.0
    vmax_value = vmax if vmax is not None else max(data_max, 1e-6)
    if vmax_value <= 0.0:
        vmax_value = 1e-3
    im = ax.imshow(data, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax_value)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Asset")
    ax.set_xticks(np.arange(len(ordered_years)))
    ax.set_xticklabels([str(int(y)) for y in ordered_years])
    ax.set_yticks(np.arange(len(ordered_assets)))
    ax.set_yticklabels(ordered_assets)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Probability")
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _resolve_map_metric(metric: str | None) -> tuple[str, str]:
    if metric and metric.lower() == "outage":
        return "outage_probability", "Service outage probability"
    return "failure_probability", "Failure probability"


def _plot_probability_map(
    df: pd.DataFrame,
    year: int,
    value_col: str,
    title_prefix: str,
    output_path: Path | None,
    vmax: float | None = None,
    *,
    cmap: str = "YlOrRd",
    dem_path: Path | None = None,
    background_image: Path | None = DEFAULT_BACKGROUND_IMAGE,
) -> plt.Figure:
    snapshot = _network_snapshot()
    coord_map_full = snapshot["coordinates"]
    nodes = list(snapshot["components"].keys())
    edges = list(snapshot["edges"])
    edge_labels = _edge_labels()

    fig, ax = plt.subplots(figsize=(10, 10))
    extent = None
    dem_source = Path(dem_path) if dem_path is not None else None
    bg_source = Path(background_image) if background_image is not None else None
    try:
        extent, _ = _add_dem_basemap(
            ax,
            coord_map_full,
            dem_path=dem_source,
            background_image=bg_source,
        )
    except Exception as exc:
        warnings.warn(
            f"Basemap unavailable ({exc}); falling back to plain background.",
            RuntimeWarning,
            stacklevel=2,
        )
        extent = None
    if extent is None:
        xmin, xmax, ymin, ymax = _map_extent(coord_map_full)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_facecolor("#f2f2f2")

    _draw_geographic_network(
        ax,
        nodes,
        edges,
        edge_labels,
        coord_map_full,
        edge_color="#ffffff",
        colored_arrows=False,
        edge_alpha=0.65,
        edge_linewidth=1.0,
        label_offset=2300.0,
        label_fontsize=7.0,
        node_size=120.0,
    )

    year_df = df[df["year"] == year]
    if not year_df.empty:
        coords_x: list[float] = []
        coords_y: list[float] = []
        values: list[float] = []
        for asset, value in year_df.set_index("asset")[value_col].items():
            coord = coord_map_full.get(asset)
            if coord is None:
                continue
            coords_x.append(coord[0])
            coords_y.append(coord[1])
            values.append(float(value))
        if values:
            vmax_value = vmax if vmax is not None else max(max(values), 1e-6)
            if vmax_value <= 0.0:
                vmax_value = 1e-3
            size_scale = 300.0
            normalized = np.array(values, dtype=np.float32)
            if vmax_value > 0:
                normalized = normalized / vmax_value
            sizes = 120.0 + size_scale * normalized
            scatter = ax.scatter(
                coords_x,
                coords_y,
                c=values,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax_value,
                s=sizes,
                edgecolors="black",
                linewidths=0.7,
                zorder=6,
            )
            cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("Probability")
            top_assets = (
                year_df.sort_values(value_col, ascending=False)
                .head(3)
                .itertuples()
            )
            for asset_row in top_assets:
                coord = coord_map_full.get(asset_row.asset)
                if coord is None:
                    continue
                ax.text(
                    coord[0],
                    coord[1] + 2500.0,
                    f"{asset_row.asset} ({getattr(asset_row, value_col)*100:.2f}%)",
                    fontsize=8,
                    color="black",
                    ha="center",
                    va="bottom",
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65),
                )

    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.set_title(f"{title_prefix} – Year {int(year)}")
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No data to display.")
        return
    for year, group in df.groupby("year"):
        print(f"\n=== Year {int(year)} ===")
        header = f"{'Asset':<28}{'Category':<16}{'Failure%':>12}{'Outage%':>12}"
        print(header)
        print("-" * len(header))
        for _, row in group.iterrows():
            fail_pct = row["failure_probability"] * 100.0
            outage_pct = row["outage_probability"] * 100.0
            print(
                f"{row['asset']:<28}{row['category']:<16}"
                f"{fail_pct:>11.2f}%{outage_pct:>11.2f}%"
            )
        worst = group.sort_values("outage_probability", ascending=False).head(3)
        worst_assets = ", ".join(
            f"{r.asset} ({r.outage_probability*100:.1f}%)" for r in worst.itertuples()
        )
        print(f"Top outage probabilities: {worst_assets}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo failure probability snapshot for TrialEnv."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1042,
        help="Random seed used when resetting the environment.",
    )
    parser.add_argument(
        "--years",
        type=float,
        nargs="*",
        help="Specific years to evaluate (defaults to four evenly spaced points).",
    )
    parser.add_argument(
        "--no-shared-samples",
        action="store_true",
        help="Draw independent Monte Carlo samples for each year (default reuses the same samples).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save the per-asset probabilities as CSV.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Directory to save probability heatmaps. Two files will be written per run.",
    )
    parser.add_argument(
        "--show-plots",
        default=True,
        action="store_true",
        help="Display generated figures interactively.",
    )
    parser.add_argument(
        "--heatmap-max",
        type=float,
        default=0.05,
        help="Upper bound for colour scale (default: 0.1).",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        help="Directory to save per-year geographic probability maps.",
    )
    parser.add_argument(
        "--map-metric",
        choices=("failure", "outage"),
        default="failure",
        help="Probability metric to visualise on geographic maps.",
    )
    parser.add_argument(
        "--map-dem-path",
        type=Path,
        help="Optional DEM raster to overlay on the geographic plots.",
    )
    parser.add_argument(
        "--map-background",
        type=Path,
        help="Optional raster/image background for geographic plots (defaults to the inundation snapshot).",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2025,
        help="Calendar year corresponding to simulation year 0 (default: 2025).",
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        help="Optional path to save an animated GIF of the geographic maps.",
    )
    parser.add_argument(
        "--gif-duration",
        type=float,
        default=5,
        help="Frame duration (seconds) for the animated GIF (default: 1.5).",
    )
    args = parser.parse_args()

    horizon = int(TEST_ENV_KWARGS.get("years", 75))
    step = int(TEST_ENV_KWARGS.get("year_step", year_step))
    base_year = int(args.base_year)

    if args.years:
        calendar_years = _snap_calendar_years(
            args.years,
            base_year=base_year,
            horizon=horizon,
            step=step,
        )
    else:
        calendar_years = _calendar_year_range(base_year, horizon, step)

    if not calendar_years:
        raise ValueError("No valid evaluation years were provided.")
    env_years = [int(year - base_year) for year in calendar_years]

    env = TrialEnv(**TEST_ENV_KWARGS)
    env.reset(seed=args.seed)
    payloads, _ = _compute_state_payloads(
        env,
        env_years,
        reuse_samples=not args.no_shared_samples,
    )
    env.close()

    df = _summarise_probabilities(env, env_years, payloads, base_year=base_year)
    _print_summary(df)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["asset_order", "env_year"]).to_csv(args.output_csv, index=False)
        print(f"\nSaved results to {args.output_csv}")

    figs: list[plt.Figure] = []

    heatmap_requested = args.plot_dir is not None or args.show_plots
    if heatmap_requested:
        failure_path = (
            args.plot_dir / "failure_probability_heatmap.png" if args.plot_dir else None
        )
        outage_path = (
            args.plot_dir / "outage_probability_heatmap.png" if args.plot_dir else None
        )
        figs.append(
            _plot_heatmap(
                df,
                "failure_probability",
                "Failure probability (per asset)",
                failure_path,
                vmax=args.heatmap_max,
            )
        )
        figs.append(
            _plot_heatmap(
                df,
                "outage_probability",
                "Service outage probability (per asset)",
                outage_path,
                vmax=args.heatmap_max,
            )
        )

    map_requested = args.map_dir is not None or args.show_plots
    if map_requested:
        metric_col, metric_label = _resolve_map_metric(args.map_metric)
        dem_path = Path(args.map_dem_path) if args.map_dem_path is not None else None
        background_image = (
            Path(args.map_background) if args.map_background is not None else DEFAULT_BACKGROUND_IMAGE
        )
        if args.heatmap_max is not None:
            map_vmax = float(args.heatmap_max)
        else:
            metric_array = df[metric_col].to_numpy(dtype=np.float32)
            if metric_array.size == 0 or not np.isfinite(metric_array).any():
                map_vmax = 1e-3
            else:
                map_vmax = float(np.nanmax(metric_array))
                if map_vmax <= 0.0:
                    map_vmax = 1e-3
        gif_frames: list[np.ndarray] = []
        for calendar_year in calendar_years:
            map_path = (
                args.map_dir / f"{metric_col}_map_year_{int(calendar_year)}.png"
                if args.map_dir
                else None
            )
            fig_map = _plot_probability_map(
                df,
                int(calendar_year),
                metric_col,
                metric_label,
                map_path,
                vmax=map_vmax,
                cmap="YlOrRd",
                dem_path=None if background_image is not None else dem_path,
                background_image=background_image,
            )
            figs.append(fig_map)
            buf = io.BytesIO()
            fig_map.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            buf.seek(0)
            gif_frames.append(iio.imread(buf.getvalue()))

        if gif_frames:
            if args.gif_path is not None:
                gif_path = args.gif_path
            elif args.map_dir is not None:
                gif_path = args.map_dir / f"{metric_col}_transition.gif"
            else:
                gif_path = Path("outputs") / f"{metric_col}_transition.gif"
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            duration = max(0.05, float(args.gif_duration))
            frame_count = len(gif_frames)
            durations = [duration] * frame_count
            iio.imwrite(
                gif_path,
                gif_frames,
                loop=0,
                duration=durations,
            )
            print(f"\nSaved animation to {gif_path}")

    if args.show_plots and figs:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)



if __name__ == "__main__":
    main()
