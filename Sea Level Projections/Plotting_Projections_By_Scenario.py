# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 18:15:04 2025

@author: thoma
"""

# plot_selected_scenarios_projections.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# ========= SETTINGS =========
CSV_PATH = r"Projections.csv"   # adjust path if needed
YEAR_MIN, YEAR_MAX = 2025, 2100

# Select scenarios to plot (exact strings as they appear in the CSV).
SCENARIOS_TO_PLOT = []  # [] means "all scenarios"

Y_LABEL = "Sea level rise (m)"
TITLE = "Median & 5–95% Quantile Sea Level Projections (IPCC AR6)"
OUT_PNG = "sea_level_scenarios_selected.png"

# --- NEW: axis label + tick font sizes ---
X_LABEL_FONTSIZE = 14
Y_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 15
LEGEND_FONTSIZE = 12
LEGEND_TITLE_FONTSIZE = 14

# --- NEW: x tick placement (starts at YEAR_MIN) ---
X_TICK_STEP = 5  # 5 or 10 are common

# --- NEW: explicit scenario colors (hex or any matplotlib color) ---
SCENARIO_COLORS = {
    "SSP1-1.9": "#36C85F",
    "SSP1-2.6": "#ECCE1B",
    "SSP2-4.5": "#5A65FF",
    "SSP3-7.0": "#A200B8",
    "SSP5-8.5": "#D8271C",
}

BAND_ALPHA = 0.15
LINE_WIDTH = 2.2
# ============================

# ---- Load CSV ----
df = pd.read_csv(CSV_PATH)

# ---- Identify year columns (strict: strings of digits within range) ----
year_cols = [
    c for c in df.columns
    if isinstance(c, str) and c.isdigit() and YEAR_MIN <= int(c) <= YEAR_MAX
]
if not year_cols:
    raise ValueError(f"No year columns in [{YEAR_MIN}, {YEAR_MAX}] found in CSV.")

# ---- Ensure required columns exist ----
required_cols = ["scenario", "quantile"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}. "
        "Expected at least 'scenario' and 'quantile'."
    )

# ---- Melt to long format: (scenario, quantile, Year, Value) ----
long_df = df.melt(
    id_vars=["scenario", "quantile"],
    value_vars=sorted(year_cols, key=int),
    var_name="Year",
    value_name="Value"
)

# ---- Coerce types ----
long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")

# ---- Normalize quantiles to probabilities: 0.05, 0.5, 0.95 ----
def to_prob(x):
    if pd.isna(x):
        return np.nan
    try:
        v = float(x)
        if v > 1:
            v = v / 100.0
        return v
    except Exception:
        s = str(x).strip().lower()
        if "median" in s or "q50" in s or "p50" in s:
            return 0.5
        if "q05" in s or "p05" in s:
            return 0.05
        if "q95" in s or "p95" in s:
            return 0.95
        m = re.search(r"(\d+(\.\d+)?)", s)
        if m:
            v = float(m.group(1))
            if v > 1:
                v /= 100.0
            return v
        return np.nan

long_df["q"] = long_df["quantile"].map(to_prob)

# Keep only needed quantiles and valid values
long_df = long_df.dropna(subset=["Year", "Value", "q"])
long_df = long_df[long_df["q"].isin([0.05, 0.5, 0.95])]

# ---- Filter scenarios if user specified any ----
all_scenarios = sorted(long_df["scenario"].unique())
if SCENARIOS_TO_PLOT:
    miss = [s for s in SCENARIOS_TO_PLOT if s not in all_scenarios]
    if miss:
        raise ValueError(
            f"Selected scenarios not found in CSV: {miss}\n"
            f"Available: {all_scenarios}"
        )
    long_df = long_df[long_df["scenario"].isin(SCENARIOS_TO_PLOT)]
else:
    SCENARIOS_TO_PLOT = all_scenarios  # plot all

# ---- Pivot so each (scenario, year) has q05, q50, q95 ----
pivot = (
    long_df.pivot_table(
        index=["scenario", "Year"],
        columns="q",
        values="Value",
        aggfunc="median"
    )
    .rename(columns={0.05: "q05", 0.5: "q50", 0.95: "q95"})
    .reset_index()
)

pivot = pivot.dropna(subset=["q05", "q50", "q95"])

# ---- Plot ----
fig, ax = plt.subplots(figsize=(11, 6))

# Fallback colormap for scenarios not listed in SCENARIO_COLORS
fallback_cmap = plt.get_cmap("Dark2")
fallback_colors = {
    sc: fallback_cmap(i % 10) for i, sc in enumerate(SCENARIOS_TO_PLOT)
}

for sc in SCENARIOS_TO_PLOT:
    sub = pivot[pivot["scenario"] == sc].sort_values("Year")
    if sub.empty:
        continue

    x   = sub["Year"].to_numpy()
    lo  = sub["q05"].to_numpy()
    med = sub["q50"].to_numpy()
    hi  = sub["q95"].to_numpy()

    color = SCENARIO_COLORS.get(sc, fallback_colors[sc])

    ax.fill_between(x, lo, hi, alpha=BAND_ALPHA, color=color)
    ax.plot(x, med, linewidth=LINE_WIDTH, color=color, label=sc)

# Limits
ax.set_xlim(YEAR_MIN, YEAR_MAX)

# --- NEW: ticks start exactly at YEAR_MIN ---
ticks = list(range(YEAR_MIN, YEAR_MAX + 1, X_TICK_STEP))
ax.set_xticks(ticks)

# --- NEW: font size controls ---
ax.set_xlabel("Year", fontsize=X_LABEL_FONTSIZE)
ax.set_ylabel(Y_LABEL, fontsize=Y_LABEL_FONTSIZE)
ax.set_title(TITLE, fontsize=TITLE_FONTSIZE)
ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)

ax.grid(True, linestyle="--", alpha=0.5)
leg = ax.legend(title="CC scenario", ncol=2, frameon=True, fontsize=LEGEND_FONTSIZE)
leg.get_title().set_fontsize(LEGEND_TITLE_FONTSIZE)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300)
print(f"Saved figure -> {Path(OUT_PNG).resolve()}")
