# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:59:46 2025

@author: thoma
"""

import pandas as pd

# Handy options
df = pd.read_excel(
    "ipcc_ar6_sea_level_projection_29_-95.xlsx",
    sheet_name="Total",
    header=0,              # row with column names
    usecols="A:W",         # or list like [0,1,3]
    dtype={"ID": "Int64"}, # force column types
    na_values=["NA","-"]   # treat these as missing
)

print(df.head())          # it's now a table object
df.to_csv("Projections.csv", index=False)

# plot_scenarios_from_wide_quantiles.py
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- settings ----------------
CSV_PATH = r"Projections.csv"   # <-- change if needed
YEAR_MIN, YEAR_MAX = 2020, 2100
Y_LABEL = "Sea level rise (m)"
TITLE = "Median & 5–95% Quantile Sea Level Projections (IPCC AR6)"
OUT_PNG = "sea_level_scenarios_2020_2100.png"
# ------------------------------------------

df = pd.read_csv(CSV_PATH)

# --- identify year columns (strings of digits like '2020', '2030', ...) ---
year_cols = []
for c in df.columns:
    if isinstance(c, str) and c.isdigit():
        y = int(c)
        if 1800 <= y <= 2300:
            year_cols.append(c)

if not year_cols:
    raise ValueError("No year columns found (e.g., '2020', '2025', ...).")

# Keep only years within desired range
year_cols = sorted([c for c in year_cols if YEAR_MIN <= int(c) <= YEAR_MAX], key=int)
if not year_cols:
    raise ValueError(f"No year columns in [{YEAR_MIN}, {YEAR_MAX}].")

# --- detect required meta columns ---
# scenario
scenario_col = None
for cand in ["scenario", "Scenario", "ssp", "SSP", "rcp", "RCP", "case", "Case"]:
    if cand in df.columns:
        scenario_col = cand
        break
if scenario_col is None:
    raise ValueError("Couldn't find a 'scenario' column.")

# quantile
quantile_col = None
for cand in ["quantile", "Quantile", "quant", "Quant", "q", "Q", "percentile", "Percentile"]:
    if cand in df.columns:
        quantile_col = cand
        break
if quantile_col is None:
    raise ValueError("Couldn't find a 'quantile' column.")

# --- melt to long: (scenario, quantile, Year, Value) ---
id_vars = [scenario_col, quantile_col] + [c for c in df.columns
                                          if c not in year_cols and c not in [scenario_col, quantile_col]]
long_df = df.melt(id_vars=[scenario_col, quantile_col],
                  value_vars=year_cols,
                  var_name="Year",
                  value_name="Value")

# coerce types
long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")

# --- normalize quantiles to 0.05, 0.5, 0.95 ---
def to_prob(x):
    if pd.isna(x):
        return np.nan
    # numeric
    try:
        v = float(x)
        if v > 1:  # 5/50/95 -> 0.05/0.5/0.95
            v = v / 100.0
        return v
    except Exception:
        s = str(x).strip().lower()
        if "median" in s or "q50" in s or "p50" in s:
            return 0.5
        if "q05" in s or "p05" in s or s in {"5", "05", "0.05"}:
            return 0.05
        if "q95" in s or "p95" in s or s in {"95", "0.95"}:
            return 0.95
        m = re.search(r"(\d+(\.\d+)?)", s)
        if m:
            v = float(m.group(1))
            if v > 1:
                v /= 100.0
            return v
        return np.nan

long_df["__q__"] = long_df[quantile_col].map(to_prob)

# keep only the three needed quantiles
long_df = long_df.dropna(subset=["Year", "Value", "__q__"])
long_df = long_df[long_df["__q__"].isin([0.05, 0.5, 0.95])]

# --- pivot so each (scenario, year) has q05, q50, q95 ---
pivot = (long_df
         .pivot_table(index=[scenario_col, "Year"], columns="__q__", values="Value",
                      aggfunc="median")
         .rename(columns={0.05: "q05", 0.5: "q50", 0.95: "q95"})
         .reset_index())

# Some rows may miss a quantile; drop incomplete ones
pivot = pivot.dropna(subset=["q05", "q50", "q95"])

# --- plot ---
fig, ax = plt.subplots(figsize=(11, 6))

scenarios = sorted(pivot[scenario_col].unique())
cmap = plt.get_cmap("tab10")
colors = {sc: cmap(i % 10) for i, sc in enumerate(scenarios)}

for sc in scenarios:
    sub = pivot[pivot[scenario_col] == sc].sort_values("Year")
    x = sub["Year"].to_numpy()
    lo = sub["q05"].to_numpy()
    med = sub["q50"].to_numpy()
    hi = sub["q95"].to_numpy()

    ax.fill_between(x, lo, hi, alpha=0.25, color=colors[sc])
    ax.plot(x, med, linewidth=2.2, color=colors[sc], label=str(sc))

ax.set_xlim(YEAR_MIN, YEAR_MAX)
ax.set_xticks(list(range(((YEAR_MIN // 10) * 10), YEAR_MAX + 1, 10)))
ax.set_xlabel("Year")
ax.set_ylabel(Y_LABEL)
ax.set_title(TITLE)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(title="Scenario", ncol=2, frameon=True)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved figure -> {Path(OUT_PNG).resolve()}")
