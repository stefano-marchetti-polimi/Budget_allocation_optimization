# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 10:33:21 2025

@author: thoma
"""

import pandas as pd

# Handy options
df = pd.read_excel(
    "DM_Scenarios_Source.xlsx",
    sheet_name="DM_Scenarios",
    header=0,              # row with column names
    usecols="A:S",         # or list like [0,1,3]
    dtype={"ID": "Int64"}, # force column types
    na_values=["NA","-"]   # treat these as missing
)

print(df.head())          # it's now a table object
df.to_csv("DM_Scenarios.csv", index=False)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import cycle

# ---- Load df (use your existing df if already defined) ----
try:
    _ = df
    _df = df.copy()
except NameError:
    _df = pd.read_csv(Path("/mnt/data/DM_Scenarios.csv"))

# ---- Make column names safe (string + strip) ----
_df.columns = pd.Index(map(str, _df.columns)).str.strip()

# ---- Normalize key columns (case-insensitive match) ----
wanted = ["scenario", "commodity", "weight"]
rename_map = {}
lower_map = {c.lower(): c for c in _df.columns}
for k in wanted:
    if k in lower_map:
        rename_map[lower_map[k]] = k
_df = _df.rename(columns=rename_map)

# Sanity: pick id_vars that actually exist
id_vars = [c for c in wanted if c in _df.columns]

# ---- Detect year columns (names that are all digits) ----
year_cols = [c for c in _df.columns if str(c).strip().isdigit()]
if not year_cols:
    raise ValueError("Could not find year columns (headers like 2025, 2030, ...).")

# ---- Long format ----
long = _df.melt(
    id_vars=id_vars,
    value_vars=year_cols,
    var_name="year",
    value_name="value"
)

# Ensure correct dtypes
long["year"] = long["year"].astype(int)
long["value"] = pd.to_numeric(long["value"], errors="coerce")

# ---- Weights & scenarios ----
# Use this preferred order if present; otherwise use what exists
preferred_weights = ["W_g", "W_e", "W_ge", "W_gs", "W_ee", "W_es"]
available_weights = list(long["weight"].dropna().unique()) if "weight" in long else []
weights = [w for w in preferred_weights if w in available_weights]
if len(weights) < 6:  # fill any missing with the rest to reach up to 6 if possible
    for w in available_weights:
        if w not in weights:
            weights.append(w)
weights = weights[:6]  # ensure at most six panels

if len(weights) == 0:
    raise ValueError("No 'weight' values found. Check that the CSV has a 'weight' column.")

scenarios = list(dict.fromkeys(sorted(long["scenario"].unique()))) if "scenario" in long else []
years_sorted = sorted(long["year"].unique())

# ---- Colors for scenarios (consistent across subplots) ----
prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None) or [f"C{i}" for i in range(10)]
scenario_colors = {s: c for s, c in zip(scenarios, cycle(prop_cycle))}

# ---- Plot: 3x2 subplots, one per weight; scenarios as colored lines ----
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
axes = axes.flatten()

for i in range(6):
    ax = axes[i]
    if i < len(weights):
        w = weights[i]
        sub = long[long["weight"] == w]

        for scen in scenarios:
            s = (
                sub[sub["scenario"] == scen]
                .set_index("year")
                .reindex(years_sorted)["value"]
            )
            ax.plot(years_sorted, s.values, marker="o", label=scen,
                    linewidth=1.8, color=scenario_colors.get(scen))
        ax.set_title(w, fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Weight")
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.05)   # 5% margin on x-axis

    else:
        ax.axis("off")

# Shared legend
if scenarios:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(scenarios), 4), frameon=False, bbox_to_anchor=(0.5, 0.01))

fig.suptitle("Preference Weights by Scenarios", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0.05, 1, 0.95])

out_name = "weights_vs_year_by_scenario_6subplots.png"
plt.savefig(out_name, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {out_name}")
