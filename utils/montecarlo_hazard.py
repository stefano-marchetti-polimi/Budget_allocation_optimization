import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt
from fragility_curves import fragility_PV, fragility_substation, fragility_compressor, fragility_thermal_unit, fragility_LNG_terminal

# Parameters
loc = 0
kH = 0.8019
sH = 0.1959
N = 10**5
n_bins = 50

# Create GPD distribution object
gpd = genpareto(c=kH, scale=sH, loc=0)

# Sample flood heights (exceedances + threshold) and reject > 4 m by resampling
samples = gpd.rvs(size=N)
mask = samples > 4.0
while np.any(mask):
    samples[mask] = gpd.rvs(size=int(np.sum(mask)))
    mask = samples > 4.0

# Create figure with 5 subplots arranged in a 3x2 grid (last subplot empty)
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
axs = axs.flatten()

# First subplot: histogram of flood heights only
axs[0].hist(samples, bins=n_bins, density=True, alpha=0.6, color='#1f77b4', label='Sampled data')
axs[0].set_xlabel('Flood height')
axs[0].set_ylabel('Density')
axs[0].set_title('Histogram of Flood Heights')
axs[0].legend()

PV_fragility = fragility_PV(samples)
substation_fragility = fragility_substation(samples)
compressor_fragility = fragility_compressor(samples)
thermal_unit_fragility = fragility_thermal_unit(samples)
LNG_terminal_fragility = fragility_LNG_terminal(samples)

# Plot each fragility histogram separately
axs[1].hist(PV_fragility, bins=n_bins, density=True, alpha=0.6, color='#e15759', label='PV Fragility')
axs[1].set_xlabel('Fragility')
axs[1].set_ylabel('Density')
axs[1].set_title('PV Fragility Distribution')
axs[1].legend()

axs[2].hist(substation_fragility, bins=n_bins, density=True, alpha=0.6, color='#59a14f', label='Substation Fragility')
axs[2].set_xlabel('Fragility')
axs[2].set_ylabel('Density')
axs[2].set_title('Substation Fragility Distribution')
axs[2].legend()

axs[3].hist(compressor_fragility, bins=n_bins, density=True, alpha=0.6, color='#f28e2b', label='Compressor Fragility')
axs[3].set_xlabel('Fragility')
axs[3].set_ylabel('Density')
axs[3].set_title('Compressor Fragility Distribution')
axs[3].legend()

axs[4].hist(thermal_unit_fragility, bins=n_bins, density=True, alpha=0.6, color='#9467bd', label='Thermal Unit Fragility')
axs[4].set_xlabel('Fragility')
axs[4].set_ylabel('Density')
axs[4].set_title('Thermal Unit Fragility Distribution')
axs[4].legend()

axs[5].hist(LNG_terminal_fragility, bins=n_bins, density=True, alpha=0.6, color='#17becf', label='LNG terminal Fragility')
axs[5].set_xlabel('Fragility')
axs[5].set_ylabel('Density')
axs[5].set_title('LNG terminal Fragility Distribution')
axs[5].legend()

plt.tight_layout()
plt.show()

# Monte Carlo simulation to determine operational status of network components
# Vectorized Monte Carlo simulation (no Python loop)
# 1 means operational, 0 means failed
rng = np.random.default_rng()
U = rng.random((8, N))  # 8 components × N samples

PV = (U[0] >= PV_fragility).astype(np.uint8)
substation_1 = (U[1] >= substation_fragility).astype(np.uint8)
substation_2 = (U[2] >= substation_fragility).astype(np.uint8)
compressor_1 = (U[3] >= compressor_fragility).astype(np.uint8)
compressor_2 = (U[4] >= compressor_fragility).astype(np.uint8)
compressor_3 = (U[5] >= compressor_fragility).astype(np.uint8)
thermal_unit = (U[6] >= thermal_unit_fragility).astype(np.uint8)
LNG_terminal = (U[7] >= LNG_terminal_fragility).astype(np.uint8)

# Calculate and print failure probabilities for each component
print("\n=== Failure Probabilities ===")
print(f"PV: {1 - np.mean(PV):.4f}")
print(f"Substation 1: {1 - np.mean(substation_1):.4f}")
print(f"Substation 2: {1 - np.mean(substation_2):.4f}")
print(f"Compressor 1: {1 - np.mean(compressor_1):.4f}")
print(f"Compressor 2: {1 - np.mean(compressor_2):.4f}")
print(f"Compressor 3: {1 - np.mean(compressor_3):.4f}")
print(f"Thermal Unit: {1 - np.mean(thermal_unit):.4f}")
print(f"LNG Terminal: {1 - np.mean(LNG_terminal):.4f}")

#define consumers split
industrial_consumer_gas = 0.4
industrial_consumer_electricity = 0.7
residential_consumer_gas = 0.6
residential_consumer_electricity = 0.3

# === Service propagation over dependency graph ===
# Interpret 1=up, 0=down arrays as booleans for logic operations
PV_b = PV.astype(bool)
sub1_b = substation_1.astype(bool)
sub2_b = substation_2.astype(bool)
comp1_b = compressor_1.astype(bool)
comp2_b = compressor_2.astype(bool)
comp3_b = compressor_3.astype(bool)
therm_b = thermal_unit.astype(bool)
LNG_b = LNG_terminal.astype(bool)

# Source services
PV_service = PV_b.copy()                 # PV generates if physically up
LNG_service = LNG_b.copy()               # LNG supply if physically up

# Substation 2 is fed by PV
sub2_service = sub2_b & PV_service
comp3_service = comp3_b & sub2_service & LNG_service
thermal_unit_service = therm_b & LNG_service & comp3_service
# Substation 1
sub1_service = sub1_b & thermal_unit_service
comp1_service = comp1_b & (sub1_service | sub2_service) & LNG_service
comp2_service = comp2_b & (sub1_service | sub2_service) & LNG_service

# === Consumer services ===
industrial_gas_service = LNG_service & comp1_service
residential_gas_service = LNG_service & comp2_service

industrial_elec_service = LNG_service & comp3_service & thermal_unit_service & sub1_service
residential_elec_service = PV_service & sub2_service

# Convert to floats for means
ig = industrial_gas_service.astype(float)
rg = residential_gas_service.astype(float)
ie = industrial_elec_service.astype(float)
re = residential_elec_service.astype(float)

print("\n=== Service Availabilities (means) ===")
print(f"Industrial gas: {ig.mean():.4f}")
print(f"Residential gas: {rg.mean():.4f}")
print(f"Industrial electricity: {ie.mean():.4f}")
print(f"Residential electricity: {re.mean():.4f}")


# === Losses and social impacts (sample-wise) ===
# Vectorized: one value per Monte Carlo sample

# Per-sample losses
gas_loss_samples = (1.0 - ig) * industrial_consumer_gas + (1.0 - rg) * residential_consumer_gas
electricity_loss_samples = (1.0 - ie) * industrial_consumer_electricity + (1.0 - re) * residential_consumer_electricity

# Per-sample social impacts (residential emphasis)
gas_social_samples = (1.0 - rg)
electricity_social_samples = (1.0 - re)

print("\n=== Sample-wise vectors ===")
print(f"gas_loss_samples shape: {gas_loss_samples.shape}")
print(f"electricity_loss_samples shape: {electricity_loss_samples.shape}")
print(f"gas_social_samples shape: {gas_social_samples.shape}")
print(f"electricity_social_samples shape: {electricity_social_samples.shape}")

# Weights
w_gas = 0.5
w_electricity = 0.5
w_gas_loss = 0.5
w_gas_social = 0.5
w_electricity_loss = 0.5
w_electricity_social = 0.5

# Per-sample reward
reward_samples = 1 - (
    w_gas * (gas_loss_samples*w_gas_loss + gas_social_samples*w_gas_social)
    + w_electricity * (electricity_loss_samples*w_electricity_loss + electricity_social_samples*w_electricity_social)
)

# Aggregate summaries (means and stds)
gas_loss_mean, gas_loss_std = float(gas_loss_samples.mean()), float(gas_loss_samples.std())
elec_loss_mean, elec_loss_std = float(electricity_loss_samples.mean()), float(electricity_loss_samples.std())
gas_soc_mean, gas_soc_std = float(gas_social_samples.mean()), float(gas_social_samples.std())
elec_soc_mean, elec_soc_std = float(electricity_social_samples.mean()), float(electricity_social_samples.std())
reward_mean, reward_std = float(reward_samples.mean()), float(reward_samples.std())

print("\n=== Losses & Social Impacts (means ± std) ===")
print(f"Gas loss: {gas_loss_mean:.4f} ± {gas_loss_std:.4f}")
print(f"Electricity loss: {elec_loss_mean:.4f} ± {elec_loss_std:.4f}")
print(f"Gas social impact: {gas_soc_mean:.4f} ± {gas_soc_std:.4f}")
print(f"Electricity social impact: {elec_soc_mean:.4f} ± {elec_soc_std:.4f}")
print(f"Reward: {reward_mean:.4f} ± {reward_std:.4f}")

p = np.percentile(reward_samples, [5, 50, 95])
print(f"Reward percentiles (5/50/95): {p}")

# === Visualize distributions (better views) ===
# 1) ECDFs (good for probabilities on [0,1])
metrics = [
    (gas_loss_samples, 'Gas loss'),
    (electricity_loss_samples, 'Electricity loss'),
    (gas_social_samples, 'Gas social (norm.)'),
    (electricity_social_samples, 'Electricity social (norm.)'),
    (reward_samples, 'Reward')
]

fig_ecdf, ax_ecdf = plt.subplots(1, 1, figsize=(10, 6))
for data, label in metrics:
    x = np.sort(data)
    y = np.linspace(0, 1, x.size, endpoint=True)
    ax_ecdf.step(x, y, where='post', label=label)
ax_ecdf.set_xlabel('Value')
ax_ecdf.set_ylabel('ECDF')
ax_ecdf.set_title('Empirical CDFs of Losses, Social Impacts, and Reward')
ax_ecdf.legend(loc='lower right')
ax_ecdf.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2) Violin comparison (compact view of distribution shapes)
fig_violin, ax_v = plt.subplots(1, 1, figsize=(10, 6))
viol = ax_v.violinplot(
    [m[0] for m in metrics],
    showmeans=True,
    showmedians=True,
    showextrema=False,
)
ax_v.set_xticks(np.arange(1, len(metrics)+1))
ax_v.set_xticklabels([m[1] for m in metrics], rotation=20, ha='right')
ax_v.set_ylabel('Value')
ax_v.set_title('Distribution comparison (violin plots)')
ax_v.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()