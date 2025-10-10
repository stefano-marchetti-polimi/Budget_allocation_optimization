import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt
from fragility_curves import fragility_PV, fragility_substation, fragility_compressor, fragility_thermal_unit, fragility_LNG_terminal
import time

# Parameters
loc = 0
kH = 0.8019
sH = 0.1959
N = 10**6
n_bins = 50
threhsold = 0

# Plot toggle and spatial variability (meters)
DO_PLOT = False
spatial_sigma = 0.25  # std dev of spatial depth variability per component (m)

t_start = time.perf_counter()

# Create GPD distribution object
gpd = genpareto(c=kH, scale=sH, loc=0)

# Sample flood heights (exceedances + threshold) and reject > 4 m by resampling
samples = gpd.rvs(size=N)
mask = samples > 4.0
while np.any(mask):
    samples[mask] = gpd.rvs(size=int(np.sum(mask)))
    mask = samples > 4.0

samples += threhsold

# Add spatial variability to depths without clipping (reject out-of-bounds and resample noise)
def add_spatial_variability(base_depths, sigma, rng):
    depths = base_depths + rng.normal(0.0, sigma, size=base_depths.shape)
    mask = (depths < 0.0) | (depths > 4.0)
    while np.any(mask):
        depths[mask] = base_depths[mask] + rng.normal(0.0, sigma, size=int(np.sum(mask)))
        mask = (depths < 0.0) | (depths > 4.0)
    return depths

# Component-specific depths with spatial variability
rng = np.random.default_rng()
samples_PV   = add_spatial_variability(samples, spatial_sigma, rng)
samples_sub1 = add_spatial_variability(samples, spatial_sigma, rng)
samples_sub2 = add_spatial_variability(samples, spatial_sigma, rng)
samples_c1   = add_spatial_variability(samples, spatial_sigma, rng)
samples_c2   = add_spatial_variability(samples, spatial_sigma, rng)
samples_c3   = add_spatial_variability(samples, spatial_sigma, rng)
samples_th   = add_spatial_variability(samples, spatial_sigma, rng)
samples_lng  = add_spatial_variability(samples, spatial_sigma, rng)

# Fragilities per component
PV_fragility = fragility_PV(samples_PV)
substation1_fragility = fragility_substation(samples_sub1)
substation2_fragility = fragility_substation(samples_sub2)
compressor1_fragility = fragility_compressor(samples_c1)
compressor2_fragility = fragility_compressor(samples_c2)
compressor3_fragility = fragility_compressor(samples_c3)
thermal_unit_fragility = fragility_thermal_unit(samples_th)
LNG_terminal_fragility = fragility_LNG_terminal(samples_lng)

if DO_PLOT:
    # Create figure with 5 subplots arranged in a 3x2 grid (last subplot empty)
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs = axs.flatten()

    # First subplot: histogram of flood heights only
    axs[0].hist(samples, bins=n_bins, density=True, alpha=0.6, color='#1f77b4', label='Sampled data')
    axs[0].set_xlabel('Flood height')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Histogram of Flood Heights')
    axs[0].legend()

    # Plot each fragility histogram separately
    axs[1].hist(PV_fragility, bins=n_bins, density=True, alpha=0.6, color='#e15759', label='PV Fragility')
    axs[1].set_xlabel('Fragility')
    axs[1].set_ylabel('Density')
    axs[1].set_title('PV Fragility Distribution')
    axs[1].legend()

    axs[2].hist(np.concatenate([substation1_fragility, substation2_fragility]), bins=n_bins, density=True, alpha=0.6, color='#59a14f', label='Substation Fragility')
    axs[2].set_xlabel('Fragility')
    axs[2].set_ylabel('Density')
    axs[2].set_title('Substation Fragility Distribution')
    axs[2].legend()

    axs[3].hist(np.concatenate([compressor1_fragility, compressor2_fragility, compressor3_fragility]), bins=n_bins, density=True, alpha=0.6, color='#f28e2b', label='Compressor Fragility')
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
rng_mc = np.random.default_rng()
U = rng_mc.random((8, N))  # 8 components × N samples

PV = (U[0] >= PV_fragility).astype(np.uint8)
substation_1 = (U[1] >= substation1_fragility).astype(np.uint8)
substation_2 = (U[2] >= substation2_fragility).astype(np.uint8)
compressor_1 = (U[3] >= compressor1_fragility).astype(np.uint8)
compressor_2 = (U[4] >= compressor2_fragility).astype(np.uint8)
compressor_3 = (U[5] >= compressor3_fragility).astype(np.uint8)
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

# THIS HAS BEEN ADDED TO ADD VARIABILITY
# Independent feeder availability from each substation to each compressor (decouple services)
# Slightly different reliability per compressor feeders
p_feeder_avail_c1 = 0.77
p_feeder_avail_c2 = 0.95
p_feeder_avail_c3 = 0.84
rng_feeder = np.random.default_rng()

# For compressor 1
feed_s1_c1 = (rng_feeder.random(N) < p_feeder_avail_c1)
feed_s2_c1 = (rng_feeder.random(N) < p_feeder_avail_c1)
# For compressor 2
feed_s1_c2 = (rng_feeder.random(N) < p_feeder_avail_c2)
feed_s2_c2 = (rng_feeder.random(N) < p_feeder_avail_c2)
# For compressor 3
feed_s1_c3 = (rng_feeder.random(N) < p_feeder_avail_c3)
feed_s2_c3 = (rng_feeder.random(N) < p_feeder_avail_c3)

# Initialize sub1 with no thermal generation yet (will be filled iteratively)
sub1_service = sub1_b & False

# Fixed-point iteration to handle the power<->compressor<->thermal_unit cycle
for _ in range(10):
    # Per-compressor power availability via feeders from substations
    power_ok1 = (sub1_service & feed_s1_c1) | (sub2_service & feed_s2_c1)
    power_ok2 = (sub1_service & feed_s1_c2) | (sub2_service & feed_s2_c2)
    power_ok3 = (sub1_service & feed_s1_c3) | (sub2_service & feed_s2_c3)

    comp1_service = comp1_b & power_ok1 & LNG_service
    comp2_service = comp2_b & power_ok2 & LNG_service
    comp3_service = comp3_b & power_ok3 & LNG_service

    thermal_unit_service = therm_b & LNG_service & comp3_service

    sub1_new = sub1_b & thermal_unit_service  # sub1 requires thermal generation

    # Check convergence
    if np.array_equal(sub1_new, sub1_service):
        sub1_service = sub1_new
        break
    sub1_service = sub1_new


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

def summarize_binary(name, a):
    uniq, counts = np.unique(a.astype(int), return_counts=True)
    print(f"{name} unique values:", dict(zip(uniq, counts)))

print("\n=== Debug: service coupling ===")
summarize_binary("industrial_gas_service", ig)
summarize_binary("residential_gas_service", rg)
summarize_binary("industrial_elec_service", ie)
summarize_binary("residential_elec_service", re)

# Pairwise agreement rates
same_gas = np.mean(ig == rg)
same_elec = np.mean(ie == re)
print(f"ig == rg fraction: {same_gas:.4f}")
print(f"ie == re fraction: {same_elec:.4f}")

# Check compressor independence (root cause for gas)
print("comp1_service vs comp2_service identical fraction:", np.mean(comp1_service == comp2_service))

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

t_end = time.perf_counter()
print(f"\nTotal runtime: {t_end - t_start:.3f} s")


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

# === Histogram comparison for losses, social impacts, and reward ===
fig_hist, axs_hist = plt.subplots(2, 3, figsize=(14, 8))
axs_hist = axs_hist.flatten()
axs_hist[0].hist(gas_loss_samples, bins=50, alpha=0.7, color='#e15759')
axs_hist[0].set_title('Gas Loss Distribution')
axs_hist[1].hist(electricity_loss_samples, bins=50, alpha=0.7, color='#59a14f')
axs_hist[1].set_title('Electricity Loss Distribution')
axs_hist[2].hist(gas_social_samples, bins=50, alpha=0.7, color='#f28e2b')
axs_hist[2].set_title('Gas Social Impact (normalized)')
axs_hist[3].hist(electricity_social_samples, bins=50, alpha=0.7, color='#9467bd')
axs_hist[3].set_title('Electricity Social Impact (normalized)')
axs_hist[4].hist(reward_samples, bins=50, alpha=0.7, color='#17becf')
axs_hist[4].set_title('Reward Distribution')
axs_hist[5].axis('off')
for ax in axs_hist[:5]:
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()