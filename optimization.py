from utils.environment_placeholder import TrialEnv
from stable_baselines3 import PPO
import numpy as np

# Define parameters
num_nodes = 8
years = 50
year_step = 1
RL_steps = 100000

# Example per-asset footprint areas (m^2); adjust to your assets if needed
area = np.array([100, 150, 150, 50, 50, 50, 200, 300], dtype=np.float32)

# Optional weights: [w_gas, w_electricity, w_gas_loss, w_gas_social, w_electricity_loss, w_electricity_social]
weights = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]

# Create environment
env = TrialEnv(
    num_nodes=num_nodes,
    years=years,
    weights=weights,
    budget=200000,
    year_step=year_step,
    area=area,
    mc_samples=10000,
    csv_path='outputs/coastal_inundation_samples.csv',
    gpd_k=0.8019,
    gpd_sigma=0.1959,
    max_depth=8.0,
    threshold_depth=0.5,
)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./log")
model.learn(total_timesteps=RL_steps)
model.save("results/policy")

# Evaluate the trained policy
actions_log = []
obs_log = []
rewards_log = []

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    actions_log.append(action)
    obs, reward, terminated, truncated, info = env.step(action)
    obs_log.append(obs)
    rewards_log.append(reward)
    done = terminated or truncated

import pandas as pd
df = pd.DataFrame({
    'action': actions_log,
    'obs': obs_log,
    'reward': rewards_log
})
df.to_csv("results/evaluation_logs.csv", index=False)

import matplotlib.pyplot as plt

# Histogram of actions
action_choices = []
for a in actions_log:
    arr = np.asarray(a)
    if arr.ndim == 0:
        action_choices.append(int(arr))
    else:
        action_choices.append(int(arr.ravel()[0]))
labels = ['No Investment', 'PV', 'Substation1', 'Substation2', 'Compressor1', 'Compressor2', 'Compressor3', 'ThermalUnit', 'LNG']
plt.figure()
plt.hist(action_choices, bins=range(len(labels)+1), align='left', rwidth=0.8)
plt.xticks(range(len(labels)), labels, rotation=45)
plt.xlabel('Action (Asset)')
plt.ylabel('Frequency')
plt.title('Action Distribution Over Evaluation Episode')
plt.tight_layout()

plt.savefig('results/action_histogram.png')
plt.close()

# Plot action over time
# Compute decision times assuming one decision per year_step
years_axis = [i * year_step for i in range(len(action_choices))]
plt.figure()
plt.plot(years_axis, action_choices, marker='o')
plt.yticks(range(len(labels)), labels, rotation=45)
plt.xlabel('Year')
plt.ylabel('Action (Asset)')
plt.title('Action Taken Over Time')
plt.tight_layout()
plt.savefig('results/action_over_time.png')
plt.close()

env.close()