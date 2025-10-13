from utils.environment_placeholder import TrialEnv
from stable_baselines3 import PPO
import numpy as np

# Define parameters
num_nodes = 8
years = 50
year_step = 1

# Example per-asset footprint areas (m^2); adjust to your assets if needed
area = np.array([100, 150, 150, 50, 50, 50, 200, 300], dtype=np.float32)

# Optional weights: [w_gas, w_electricity, w_gas_loss, w_gas_social, w_electricity_loss, w_electricity_social]
weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Create environment
env = TrialEnv(
    num_nodes=num_nodes,
    years=years,
    weights=weights,
    budget=100000,
    year_step=year_step,
    area=area,
    mc_samples=10000,           # reduce for speed during training
    csv_path='outputs/coastal_inundation_samples.csv',
    gpd_k=0.8019,
    gpd_sigma=0.1959,
    max_depth=8.0,
    threshold_depth=0.5,
    enforce_budget_normalization=True,
)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./log")
model.learn(total_timesteps=10000000)
model.save("policy")

# Evaluate the trained policy
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()