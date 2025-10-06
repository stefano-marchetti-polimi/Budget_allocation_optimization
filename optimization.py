from environment_placeholder import TrialEnv
from stable_baselines3 import PPO
import numpy as np

# Define utility functions
def util_econ(x):
    # Example: minimize economic impact (linear)
    return -x

def util_soc(x):
    # Example: nonlinear utility, penalize large impacts more
    return -np.log1p(x)

# Define parameters
num_nodes = 5
years = 50
year_step = 1
weight_econ = 0.6
weight_soc = 0.4

# Create environment
env = TrialEnv(
    num_nodes=num_nodes,
    years=years,
    weight_econ=weight_econ,
    weight_soc=weight_soc,
    util_econ=util_econ,
    util_soc=util_soc,
    alpha=0.2,
    decay=0.05,
    budget=100000.0,
    year_step=year_step,
    seed=42
)

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./log")
model.learn(total_timesteps=20000)
model.save("policy")

# Evaluate the trained policy
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

env.close()