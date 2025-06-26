from stable_baselines3.common.env_checker import check_env
from MazeGameEnv import MazeGameEnv
from stable_baselines3 import PPO

# Instantiate the environment with a dynamic maze size
env = MazeGameEnv(size=6)

# Validate the environment
check_env(env, warn=True)

# Create the PPO model with MultiInputPolicy
model = PPO("MultiInputPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=500000)

# Save the model
model.save("ppo_dynamic_maze_model")

# Close the environment
env.close()
