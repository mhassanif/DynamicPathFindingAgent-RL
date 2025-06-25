from stable_baselines3.common.env_checker import check_env
from MazeGameEnv import MazeGameEnv
from stable_baselines3 import PPO

# Define your maze
maze = [
    ['S', '.', '.', 'G'],
    ['#', '#', '.', '#'],
    ['.', 'P', '.', '.'],
    ['#', '.', '.', '#']
]

# Instantiate the environment
env = MazeGameEnv(maze)

# Validate the environment
check_env(env, warn=True)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_maze_model")

# Close the environment
env.close()

