import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from MazeGameEnv import MazeGameEnv

# Define the maze
maze = [
    ['S', '.', '.', '.'],
    ['.', '#', 'P', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]

# Initialize the environment
env = DummyVecEnv([lambda: MazeGameEnv(maze)])  # Wrap the custom env in a DummyVecEnv

# Train the PPO model
model = PPO("MlpPolicy", env, verbose=1)
print("Training started...")
model.learn(total_timesteps=10000)
print("Training finished!")

# Save the trained model
model.save("ppo_maze_model")
print("Model saved as ppo_maze_model.zip")
