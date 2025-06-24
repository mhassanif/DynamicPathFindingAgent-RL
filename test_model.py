import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from MazeGameEnv import MazeGameEnv
from time import sleep

# Define the maze
maze = [
    ['S', '.', '.', '.'],
    ['.', '#', 'P', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]

# Initialize the environment
env = DummyVecEnv([lambda: MazeGameEnv(maze)])  # Wrap the custom env in a DummyVecEnv

# Load the trained model
model = PPO.load("ppo_maze_model")
print("Model loaded!")

# Test the trained model
obs = env.reset()

for _ in range(1000):  # Run for a maximum of 1000 steps
    # Get the action from the trained model
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    sleep(1)  # Sleep for a short duration to visualize the environment

    # Render the environment
    env.envs[0].render()

    # Exit if the episode is done
    if done:
        print("Episode finished!")
        obs = env.reset()
