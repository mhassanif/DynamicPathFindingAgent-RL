from stable_baselines3 import PPO
from MazeGameEnv import MazeGameEnv

# Define your maze
maze = [
    ['S', '.', '.', 'G'],
    ['#', '#', '.', '#'],
    ['.', 'P', '.', '.'],
    ['#', '.', '.', '#']
]

# Instantiate the environment
env = MazeGameEnv(maze)

# Load the trained model
model = PPO.load("ppo_maze_model")

# Test the model
obs, _ = env.reset()
for step in range(100):  # Run for 100 steps
    action, _ = model.predict(obs, deterministic=True)  # Predict the best action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print(f"Episode finished: {info}")
        break

# Close the environment
env.close()
