from stable_baselines3 import PPO
from MazeGameEnv import MazeGameEnv

# Define your maze
maze = [
    ['S', '.', '.', '.'],
    ['.', '#', 'P', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]

# Instantiate the environment
env = MazeGameEnv(maze)

# Load the trained model
model = PPO.load("ppo_maze_model")

# Test the model
obs, _ = env.reset()
for step in range(1000):  
    action, _ = model.predict(obs, deterministic=True)  
    print(f"Step {step}: Action taken: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print(f"Episode finished: {info}")
        obs, _ = env.reset()

# Close the environment
env.close()
