from MazeGameEnv import MazeGameEnv
from stable_baselines3 import PPO
import pygame

# Initialize the environment with a dynamic maze size
env = MazeGameEnv(size=6)

# Load the trained model
model = PPO.load("ppo_dynamic_maze_model")

# Test the model
num_episodes = 5
max_steps_per_episode = 50

for episode in range(num_episodes):
    print(f"Starting Episode {episode + 1}")
    obs, _ = env.reset()
    env.render()

    for step in range(max_steps_per_episode):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        # Use the trained model to predict actions
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step + 1}: Action taken: {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        print(f"Reward: {reward}, Info: {info['reason']}")

        if terminated or truncated:
            print(f"Episode {episode + 1} finished after {step + 1} steps!")
            break

        pygame.time.wait(200)

env.close()
