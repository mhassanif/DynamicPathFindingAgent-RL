import pygame
from MazeGameEnv import MazeGameEnv

# Initialize the environment with a dynamic maze size
env = MazeGameEnv(size=6)

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

        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        print(f"Step: {step + 1}")
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Info: {info['reason']}")
        print("------------------")

        if terminated or truncated:
            print(f"Episode {episode + 1} finished after {step + 1} steps!")
            break

        pygame.time.wait(200)

env.close()
