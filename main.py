import gymnasium as gym
from MazeGameEnv import MazeGameEnv
import pygame

# Register the environment
gym.register(
    id='MazeGame-v0',
    entry_point=MazeGameEnv,
    kwargs={'maze': None}
)

# Maze configuration
maze = [
    ['S', '.', '.', '.'],
    ['.', '#', '.', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]

# Test the environment
env = MazeGameEnv(maze)
obs = env.reset()
env.render()

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()

    action = env.action_space.sample()  # Random action selection
    obs, reward, done, _ = env.step(action)
    env.render()
    print('Reward:', reward)
    print('Done:', done)

    pygame.time.wait(200)

env.close()
