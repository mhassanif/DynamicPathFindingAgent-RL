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
    ['.', '#', 'P', '#'],
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
    obs, reward, done, info = env.step(action)
    env.render()
    print('__________________')
    print('Reward:', reward)
    print('Done:', done)
    print('Info:', info['reason'])

    pygame.time.wait(200)

env.close()
