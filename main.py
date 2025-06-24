import gymnasium as gym
from  MazeGameEnv import MazeGameEnv
import pygame

# Register the environment
gym.register(
    id='MazeGame-v0',
    entry_point='mazegame:MazeGameEnv', 
    kwargs={'maze': None} 
)


#Maze config

maze = [
    ['S', '', '.', '.'],
    ['.', '#', '.', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]
# Test the environment
env = gym.make('MazeGame-v0',maze=maze)
obs = env.reset()
env.render()

done = False
while True:
    pygame.event.get()
    action = env.action_space.sample()  # Random action selection
    obs, reward, done, _ = env.step(action)
    env.render()
    print('Reward:', reward)
    print('Done:', done)

    pygame.time.wait(200)
    

