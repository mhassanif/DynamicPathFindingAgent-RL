import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class MazeGameEnv(gym.Env):
    def __init__(self, maze):
        super(MazeGameEnv, self).__init__()
        self.maze = np.array(maze)  # Maze represented as a 2D numpy array
        self.start_pos = np.where(self.maze == 'S')  # Starting position
        self.goal_pos = np.where(self.maze == 'G')  # Goal position
        self.current_pos = self.start_pos #starting position is current posiiton of agent
        self.num_rows, self.num_cols = self.maze.shape
        

        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)  

        # Observation space is grid of size:rows x columns
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))

        # Initialize Pygame
        pygame.init()
        self.cell_size = 125

        # setting display size
        self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action):
        # Move the agent based on the selected action
        new_pos = np.array(self.current_pos)
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos

        # Reward function
        if np.array_equal(self.current_pos, self.goal_pos):
            reward = 1.0
            done = True
            info = {"reason": "Goal reached!"}
        elif self.death_pit(self.current_pos):
            reward = -1.0
            done = True
            info = {"reason": "Fell into death pit!"}
        else:
            reward = 0.0
            done = False
            info = {"reason": "Continue exploring"}

        return self.current_pos, reward, done, info

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.maze[row, col] == '#':
            return False
        return True
    
    def death_pit(self,pos):
        row, col = pos
        if self.maze[row, col] == 'P':
            return True

    def render(self):
        # Load the PNG images (do this once, not every frame ideally)
        obstacle_img = pygame.image.load("utils/obstacle.png")
        obstacle_img = pygame.transform.scale(obstacle_img, (self.cell_size, self.cell_size))

        start_img = pygame.image.load("utils/start.png")
        start_img = pygame.transform.scale(start_img, (self.cell_size, self.cell_size))

        goal_img = pygame.image.load("utils/goal.png")
        goal_img = pygame.transform.scale(goal_img, (self.cell_size, self.cell_size))

        pit_img = pygame.image.load("utils/fire.png")
        pit_img = pygame.transform.scale(pit_img, (self.cell_size, self.cell_size))

        agent_img = pygame.image.load("utils/spider.png")
        agent_img = pygame.transform.scale(agent_img, (self.cell_size, self.cell_size))

        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Draw the environment
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

                if self.maze[row, col] == '#':  # Obstacle
                    self.screen.blit(obstacle_img, (cell_left, cell_top))
                elif self.maze[row, col] == 'S':  # Starting position
                    self.screen.blit(start_img, (cell_left, cell_top))
                elif self.maze[row, col] == 'G':  # Goal position
                    self.screen.blit(goal_img, (cell_left, cell_top))
                elif self.maze[row, col] == 'P':  # Pit
                    self.screen.blit(pit_img, (cell_left, cell_top))

                # Agent position
                if np.array_equal(np.array(self.current_pos), np.array([row, col]).reshape(-1, 1)):
                    self.screen.blit(agent_img, (cell_left, cell_top))

        # Update the display
        pygame.display.update()
