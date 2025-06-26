import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class MazeGameEnv(gym.Env):
    def __init__(self, size):
        super(MazeGameEnv, self).__init__()
        self.size = size  # size x size grid 
        self.num_obstacles = int(0.1 * size * size)  # 10% of cells
        self.num_pits = int(0.05 * size * size)  # 5% of cells

        self.start_pos = None
        self.goal_pos = None
        self.obstacle_positions = []
        self.pit_positions = []
        self.current_pos = None

        # 4 possible movments 
        self.action_space = spaces.Discrete(4)

        # current position normalized to [0, 1] regardless of maze size
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # Max steps before truncation
        self.max_steps = 200
        self.steps = 0

        # Initialize Pygame
        pygame.init()
        self.cell_size = 70
        self.screen = None

    def _generate_maze(self):
        self.start_pos = (0, 0)  # top-left corner
        self.goal_pos = (self.size - 1, self.size - 1)  # Bottom-right corner

        # Generate valid positions excluding start and goal
        valid_positions = [
            (r, c) for r in range(self.size) for c in range(self.size)
            if (r, c) not in [self.start_pos, self.goal_pos,(0,1),(1,0),(self.size - 2, self.size - 1),(self.size - 1, self.size - 2)]
        ]

        # Place obstacles
        self.obstacle_positions = random.sample(valid_positions, self.num_obstacles)

        # remaining positions for pits
        remaining_positions = [
            pos for pos in valid_positions if pos not in self.obstacle_positions
        ]
        self.pit_positions = random.sample(remaining_positions, self.num_pits)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Reset environment with a new seed
        self._generate_maze()
        self.current_pos = self.start_pos
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # Move the agent
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
            self.current_pos = tuple(new_pos)

        # Increment step count
        self.steps += 1

        # Calculate reward and termination conditions
        if self.current_pos == self.goal_pos:
            reward = 1.0
            terminated = True
            truncated = False
            info = {"reason": "Goal reached!"}
        elif self.current_pos in self.pit_positions:
            reward = -1.0
            terminated = True
            truncated = False
            info = {"reason": "Fell into a pit!"}
        elif self.steps >= self.max_steps:
            reward = 0.0
            terminated = True
            truncated = True
            info = {"reason": "Time limit reached"}
        else:
            reward = -0.01  # Step penalty
            terminated = False
            truncated = False
            info = {"reason": "Keep exploring"}

        return self._get_obs(), reward, terminated, truncated, info

    def _is_valid_position(self, pos):
        row, col = pos
        if row < 0 or col < 0 or row >= self.size or col >= self.size:
            return False
        if (row, col) in self.obstacle_positions:
            return False
        return True

    def _get_obs(self):
        return np.array(self.current_pos, dtype=np.float32) / (self.size - 1)

    def render(self):
        try:
            # Load and scale images
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
        except FileNotFoundError as e:
            print(f"Error loading images: {e}")
            return  # Skip rendering if images are unavailable

        # Initialize the screen if not already set
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))

        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Draw the environment
        for row in range(self.size):
            for col in range(self.size):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

                if (row, col) in self.obstacle_positions:
                    self.screen.blit(obstacle_img, (cell_left, cell_top))
                elif (row, col) in self.pit_positions:
                    self.screen.blit(pit_img, (cell_left, cell_top))
                elif (row, col) == self.start_pos:
                    self.screen.blit(start_img, (cell_left, cell_top))
                elif (row, col) == self.goal_pos:
                    self.screen.blit(goal_img, (cell_left, cell_top))

                # Draw the agent
                if np.array_equal(self.current_pos, [row, col]):
                    self.screen.blit(agent_img, (cell_left, cell_top))

        # Update the display
        pygame.display.update()


    def close(self):
        if pygame.get_init():
            pygame.quit()
