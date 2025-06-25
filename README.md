# **Custom Maze Game Environment for Reinforcement Learning**

## **Overview**

This project demonstrates the creation and utilization of a custom Gymnasium environment for Reinforcement Learning (RL). The environment simulates a maze with customizable elements such as obstacles, death-pits, a start position, and a goal. The agent learns to navigate the maze using **Stable-Baselines3 (SB3)** reinforcement learning algorithms.

The environment is fully compatible with SB3, allowing seamless integration for training and testing RL models.

---

## **Demo**

![Environment Screenshot](utils/screenshot.png)

**Watch the Demo Video**:
[Demo Video Link](https://drive.google.com/file/d/11j2hhNSLZNuGT9sk-3PGglSV1LwqpxXw/view?usp=sharing)

---

## **Features**

1. **Customizable Maze Layout**:

   * Define maze elements with:

     * `S`: Starting position
     * `G`: Goal position
     * `#`: Obstacles
     * `P`: Death-pits (penalize and terminate the episode)
     * `.`: Empty cells
2. **Dynamic Agent Behavior**:

   * The agent receives rewards for reaching the goal and penalties for inefficiency or stepping into a death-pit.
3. **Rendering with Pygame**:

   * Visualize the maze environment with graphical elements.
4. **Trained Models**:

   * Train RL models using PPO (Proximal Policy Optimization) or other algorithms in SB3.
   * Save and load models for reuse.
5. **Validation**:

   * The environment passes SB3's `check_env` validation for RL compatibility.

---

## **Project Structure**

* **`MazeGameEnv.py`**: Defines the custom environment.
* **`train_model.py`**: Script to train the model using PPO.
* **`test_model.py`**: Script to test the trained model in a maze.
* **`env_test.py`**: Script to run a random agent in the environment for debugging.

---

## **Dependencies**

Install the required Python libraries:

```bash
pip install gymnasium stable-baselines3 pygame numpy
```

---

## **Custom Maze Environment**

The maze is a 2D grid where each cell can have one of the following:

* `S`: Starting position for the agent.
* `G`: Goal cell with a positive reward.
* `#`: Obstacles that the agent cannot pass through.
* `P`: Death-pits that terminate the episode with a penalty.
* `.`: Empty space where the agent can move freely.

**Example Maze Configuration**:

```python
maze = [
    ['S', '.', '.', '.'],
    ['.', '#', 'P', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]
```

---

## **How to Use**

### **1. Environment Validation**

Validate the environment using SB3’s `check_env` utility to ensure compatibility:

```python
from stable_baselines3.common.env_checker import check_env
from MazeGameEnv import MazeGameEnv

maze = [
    ['S', '.', '.', '.'],
    ['.', '#', 'P', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]

env = MazeGameEnv(maze)
check_env(env, warn=True)
```

---

### **2. Training the Model**

Train the agent to solve the maze using PPO:

```python
from stable_baselines3 import PPO
from MazeGameEnv import MazeGameEnv

maze = [
    ['S', '.', '.', '.'],
    ['.', '#', 'P', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]

# Instantiate the environment
env = MazeGameEnv(maze)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_maze_model")

# Close the environment
env.close()
```

---

### **3. Testing the Model**

Test the trained model on the maze:

```python
from stable_baselines3 import PPO
from MazeGameEnv import MazeGameEnv

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
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        print(f"Episode finished: {info}")
        obs, _ = env.reset()

env.close()
```

---

### **4. Debugging with a Random Agent**

Run a random agent in the environment to debug:

```python
from MazeGameEnv import MazeGameEnv
import pygame

maze = [
    ['S', '.', '.', '.'],
    ['.', '#', 'P', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]

env = MazeGameEnv(maze)
obs, _ = env.reset()

for step in range(20):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        break

env.close()
```

---

## **Tips**

* Modify `max_steps` in `MazeGameEnv` for longer or shorter episodes.
* Train on diverse maze layouts to make the agent robust to variations.
* Use exploration during testing (`deterministic=False`) to discover alternative paths.
