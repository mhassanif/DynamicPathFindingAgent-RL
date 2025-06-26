# **DynamicPathFindingAgent-RL**

## **Overview**

**DynamicPathFindingAgent-RL** is a custom Gymnasium environment designed for training reinforcement learning (RL) agents to navigate dynamically generated mazes using **Proximal Policy Optimization (PPO)**. This project emphasizes creating dynamic mazes with randomized layouts on every reset, allowing RL models to generalize better to unseen environments. The agent learns to traverse obstacles and avoid pitfalls to reach a designated goal.

The environment is fully compatible with **Stable-Baselines3 (SB3)**, providing seamless integration for training and testing RL models.

---

## **Features**

1. **Dynamic Maze Generation**:

   * On every environment reset, the maze layout is randomly generated, including:

     * **Obstacles**: Blocks that restrict movement.
     * **Pits**: Terminate the episode with penalties.
     * **Start and Goal Positions**: Randomly placed ensuring path feasibility.
   * Fully configurable maze size through a `size` parameter.

2. **SB3 Compatibility**:

   * The environment is validated with **SB3's `check_env`** tool, ensuring compatibility with SB3 algorithms such as PPO.

3. **Custom Observations**:

   * The agent receives a normalized position and visibility of neighboring cells (up, down, left, right).

4. **Graphical Rendering**:

   * The environment uses **Pygame** to render the maze, displaying obstacles, pits, start and goal positions, and the agent.

5. **Training and Testing**:

   * Leverages **Stable-Baselines3 PPO** for training.
   * Provides scripts to save and load trained models for evaluation.

---

## **Demo**

![Environment Screenshot](utils/screenshot.png)

**Watch the Demo Video**:
[Demo Video Link](https://drive.google.com/file/d/1maQVQ_X9GDguwgR0MXM91u65dXG7be1s/view?usp=sharing)

---

## **Project Structure**

* **`MazeGameEnv.py`**: Defines the custom Gymnasium environment with dynamic maze generation.
* **`train_model.py`**: Trains the agent using PPO.
* **`test_model.py`**: Tests the trained model in the dynamic maze environment.
* **`env_test.py`**: Validates the environment with a random agent for debugging.

---

## **Setup and Run**

### **1. Clone the Repository**

```bash
git clone https://github.com/mhassanif/DynamicPathFindingAgent-RL.git
cd DynamicPathFindingAgent-RL
```

### **2. Install Dependencies**

Install the required Python libraries:

```bash
pip install gymnasium stable-baselines3 pygame numpy
```

### **3. Validate the Environment (Optional)**

Run the `env_test.py` script to test the environment with a random agent:

```bash
python env_test.py
```

### **4. Train the Model**

Use the `train_model.py` script to train an agent with PPO:

```bash
python train_model.py
```

The script saves the trained model as `ppo_dynamic_maze_model`.

### **5. Test the Trained Model**

Run the `test_model.py` script to evaluate the trained model:

```bash
python test_model.py
```

This script loads the trained model (`ppo_dynamic_maze_model`) and tests it in the environment.

---

## **Using the Custom Environment**

The maze environment can be configured dynamically by specifying the maze size during initialization.

```python
from MazeGameEnv import MazeGameEnv

# Create a 6x6 maze environment
env = MazeGameEnv(size=6)

# Reset the environment (randomizes the maze)
obs, _ = env.reset()
```

**Parameters**:

* **`size`**: Integer defining the grid size (e.g., `size=6` creates a 6x6 maze).

---

## **Training Process**

The environment leverages **Proximal Policy Optimization (PPO)** for training:

* **Policy**: MultiInputPolicy to handle dictionary observations.
* **Reward Structure**:

  * **+1**: Reaching the goal.
  * **-1**: Falling into a pit.
  * **-0.01**: Per step penalty to encourage efficiency.
* **Termination Conditions**:

  * Agent reaches the goal.
  * Agent falls into a pit.
  * Maximum steps (`100`) reached.

```python
from stable_baselines3 import PPO

# Instantiate the environment
env = MazeGameEnv(size=6)

# Validate the environment
from stable_baselines3.common.env_checker import check_env
check_env(env, warn=True)

# Train the model
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=500000)

# Save the trained model
model.save("ppo_dynamic_maze_model")
```

---

## **Testing Process**

Evaluate the agent using the trained model:

```python
from MazeGameEnv import MazeGameEnv
from stable_baselines3 import PPO

# Load the trained model
model = PPO.load("ppo_dynamic_maze_model", policy="MultiInputPolicy")

# Initialize the environment
env = MazeGameEnv(size=6)

# Test the model
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
```

Enjoy experimenting with **DynamicPathFindingAgent-RL**! 🎮
