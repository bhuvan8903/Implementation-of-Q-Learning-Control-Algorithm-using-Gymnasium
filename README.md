# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium
`FrozenLake-v1` environment and learn an optimal action-value function that
enables the agent to select suitable actions for reaching the goal state while
avoiding holes.

---

## Problem Statement

Implement a **model-free Q-Learning control algorithm** using the
Gymnasium `FrozenLake-v1` environment.

The objective is to train an agent that learns the optimal action-value
function `Q(s,a)` through repeated interaction with the environment.

The agent should:

- Observe the current state.
- Select an action using an epsilon-greedy strategy.
- Receive a reward from the environment.
- Observe the next state.
- Update the Q-table using the Q-Learning update rule.
- Gradually reduce exploration during training.
- Learn a policy that attempts to reach the goal while avoiding holes.
- Display the final Q-table.
- Display the estimated state-value function.
- Display the learned policy.
- Plot the learning curve.
- Calculate the average reward obtained over the last 1000 episodes.

---

## Software Requirements

### Programming Language

- Python 3

### Libraries

The following Python libraries are required:

```text
gymnasium
numpy
matplotlib
````

Install the required packages using:

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

The experiment uses the **FrozenLake-v1** environment provided by
Gymnasium.

FrozenLake is a grid-world reinforcement learning environment in which the
agent moves across a frozen lake.

The environment contains:

* **Start state (S)** – The initial position of the agent.
* **Frozen states (F)** – Safe states where the agent can move.
* **Hole states (H)** – Dangerous states that terminate the episode.
* **Goal state (G)** – The destination the agent attempts to reach.

The environment uses a **4 × 4 grid**, giving a total of:

```text
16 states
```

There are four possible actions:

| Action | Meaning |
| ------ | ------- |
| `0`    | Left    |
| `1`    | Down    |
| `2`    | Right   |
| `3`    | Up      |

The environment is created using:

```python
env = gym.make("FrozenLake-v1", is_slippery=True)
```

The `is_slippery=True` setting makes the environment stochastic, meaning that
the agent may not always move exactly in the intended direction.

This makes the learning problem more challenging and allows Q-Learning to
learn from interaction with a stochastic environment.

---

## Theory

### Q-Learning

Q-Learning is a **model-free reinforcement learning control algorithm** that
learns the optimal action-value function directly.

The action-value function `Q(s,a)` represents the expected return obtained
when the agent takes action `a` in state `s` and subsequently follows the
best possible policy.

The Q-Learning update rule is:

```math
Q(S_t,A_t) \leftarrow Q(S_t,A_t) +
\alpha [R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)]
```

Where:

| Symbol               | Meaning                                   |
| -------------------- | ----------------------------------------- |
| `S_t`                | Current state                             |
| `A_t`                | Current action                            |
| `R_{t+1}`            | Reward received after taking action `A_t` |
| `S_{t+1}`            | Next state                                |
| `α`                  | Learning rate                             |
| `γ`                  | Discount factor                           |
| `Q(s,a)`             | Action-value function                     |
| `max_a Q(S_{t+1},a)` | Maximum action value in the next state    |

### Learning Rate

The learning rate `α` determines how strongly newly obtained information
changes the existing Q-value.

A larger learning rate gives more importance to newly observed information.

In this implementation:

```text
α = 0.8
```

### Discount Factor

The discount factor `γ` determines how much importance is given to future
rewards.

In this implementation:

```text
γ = 0.95
```

A value close to `1` encourages the agent to consider future rewards.

### Q-Learning Target

For a non-terminal next state, the target is:

```math
R_{t+1} + \gamma \max_a Q(S_{t+1},a)
```

For a terminal state, there is no future state value, so the target becomes:

```math
R_{t+1}
```

---

## Epsilon-Greedy Action Selection

During training, the agent uses **epsilon-greedy action selection**.

With probability `ε`, the agent explores by selecting a random action.

With probability `1 − ε`, the agent exploits by selecting the action with
the highest Q-value.

```math
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
```

The initial exploration rate is:

```text
ε = 1.0
```

The minimum exploration rate is:

```text
ε_min = 0.01
```

The exploration rate is gradually reduced using:

```text
ε = max(ε_min, ε × ε_decay)
```

In this implementation:

```text
ε_decay = 0.995
```

This allows the agent to explore different actions during the early stages
of training and gradually exploit the learned Q-values.

---

## Algorithm

### Q-Learning Algorithm

1. Create the `FrozenLake-v1` environment.
2. Determine the number of states and actions.
3. Initialize the Q-table with zeros.
4. Set the learning rate `α`.
5. Set the discount factor `γ`.
6. Initialize the exploration rate `ε`.
7. Repeat for the specified number of episodes:

   * Reset the environment.
   * Obtain the initial state.
   * Select an action using the epsilon-greedy strategy.
   * Execute the action.
   * Observe the reward and next state.
   * Calculate the Q-Learning target.
   * Update the Q-value.
   * Move to the next state.
   * Continue until the episode terminates.
   * Store the total reward.
   * Reduce epsilon.
8. Calculate the state-value function from the Q-table.
9. Extract the greedy policy from the Q-table.
10. Display the final Q-table.
11. Display the estimated state-value function.
12. Display the learned policy.
13. Calculate the average reward over the last 1000 episodes.
14. Plot the learning curve.
15. Close the environment.

---

## Python Program

```python
# -------------------------------------------------
# Imports
# -------------------------------------------------

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=True)

n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

learning_rate = 0.8
gamma = 0.95

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

num_episodes = 20000


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def choose_action(state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()

    return int(np.argmax(Q[state]))


# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    state, _ = env.reset()

    total_reward = 0

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Select action using epsilon-greedy strategy
        action = choose_action(state, epsilon)

        # Take action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Calculate Q-Learning target
        if terminated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        # Update Q-value
        Q[state, action] += learning_rate * (
            target - Q[state, action]
        )

        # Move to next state
        state = next_state

        total_reward += reward

    # Store total reward of the episode
    episode_rewards.append(total_reward)

    # Reduce exploration gradually
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# State-Value Function and Learned Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [
            action_symbols[action]
            for action in policy
        ]
    ).reshape(4, 4)

    print("\nLearned Policy:")

    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")

print(np.round(Q, 3))

print_value_function(state_values)

print_policy(learned_policy)

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    average_reward
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "Q-Learning Curve - FrozenLake"
)

plt.grid(True)

plt.show()

env.close()
```

---

## Output

```text
Name : Bhuvaneshwaran H
REG NO: 212223240018

Final Q-table:
[[0.085 0.055 0.114 0.02 ]
 [0.004 0.003 0.001 0.02 ]
 [0.004 0.006 0.006 0.006]
 [0.001 0.    0.001 0.006]
 [0.231 0.068 0.003 0.052]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.003 0.128 0.002 0.415]
 [0.008 0.708 0.006 0.006]
 [0.524 0.005 0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.007 0.124 0.885 0.077]
 [0.152 0.991 0.079 0.203]
 [0.    0.    0.    0.   ]]

Estimated State-Value Function:
[[0.114 0.02  0.006 0.006]
 [0.231 0.    0.    0.   ]
 [0.415 0.708 0.524 0.   ]
 [0.    0.885 0.991 0.   ]]

Learned Policy:
[['R' 'U' 'U' 'U']
 ['L' 'L' 'D' 'L']
 ['U' 'D' 'L' 'L']
 ['L' 'R' 'D' 'L']]

Average reward over last 1000 episodes: 0.48
---
```
<img width="699" height="468" alt="image" src="https://github.com/user-attachments/assets/31325dc1-610e-4b05-99f1-57f917c2fed9" />



### Result

The Q-Learning algorithm was successfully implemented in FrozenLake-v1, learning Q-values through repeated interaction using Q-table updates and an epsilon-greedy strategy.
The learned Q-table produced the state-value function and greedy policy, while the learning curve and final average reward were used to evaluate performance.

### Inference

The experiment demonstrates that Q-Learning is a model-free control algorithm that learns a suitable policy through trial and error without requiring a predefined environment model.
Through exploration and exploitation, the agent improves its Q-values and learns to reach the goal while avoiding holes in the stochastic FrozenLake environment.
