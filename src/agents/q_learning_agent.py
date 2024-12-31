import numpy as np
import random

class QLearningAgent:
    """
    A Q-Learning agent for reinforcement learning.

    Attributes:
        q_table (ndarray): The Q-table storing state-action values.
        alpha (float): The learning rate.
        gamma (float): The discount factor for future rewards.
        epsilon (float): The exploration rate.
        epsilon_decay (float): The rate at which epsilon decays.
        action_space (int): The number of possible actions.

    Methods:
        select_action(state): Selects an action using epsilon-greedy strategy.
        update(state, action, reward, next_state): Updates the Q-table based on the experience.
        decay_epsilon(): Reduces the exploration rate (epsilon).
    """
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99):
        self.q_table = np.zeros((*state_space, action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1) 
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Bellman equation.

        Args:
            state (tuple): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (tuple): The next state after the action.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
