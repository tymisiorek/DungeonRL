import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNetwork(nn.Module):
    """
    A simple feedforward neural network for DQN.
    """
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    DQN Agent for Deep Q-Learning.
    """
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1, buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Neural networks
        self.q_network = DQNetwork(state_dim, action_dim)
        self.target_network = DQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Copy weights
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Explore
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # Exploit

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        """
        Trains the Q-network using a batch of experiences.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data

        # Sample a batch of experiences
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Flatten states for input to the network
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self):
        """
        Updates the target network with the Q-network weights.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """
        Decays the exploration rate (epsilon).
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
