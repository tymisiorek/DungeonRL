# environment/grid_environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ComplexGridEnvironment(gym.Env):
    """
    A more complex grid environment compatible with Gymnasium.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5, num_obstacles=3, num_traps=2, has_key=True):
        super(ComplexGridEnvironment, self).__init__()
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.num_traps = num_traps
        self.has_key = has_key

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Actions: up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(grid_size, grid_size), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state and returns the initial observation.
        """
        if seed is not None:
            self.seed(seed)

        self.agent_position = [0, 0]
        self.goal_position = [self.grid_size - 1, self.grid_size - 1]
        self.goal_unlocked = not self.has_key
        self.previous_distance = self._calculate_distance(self.agent_position, self.goal_position)
        self.step_count = 0  # Reset step counter

        self.obstacles = self._generate_positions(self.num_obstacles)
        self.traps = self._generate_positions(self.num_traps, exclude=self.obstacles)

        if self.has_key:
            key_candidates = self._generate_positions(
                1, exclude=self.obstacles | self.traps | {tuple(self.goal_position)}
            )
            self.key_position = list(key_candidates)[0]
        else:
            self.key_position = None

        return self._get_state(), {}

    def step(self, action):
        """
        Executes an action and updates the agent's position and environment.
        """
        # Move the agent
        if action == 0 and self.agent_position[0] > 0:  # Up
            self.agent_position[0] -= 1
        elif action == 1 and self.agent_position[0] < self.grid_size - 1:  # Down
            self.agent_position[0] += 1
        elif action == 2 and self.agent_position[1] > 0:  # Left
            self.agent_position[1] -= 1
        elif action == 3 and self.agent_position[1] < self.grid_size - 1:  # Right
            self.agent_position[1] += 1

        # Initialize default reward and termination flags
        reward = -1
        terminated = False
        truncated = False

        # Check for obstacles
        current_pos = tuple(self.agent_position)
        if current_pos in self.obstacles:
            reward -= 10
            self.agent_position = [0, 0]

        # Check for traps
        elif current_pos in self.traps:
            reward -= 5

        # Check for key
        elif self.has_key and current_pos == self.key_position:
            reward += 10
            self.goal_unlocked = True
            self.key_position = None

        # Check for goal
        if current_pos == tuple(self.goal_position) and self.goal_unlocked:
            reward += 100
            terminated = True  # End the episode successfully

        # Distance-based reward
        current_distance = self._calculate_distance(self.agent_position, self.goal_position)
        reward += max(0, self.previous_distance - current_distance) * 5
        self.previous_distance = current_distance

        # Truncate episode after a set number of steps
        if self.step_count >= 100:  # Assuming step_limit is 100
            truncated = True

        self.step_count += 1  # Increment step counter
        return self._get_state(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        """
        Renders the current state of the environment.
        Returns:
            grid (np.ndarray): A 2D array representing the grid.
        """
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)

        for obs in self.obstacles:
            grid[obs] = 'X'
        for trap in self.traps:
            grid[trap] = 'T'
        if self.key_position:
            grid[self.key_position] = 'K'
        grid[tuple(self.goal_position)] = 'G'
        grid[tuple(self.agent_position)] = 'A'

        # Optionally still print if you want textual output
        # print("\n".join(" ".join(row) for row in grid))
        # print()

        return grid  # Return the grid for visualization

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        state[tuple(self.agent_position)] = 1
        state[tuple(self.goal_position)] = 2
        if self.key_position:
            state[self.key_position] = 3
        for obs in self.obstacles:
            state[obs] = -1
        for trap in self.traps:
            state[trap] = -0.5
        return state

    def _calculate_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _generate_positions(self, count, exclude=None):
        exclude = exclude or set()
        positions = set()
        while len(positions) < count:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in exclude and pos != tuple(self.agent_position):
                positions.add(pos)
        return positions
