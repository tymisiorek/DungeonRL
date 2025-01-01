import numpy as np

class ComplexGridEnvironment:
    """
    A more complex grid environment with obstacles, traps, and keys.
    """
    def __init__(self, grid_size=5, num_obstacles=3, num_traps=2, has_key=True):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.num_traps = num_traps
        self.has_key = has_key

        self.obstacles = set()
        self.traps = set()
        self.key_position = None
        self.goal_unlocked = not self.has_key  # Initially locked if a key is required
        self.reset()

    def reset(self):
        """
        Resets the environment and places obstacles, traps, key, and goal.
        """
        self.agent_position = [0, 0]
        self.goal_position = [self.grid_size - 1, self.grid_size - 1]
        self.goal_unlocked = not self.has_key

        # Place obstacles
        self.obstacles = self._generate_positions(self.num_obstacles)

        # Place traps
        self.traps = self._generate_positions(self.num_traps, exclude=self.obstacles)

        # Place the key
        if self.has_key:
            key_candidates = self._generate_positions(
                1,
                exclude=self.obstacles | self.traps | {tuple(self.goal_position)}
            )
            self.key_position = list(key_candidates)[0]
        else:
            self.key_position = None

        return self._get_state()

    def _generate_positions(self, count, exclude=None):
        """
        Generates random positions on the grid, avoiding excluded positions.
        """
        exclude = exclude or set()
        positions = set()
        while len(positions) < count:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in exclude and pos != tuple(self.agent_position):
                positions.add(pos)
        return positions

    def _get_state(self):
        """
        Returns the current state of the environment: a flattened representation of the grid.
        """
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        state[tuple(self.agent_position)] = 1  # Mark agent position
        state[tuple(self.goal_position)] = 2  # Mark goal position
        if self.key_position:
            state[tuple(self.key_position)] = 3  # Mark key position
        for obs in self.obstacles:
            state[obs] = -1  # Mark obstacles
        for trap in self.traps:
            state[trap] = -0.5  # Mark traps
        return state.flatten()  # Return a flattened array

    def step(self, action):
        """
        Executes an action and updates the agent's position and environment.

        Args:
            action (int): The action to perform (0 = up, 1 = down, 2 = left, 3 = right).

        Returns:
            tuple: The next state, reward, and a boolean indicating if the episode is done.
        """
        # Move the agent
        if action == 0 and self.agent_position[0] > 0:          # Up
            self.agent_position[0] -= 1
        elif action == 1 and self.agent_position[0] < self.grid_size - 1:  # Down
            self.agent_position[0] += 1
        elif action == 2 and self.agent_position[1] > 0:         # Left
            self.agent_position[1] -= 1
        elif action == 3 and self.agent_position[1] < self.grid_size - 1:  # Right
            self.agent_position[1] += 1

        current_pos = tuple(self.agent_position)
        reward = -1  # Default penalty for moving

        # Check for obstacles
        if current_pos in self.obstacles:
            reward -= 10  # Penalty for hitting an obstacle
            self.agent_position = [0, 0]  # Reset to start

        # Check for traps
        elif current_pos in self.traps:
            reward -= 5  # Extra penalty for stepping on a trap

        # Check for key
        elif self.has_key and current_pos == self.key_position:
            reward += 5  # Reward for collecting the key
            self.goal_unlocked = True
            self.key_position = None  # Remove the key from the grid

        # Check for goal
        done = (current_pos == tuple(self.goal_position) and self.goal_unlocked)
        if done:
            reward += 10  # Big reward for reaching the goal

        return self._get_state(), reward, done

    def render(self, path=None):
        """
        Returns a visual representation of the current grid state.

        Args:
            path (list of tuples): The path taken by the agent.

        Returns:
            np.ndarray: 2D array representing the grid.
        """
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)

        # Place elements on the grid
        for obs in self.obstacles:
            grid[obs] = 'X'  # Obstacles
        for trap in self.traps:
            grid[trap] = 'T'  # Traps
        if self.key_position:
            grid[self.key_position] = 'K'  # Key
        grid[tuple(self.goal_position)] = 'G'  # Goal

        # Mark the path
        if path:
            for pos in path:
                if pos != tuple(self.agent_position) and pos != tuple(self.goal_position):
                    grid[pos] = 'o'  # Path

        # Place the agent
        grid[tuple(self.agent_position)] = 'A'

        return grid