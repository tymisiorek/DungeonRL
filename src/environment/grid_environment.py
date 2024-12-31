import numpy as np

class GridEnvironment:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.agent_position = [0, 0]
        self.goal_position = [self.grid_size - 1, self.grid_size - 1]
        return self._get_state()

    def _get_state(self):
        return tuple(self.agent_position)

    def step(self, action):
        #Actions: 0 = up, 1 = down, 2 = left, 3 = right
        if action == 0 and self.agent_position[0] > 0:
            self.agent_position[0] -= 1
        elif action == 1 and self.agent_position[0] < self.grid_size - 1:
            self.agent_position[0] += 1
        elif action == 2 and self.agent_position[1] > 0:
            self.agent_position[1] -= 1
        elif action == 3 and self.agent_position[1] < self.grid_size - 1:
            self.agent_position[1] += 1

        reward = -1
        done = self.agent_position == self.goal_position
        if done:
            reward = 10

        return self._get_state(), reward, done

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        grid[tuple(self.agent_position)] = 'A'
        grid[tuple(self.goal_position)] = 'G'
        print("\n".join(" ".join(row) for row in grid))
        print()