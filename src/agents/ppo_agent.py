# agents/ppo_agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from environment.grid_environment import ComplexGridEnvironment

class PPOAgent:
    """
    Encapsulation of PPO training and evaluation.
    """

    def __init__(self, grid_size=5, total_timesteps=10000, num_envs=1):
        """
        Initializes the PPOAgent with a vectorized environment and training parameters.

        Args:
            grid_size (int): Size of the dungeon grid.
            total_timesteps (int): Total timesteps for training.
            num_envs (int): Number of parallel environments.
        """
        self.grid_size = grid_size
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs

        # Create a vectorized environment
        self.env = make_vec_env(lambda: ComplexGridEnvironment(grid_size=grid_size), n_envs=num_envs)

        # Initialize PPO model
        self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self):
        """
        Trains the PPO model.
        """
        print(f"Training PPO model for {self.total_timesteps} timesteps...")
        self.model.learn(total_timesteps=self.total_timesteps)
        print("Training complete.")

    def evaluate(self, episodes=10):
        """
        Evaluates the PPO model in the environment.

        Args:
            episodes (int): Number of evaluation episodes.

        Returns:
            cumulative_visits: A heatmap array of visit frequencies.
        """
        cumulative_visits = np.zeros((self.grid_size, self.grid_size))

        for episode in range(episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.env.step(action)
                agent_pos = tuple(self.env.envs[0].env.agent_position)
                cumulative_visits[agent_pos] += 1
                total_reward += reward

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        return cumulative_visits
