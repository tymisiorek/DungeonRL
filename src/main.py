from environment.grid_environment import GridEnvironment
from agents.q_learning_agent import QLearningAgent
import matplotlib.pyplot as plt

def train_q_learning(grid_size=8, episodes=1000):
    """
    Trains a Q-Learning agent in a grid environment.

    Args:
        grid_size (int): The size of the grid. Default is 5.
        episodes (int): The number of training episodes. Default is 500.

    Returns:
        QLearningAgent: The trained Q-learning agent.
        GridEnvironment: The grid environment used for training.
        list: Total rewards per episode for visualization.
    """
    env = GridEnvironment(grid_size=grid_size)
    agent = QLearningAgent(
        state_space=(grid_size, grid_size),
        action_space=4
    )

    rewards = []  # Track total rewards per episode

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)

        if(episode % 100 == 0):
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    return agent, env, rewards


if __name__ == "__main__":
    agent, env, rewards = train_q_learning()

    # Plot the learning curve
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.show()

    print("Training Complete. Final Q-Table:")
    print(agent.q_table)
    env.render()
