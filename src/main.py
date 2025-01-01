from environment.grid_environment import ComplexGridEnvironment
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def visualize_extended_dungeon_gif(grid_size=5, episodes=500, visualize_every=50, output_file="extended_dungeon.gif", use_dqn=False):
    """
    Visualizes the agent navigating the dungeon and creates a GIF of the agent's performance.

    Args:
        grid_size (int): Size of the dungeon grid.
        episodes (int): Number of episodes to train the agent.
        visualize_every (int): Save a frame every `visualize_every` episodes.
        output_file (str): Name of the output GIF file.
        use_dqn (bool): Whether to use DQNAgent or QLearningAgent.
    """
    env = ComplexGridEnvironment(grid_size=grid_size, num_obstacles=3, num_traps=2, has_key=True)
    if use_dqn:
        agent = DQNAgent(
            state_dim=grid_size * grid_size,  # Flattened grid as input
            action_dim=4,  # Four actions (up, down, left, right)
        )
    else:
        agent = QLearningAgent(
            state_space=(grid_size, grid_size),
            action_space=4
        )

    frames = []  # Store frames for the GIF
    success_count = 0  # Count successful episodes
    failed_episodes = []  # Track episodes that fail to reach the goal
    step_limit = 100  # Maximum steps per episode

    for episode in range(episodes):
        state = env.reset()
        done = False
        path = []

        for step in range(step_limit):
            # Flatten state for DQN
            # flat_state = env._get_state()  # Already flattened by ComplexGridEnvironment
            flat_state = np.ravel(env._get_state()) if use_dqn else env._get_state()

            path.append(tuple(env.agent_position))
            action = agent.select_action(flat_state)
            next_state, reward, done = env.step(action)

            # Flatten next_state for DQN
            flat_next_state = np.ravel(next_state) if use_dqn else next_state

            if use_dqn:
                agent.store_experience(flat_state, action, reward, flat_next_state, done)
                agent.train()
            else:
                agent.update(flat_state, action, reward, flat_next_state)

            state = next_state

            if done:
                success_count += 1
                break
        else:
            failed_episodes.append(episode)
            if len(failed_episodes) % 10 == 0:
                print(f"{len(failed_episodes)} episodes reached step limit.")

        agent.decay_epsilon()

        if (episode + 1) % visualize_every == 0 or episode == episodes - 1:
            try:
                path.append(tuple(env.agent_position))  # Append the final position
                grid = env.render(path=path)

                # Plot the grid
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(np.zeros((grid_size, grid_size)), cmap="gray", alpha=0.8)
                for i in range(grid_size):
                    for j in range(grid_size):
                        cell = grid[i][j]
                        if cell == 'A':
                            ax.text(j, i, 'A', ha='center', va='center', fontsize=16, color='red')
                        elif cell == 'G':
                            ax.text(j, i, 'G', ha='center', va='center', fontsize=16, color='green')
                        elif cell == 'K':
                            ax.text(j, i, 'K', ha='center', va='center', fontsize=12, color='gold')
                        elif cell == 'T':
                            ax.text(j, i, 'T', ha='center', va='center', fontsize=12, color='purple')
                        elif cell == 'X':
                            ax.text(j, i, 'X', ha='center', va='center', fontsize=12, color='black')
                        elif cell == 'o':
                            ax.text(j, i, 'o', ha='center', va='center', fontsize=10, color='blue')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Episode {episode + 1}")
                plt.tight_layout()

                # Save the current frame
                plt.savefig("frame.png")
                plt.close()

                # Load the frame and append to frames list
                try:
                    frame = Image.open("frame.png")
                    frames.append(frame.copy())  # Ensure a copy is stored to prevent truncation
                    frame.close()
                except Exception as e:
                    print(f"Error loading frame for episode {episode + 1}: {e}")
            except Exception as e:
                print(f"Error plotting frame for episode {episode + 1}: {e}")

    # Create and save the GIF
    try:
        if frames:
            frames[0].save(
                output_file,
                save_all=True,
                append_images=frames[1:],
                duration=300,
                loop=0
            )
            print(f"GIF saved as {output_file}")
        else:
            print("No valid frames to create a GIF.")
    except Exception as e:
        print(f"Error creating GIF: {e}")

    success_rate = (success_count / episodes) * 100
    print(f"Success Rate: {success_rate:.2f}% ({success_count}/{episodes} episodes)")

if __name__ == "__main__":
    visualize_extended_dungeon_gif(
        grid_size=5,
        episodes=500,
        visualize_every=50,
        output_file="extended_dungeon.gif",
        use_dqn=True  # Set to False for QLearningAgent
    )
