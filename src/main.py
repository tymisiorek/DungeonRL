# main.py

import matplotlib.pyplot as plt
import numpy as np
from agents.ppo_agent import PPOAgent
from environment.grid_environment import ComplexGridEnvironment
import seaborn as sns
import imageio  # Add imageio for GIF creation


def create_gif(model, env, output_file="agent_path.gif", episodes=5):
    """
    Creates a dynamic GIF of the agent's path across multiple episodes.

    Args:
        model: Trained PPO model.
        env: Environment to evaluate.
        output_file (str): Name of the output GIF file.
        episodes (int): Number of episodes to include in the GIF.
    """
    # Set DPI to ensure the figure size matches the expected pixel dimensions
    dpi = 100
    figsize = (6, 6)  # 6x6 inches

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ims = []  # Store frames for the GIF
    cumulative_visits = np.zeros((env.grid_size, env.grid_size))  # Track cumulative visits

    for episode in range(episodes):
        obs, _ = env.reset()  # Reset environment and extract observation
        done = False
        path = []  # Track the agent's path
        steps = 0
        max_steps = 1000  # Prevent infinite loops

        print(f"=== Starting Episode {episode + 1} ===")

        while not done and steps < max_steps:
            # Predict action using the model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update path and cumulative visits
            agent_pos = tuple(env.agent_position)
            path.append(agent_pos)
            cumulative_visits[agent_pos] += 1

            # Render the grid
            grid = env.render()
            ax.clear()
            ax.imshow(np.zeros((env.grid_size, env.grid_size)), cmap="gray", alpha=0.8)

            # Draw grid elements
            for i in range(env.grid_size):
                for j in range(env.grid_size):
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

            # Draw the cumulative heatmap
            ax.imshow(cumulative_visits, cmap="Reds", alpha=0.3, origin="upper")

            # Draw the current episode's path
            if len(path) > 1:
                xs, ys = zip(*path)
                ax.plot(ys, xs, marker='o', color='blue', linewidth=2, markersize=4)

            # Titles and ticks
            ax.set_title(f"Episode {episode + 1}, Step {steps + 1}")
            ax.set_xticks([])
            ax.set_yticks([])

            # Render the figure and append the frame
            fig.canvas.draw()

            # Capture the frame as RGB
            frame = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            width, height = fig.canvas.get_width_height()
            frame = frame.reshape(height, width, 3)
            ims.append(frame)

            steps += 1

        if steps >= max_steps:
            print(f"Episode {episode + 1} reached the maximum step limit.")
        print(f"Completed Episode {episode + 1} in {steps} steps.\n")

    plt.close()

    # Save the frames as a GIF using imageio
    imageio.mimsave(output_file, ims, fps=10)
    print(f"GIF saved as {output_file}")


if __name__ == "__main__":
    # Initialize PPO agent
    grid_size = 9
    total_timesteps = 20000
    agent = PPOAgent(grid_size=grid_size, total_timesteps=total_timesteps)

    # Train the PPO model
    agent.train()

    # Create the environment for visualization
    # If using a vectorized environment, access the underlying one
    if hasattr(agent.env, 'envs') and len(agent.env.envs) > 0:
        env = agent.env.envs[0].env  # Access the first (and only) environment
    else:
        env = agent.env

    # Generate a GIF of the agent's performance across multiple episodes
    create_gif(model=agent.model, env=env, output_file="agent_path.gif", episodes=12)
