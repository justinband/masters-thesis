import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_carbons(carbons, optimal_carbons):
    """
    Plot the carbon intensities incurred during training.

    Args:
        carbons: list of carbon intensities incurred during training
        optimal_carbons: optimal carbons based on other derivations
    """
    plot_styling()
    plt.figure(figsize=(10, 6))
    plt.plot(carbons, alpha=0.5, label='Agent Carbon')
    plt.plot(optimal_carbons, alpha=0.5, label='Optimal Carbon')
    plt.xlabel("Episode")
    plt.ylabel('Total Carbon')
    plt.title('Carbon incurred during training')

    window_size = min(100, len(carbons)//10)
    smoothed_carbon = np.convolve(carbons, np.ones(window_size)/window_size, mode='valid')
    smoothed_opt_carbon = np.convolve(optimal_carbons, np.ones(window_size)/window_size, mode='valid')

    plt.plot(range(window_size-1, len(carbons)), smoothed_carbon, 'r-', linewidth=2, label='Average Agent Carbon')
    plt.plot(range(window_size-1, len(optimal_carbons)), smoothed_opt_carbon, 'g-', linewidth=2, label='Average Optimal Carbon')

    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    return plt.gcf()

def plot_training_progress(losses):
    """
    Plot the training progress (losses per episode).

    Args:
        losses: list of losses from training
    """
    plot_styling()
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    
    # Add a trend line
    window_size = min(10, len(losses)//10)
    smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(losses)), smoothed, 'r-', linewidth=2)
    
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    return plt.gcf()

def plot_regret(regrets):
    """
    Plot the training regret and the cumulative regret

    Args:
        regrets: regrets per episode
    """
    plot_styling()
    cumulative_regret = np.cumsum(regrets)

    # plt.figure(figsize=(10, 6))
    # plt.plot(regrets)
    # plt.xlabel('Episode')
    # plt.ylabel('Regret')
    # plt.title('Training Regret')

    # plt.grid(True)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_regret, label='Cumulative Regret')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Regret')
    plt.title('Training Cumulative Regret')

    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    return plt.gcf()

def plot_evaluation_results(actions, intensities, losses, q_vals):
    """
    Plots details for an evaluation. Notably, the actions taken at intervals,
    with corresponding intensities, losses, and q-values.

    Args:
        actions: list of actions taken by the agent
        intensities: list of intensities observed by the agent
        losses: list of losses incurred by the agent
        q_vals: q-values of the model as actions are executed
    """
    plot_styling()
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Plot carbon intensity
    run_indices = np.flatnonzero(actions)
    intensities = np.array(intensities)
    axes[0].plot(intensities, 'g-')
    axes[0].plot(run_indices, intensities[run_indices], 'ro', label='Run Actions')
    axes[0].set_title('Carbon Intensity')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot actions (Run/Pause)
    axes[1].plot(actions, 'bo-', drawstyle='steps-post')
    axes[1].set_title('Actions (0=Pause, 1=Run)')
    axes[1].set_ylabel('Action')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True)
    
    # Plot losses
    losses = np.array(losses)
    axes[2].plot(losses, 'b-')
    axes[2].plot(run_indices, losses[run_indices], 'ro', label='Run Actions')
    axes[2].set_title('Losses')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot Q-values
    pause_q = [q[0] for q in q_vals]
    run_q = [q[1] for q in q_vals]
    axes[3].plot(pause_q, 'c-', label='Pause Q-value')
    axes[3].plot(run_q, 'm-', label='Run Q-value')
    axes[3].set_title('Q-Values')
    axes[3].set_xlabel('Time Step')
    axes[3].set_ylabel('Q-Value')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    return plt.gcf()

def visualize_weights(feature_names, approximators):
    """
    Plot weight importance of a model's approximators

    Args:
        feature_names: list of the features names
        approximators: all function approximators, these must contain the weights
    """
    plot_styling()
    plt.figure(figsize=(12, 10))

    for i, action_name in enumerate(["Pause", "Run"]):
        plt.subplot(2, 1, i+1)
        weights = approximators[i].weights
        abs_weights = np.abs(weights)
        sorted_idx = np.argsort(abs_weights)[::-1] # Sort by absolute weight magnitude

        plt.bar(range(len(weights)), abs_weights[sorted_idx], color='skyblue')
        plt.xticks(range(len(weights)), [feature_names[j] for j in sorted_idx], rotation=45)
        plt.title(f"Feature Importance for {action_name} Action")
        plt.ylabel("Absolute Weight Value")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    return plt.gcf()

def plot_styling():
    sns.set_theme(style='darkgrid')