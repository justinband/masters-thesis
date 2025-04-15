import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from collections import deque
from tqdm import tqdm
from algorithms import utils
from mdps import JobEnv
from collections import deque
from datasets import DataLoader

class LinearFunctionApproximator:
    def __init__(self, feature_dim):
        self.weights = np.random.randn(feature_dim) * 0.01

    def predict(self, features):
        return np.dot(features, self.weights)
    
    def update(self, features, error, lr, ep):
        self.weights += lr * error * features

class LinearQLearning():
    def __init__(self, env: JobEnv, lr=0.01, epsilon = 1, tradeoff=1.0):
        self.env = env
        self.tradeoff = tradeoff
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        self.max_time = 500 # TODO: Fix this. Maybe in relation to upper-bound?

        self.state_dim = 3
        self.feature_dim = 5
        self.action_dim = [0, 1]

        # Use one for each action because it's a clear delineation
        self.approximators = [
            LinearFunctionApproximator(self.feature_dim),
            LinearFunctionApproximator(self.feature_dim)
        ]

        self.memory = deque(maxlen=1000)
        self.batch_size = 32

    def _get_features(self, state):
        job_state, time, carbon = state

        # Basic features
        features = [
            # 1.0,  # Bias term
            job_state,          # Normalized
            time,                   # Normalized
            carbon,
            job_state * carbon, # Interaction between job state and carbon. How far job is and how 'bad' it is to run now
            # time * carbon,       # Interaction between time and carbon
            # job_state * time,    # Interaction between job state and time
            # job_state**2,        # Quadratic terms
            # time**2,
            # carbon**2,
            np.exp(-job_state),  # Non-linear transformations
            # np.log(job_state + 1)
        ]
        
        return np.array(features)

    def _choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            features = self._get_features(state)
            q_values = [approx.predict(features) for approx in self.approximators] # FIXME: This is [nan, nan]
            return np.argmin(q_values)
    
    def _create_state(self, intensity):
        normal_job_state = self.env.job_state / self.env.job_size # normalize
        normal_time = self.env.time / self.max_time

        assert 0 <= intensity <= 1, AssertionError("Intensities are not normalized")
        state = [normal_job_state, normal_time, intensity]
        assert len(state) == self.state_dim, AssertionError(f"State creation must be size {self.state_dim}. Currently is size {len(state)}.")

        return state
    
    def _save_in_memory(self, state, action, loss, next_state, done):
        self.memory.append((state, action, loss, next_state, done))
    
    def _replay(self, ep):
        if len(self.memory) < self.batch_size:
            return  # Not enough to sample

        batch = random.sample(self.memory, min(self.batch_size, ep))
        for state, action, loss, next_state, done in batch:
            features = self._get_features(state)

            # Compare it to some target
            if not done:
                next_features = self._get_features(next_state)   # Get features from next state
                next_q_vals = [approx.predict(next_features) for approx in self.approximators]
                target = loss + min(next_q_vals) # FIXME: This is giving [nan, nan]
            else:
                target = loss
            
            curr_q_val = self.approximators[action].predict(features) # FIXME: This is giving [nan, nan]
            td_error = target - curr_q_val

            self.approximators[action].update(features, td_error, self.lr, ep)

    
    def train_episode(self, normalize, episode):
        self.env.train()
        start_idx = self.env.get_random_index()
        self.env.reset(start_idx, normalize)
        done = False
        total_loss = 0
        total_carbon = 0

        while not done:

            curr_intensity = self.env.get_carbon()
            state = self._create_state(curr_intensity)
            action = self._choose_action(state)

            # Store carbon if run
            if action == self.env.run:
                total_carbon += curr_intensity


            # Perform action
            next_job_state, loss, done = self.env.step(action, self.tradeoff)

            # Store experience and train approximator
            next_intensity = self.env.get_carbon()
            next_state = self._create_state(next_intensity)
            self._save_in_memory(state, action, loss, next_state, done)

            self._replay(episode)

            total_loss += loss 

            if self.env.time > self.max_time:
                done = True

        # Regret
        optimal_carbon = self.env.get_optimal_carbon(idx=start_idx, tradeoff=self.tradeoff)
        # total_regret = utils.calculate_regret(total_loss, optimal_carbon)

        # Epsilon Decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Loss: {total_loss:.2f}, Epsilon: {self.epsilon:.2f}")

        return total_loss, total_carbon, optimal_carbon, self.env.time
    
    def evaluate(self, normalize):
        """
        Given a trained policy, evaluates it. This uses the test set defined
        on environment creation.
        """
        print("Beginning Evaluation...")
        self.env.test()
        start_idx = self.env.get_random_index()
        self.env.reset(start_idx, normalize)
        done = False
        total_loss = 0
        intensity_history = []
        state_history = []
        action_history = []
        q_vals_history = []
        loss_history = []

        while not done:
            curr_intensity = self.env.get_carbon()
            intensity_history.append(curr_intensity)

            # Create state and get features
            state = self._create_state(curr_intensity)
            state_history.append(state)
            features = self._get_features(state)

            # Get Q-Values for all actions
            q_vals = [approx.predict(features) for approx in self.approximators]
            q_vals_history.append(q_vals)

            # Select action
            action = np.argmin(q_vals)
            action_history.append(action)

            # Perform action
            _, loss, done = self.env.step(action, self.tradeoff)
            loss_history.append(loss)
            total_loss += loss

        return total_loss, action_history, intensity_history, state_history, loss_history, q_vals_history

    
    def visualize_weight_importance(self):
        feature_names = [
            "Job State", "Time", "Carbon", "Job x Carbon", "exp(-Job State)"
        ]
        # feature_names = [
        #     "Bias", "Job State", "Time", "Carbon", 
        #     "Job×Carbon", "Time×Carbon", "Job×Time",
        #     "Job²", "Time²", "Carbon²", 
        #     "exp(-Job)", "log(Job+1)"
        # ]

        plt.figure(figsize=(12, 10))

        for i, action_name in enumerate(["Pause", "Run"]):
            plt.subplot(2, 1, i+1)
            weights = self.approximators[i].weights
            abs_weights = np.abs(weights)
            # Sort by absolute weight magnitude
            sorted_idx = np.argsort(abs_weights)[::-1]
            
            plt.bar(range(len(weights)), abs_weights[sorted_idx], color='skyblue')
            plt.xticks(range(len(weights)), [feature_names[j] for j in sorted_idx], rotation=45)
            plt.title(f"Feature Importance for {action_name} Action")
            plt.ylabel("Absolute Weight Value")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.title(f"Tradeoff = {self.tradeoff}")
        return plt.gcf()

def plot_training_carbons(carbons, optimal_carbons):
    """Plot the carbon intensities incurred during training (carbon per episode)"""
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
    """Plot the training progress (losses per episode)."""
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

def plot_evaluation_results(actions, intensities, losses, q_vals):
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    # Plot carbon intensity
    axes[0].plot(intensities, 'g-')
    axes[0].set_title('Carbon Intensity')
    axes[0].set_ylabel('Intensity')
    axes[0].grid(True)
    
    # Plot actions (Run/Pause)
    axes[1].plot(actions, 'bo-', drawstyle='steps-post')
    axes[1].set_title('Actions (0=Pause, 1=Run)')
    axes[1].set_ylabel('Action')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True)
    
    # Plot losses
    axes[2].plot(losses, 'r-')
    axes[2].set_title('Losses')
    axes[2].set_ylabel('Loss')
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
    return fig



def main():
    print("Running LFA_QL")
    # Hyper Parameters
    job_size = 10
    # seed = 100
    alpha = 1e-5
    tradeoff = 0.05 # This proudces interesting results w/ normalize=False
    tradeoff = 0.05
    episodes = 1250
    normalize = True

    # Environment Parameters
    train_size = 0.8
    energy_df = DataLoader().data
    # energy_df = DataLoader(seed=seed).data
    # np.random.seed(seed)
    env = JobEnv(job_size, energy_df, train_size)

    # Agent
    agent = LinearQLearning(env, lr=alpha, tradeoff=tradeoff)

    # Tracking
    losses = []
    carbons = []
    optimal_carbons = []

    ## Training
    for e in tqdm(range(episodes)):
        loss, carbon, opt_carbon, _ = agent.train_episode(normalize, e)
        losses.append(loss)
        carbons.append(carbon)
        optimal_carbons.append(opt_carbon)

    ## Evaluate
    total_loss, action_history, intensity_history, state_history, loss_history, q_vals_history = agent.evaluate(normalize)

    plot_training_progress(losses)
    plt.show()

    plot_training_carbons(carbons, optimal_carbons)
    plt.show()

    plot_evaluation_results(action_history, intensity_history, loss_history, q_vals_history)
    print(f"Total loss: {total_loss:.2f}")
    print(f"Job completed in {len(action_history)} hours")
    plt.show()

    agent.visualize_weight_importance()
    plt.show()

if __name__ == '__main__':
    main()


