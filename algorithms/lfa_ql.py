import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from collections import deque
from tqdm import tqdm
from utils import generic, plotting
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
        self.feature_dim = 4
        self.feature_names = [
            "Job State", "Time", "Carbon", "exp(-Job State)"
        ]
        # self.feature_names = [
        #     "Bias", "Job State", "Time", "Carbon", 
        #     "Job×Carbon", "Time×Carbon", "Job×Time",
        #     "Job²", "Time²", "Carbon²", 
        #     "exp(-Job)", "log(Job+1)"
        # ]
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
            # job_state * carbon, # Interaction between job state and carbon. How far job is and how 'bad' it is to run now
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
        optimal_loss, optimal_carbon, optimal_time = self.env.calc_opt_carbon(start_idx=start_idx)

        regret = generic.calculate_regret(total_loss, optimal_loss)

        # Epsilon Decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Loss: {total_loss:.2f}, Optimal Loss: {optimal_loss:.2f}, TimeDiff: {optimal_time - self.env.time}, Epsilon: {self.epsilon:.2f}")

        return total_loss, total_carbon, optimal_carbon, regret
    
    def evaluate(self, normalize, start_idx=None):
        """
        Given a trained policy, evaluates it. This uses the test set defined
        on environment creation.
        """
        print("Beginning Evaluation...")
        self.env.test()
        if start_idx is None:
            start_idx = self.env.get_random_index()
        start_idx = 6500
        self.env.reset(start_idx, normalize)
        done = False
        total_loss = 0
        total_carbon = 0
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
            if action == self.env.run:
                total_carbon += curr_intensity

        return total_loss, action_history, intensity_history, state_history, loss_history, q_vals_history, total_carbon

def main():
    print("Running LFA_QL")
    # Hyper Parameters
    job_size = 30
    lr = 1e-5
    tradeoff = 0.05
    episodes = 4000
    normalize = True
    alpha = 4

    # Environment Parameters
    train_size = 0.8
    dataloader = DataLoader()
    env = JobEnv(job_size, alpha, dataloader, train_size)

    # Agent
    agent = LinearQLearning(env, lr=lr, tradeoff=tradeoff)

    # Tracking
    losses = []
    carbons = []
    optimal_carbons = []
    regrets = []
    cum_regret = []

    ## Training
    for e in tqdm(range(episodes)):
        loss, carbon, opt_carbon, regret = agent.train_episode(normalize, e)
        losses.append(loss)
        carbons.append(carbon)
        optimal_carbons.append(opt_carbon)
        regrets.append(regret)

        if e == 0:
            cum_regret.append(regret)
        else:
            cum_regret.append(cum_regret[-1] + regret)

    ## Evaluate
    total_loss, action_history, intensity_history, state_history, loss_history, q_vals_history, total_carbon = agent.evaluate(normalize)

    plotting.plot_training_progress(losses)


    plotting.plot_training_carbons(carbons, optimal_carbons)
    plt.show()

    plotting.plot_regret(regrets, cum_regret)
    plt.show()

    print(f"Total loss: {total_loss:.2f}")
    print(f"Total Carbon: {total_carbon:.5f}")
    print(f"Job completed in {len(action_history)} hours")
    plotting.plot_evaluation_results(action_history, intensity_history, loss_history, q_vals_history)
    plt.show()

    plotting.visualize_weights(agent.feature_names, agent.approximators)
    plt.show()

if __name__ == '__main__':
    main()


