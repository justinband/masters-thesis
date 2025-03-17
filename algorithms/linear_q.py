import numpy as np
import os
import pickle
from pathlib import Path
from algorithms import LearningAlg
from scipy.stats import linregress
from collections import deque

class LinearQ(LearningAlg):

    def __init__(self, env, alpha = 1e-6, epsilon = 0.1, latency=0):
        super().__init__(env, epsilon, alpha, latency)

        self.state_dim = 4
        self.action_dim = 2

        self.w = np.zeros((self.action_dim, self.state_dim))  # Linear weights
        # self.w = np.zeros((self.action_dim, self.state_dim))  # Linear weights

    def create_state(self, progress, intensity_q, latency):
        '''
        This function incorporates a variety of things into the state.

        For example, job progress and the current trend may be included.
        '''
        slope = np.nan_to_num(linregress(x=np.arange(len(intensity_q)), y=intensity_q).slope)
        curr_intensity = intensity_q[-1]

        out_state = np.array([progress, curr_intensity, slope, latency])
        assert len(out_state) == self.state_dim, AssertionError(f"State creation must be size {self.state_dim}. Currently is size {len(out_state)}.")

        return out_state
    
    def choose_action(self, state, curr_latency):
        if (self.max_latency == 0) or (curr_latency >= self.max_latency):
            return self.env.run
        elif np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_vals = np.dot(self.w, state)
            return np.argmin(q_vals)
        
    def update_weights(self, state, action, loss, s_prime):
        curr_q = np.dot(self.w[action], state)    
        next_q = np.min(np.dot(self.w, s_prime))
        error = (loss + next_q) - curr_q
        # error = loss - curr_q
        self.w[action] += self.alpha * error * state

    def run_episode(self, start_idx, train: bool):
        progress = self.env.reset(start_idx)
        is_done = False
        episode_losses = []
        episode_latencies = []
        episode_intensities = []
        
        intensity_q = deque(maxlen=2)

        while not is_done:
            # Get State
            curr_latency = self.env.get_latency()
            episode_latencies.append(curr_latency)

            curr_intensity = self.env.get_loss()
            intensity_q.append(curr_intensity) # Add curr intensity to Queue
            episode_intensities.append(self.env.get_intensity())

            state = self.create_state(progress, intensity_q, curr_latency)
            # state = self.create_state(progress, curr_intensity, prev_intensity)

            # Select Action
            action = self.choose_action(state, curr_latency)

            # Execute Action
            s_prime, loss, is_done = self.env.step(action)

            # Update the weights
            if train:
                self.update_weights(state, action, loss, s_prime)

            # Update tracking
            episode_losses.append(loss)
            progress = s_prime

        return episode_losses, episode_latencies, episode_intensities, self.env.time

    def save_weights(self, path=None):
        DATA_DIR = Path(__file__).parent
        if path:
            DATA_DIR = path
        
        filename = os.path.join(DATA_DIR, "linq_weights.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self.w, f)
    
    def reset(self):
        self.w = np.zeros((self.action_dim, self.state_dim))