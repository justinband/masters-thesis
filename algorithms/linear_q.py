import numpy as np
import os
import pickle
from pathlib import Path
from algorithms import LearningAlg
from scipy.stats import linregress
from collections import deque

class LinearQ(LearningAlg):

    def __init__(self, env, alpha = 1e-3, epsilon = 0.2, decay_rate = 1e-6, latency=0):
        super().__init__(env, epsilon, alpha, latency)

        self.state_dim = 4
        self.action_dim = 2
        self.min_epsilon = 0.1
        self.decay_rate = decay_rate
        self.curr_epsilon = self.epsilon
        self.error_found = False

        self.w = np.zeros((self.action_dim, self.state_dim))  # Linear weights
        # self.w = np.zeros((self.action_dim, self.state_dim))  # Linear weights

    def create_state(self, progress, last_intensities, latency):
        '''
        This function incorporates a variety of things into the state.

        For example, job progress and the current trend may be included.
        '''
        slope = np.nan_to_num(linregress(x=np.arange(len(last_intensities)), y=last_intensities).slope)
        curr_intensity = last_intensities[-1]

        out_state = np.array([progress, curr_intensity, slope, latency])
        assert len(out_state) == self.state_dim, AssertionError(f"State creation must be size {self.state_dim}. Currently is size {len(out_state)}.")

        return out_state
    
    def choose_action(self, state, curr_latency):
        if (self.max_latency == 0) or (curr_latency >= self.max_latency):
            return self.env.run
        elif np.random.rand() < self.curr_epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_vals = np.dot(self.w, state)
            return np.argmin(q_vals)
        
    def update_weights(self, state, action, loss, s_prime, episode):
        curr_q = np.dot(self.w[action], state)    
        next_q = np.min(np.dot(self.w, s_prime))

        error = (loss + next_q) - curr_q
        # error = np.clip(error, -10, 10)

        if (not self.error_found) and (np.isnan(state).any() or np.isnan(error)):
            self.error_found = True
        # error = loss - curr_q
        self.w[action] += self.alpha * error * state
        self.w = np.clip(self.w, -1e6, 1e6)

    def run_episode(self, start_idx, episode, train: bool):
        progress = self.env.reset(start_idx)
        is_done = False
        cum_loss = 0
        latencies = []
        intensities = []
        last_intensities = deque(maxlen=2)

        self.curr_epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate * episode)
        # if self.curr_epsilon == 0.1:
        #     print(f"Episode min epsilon = {episode}")

        while not is_done:
            curr_latency = self.env.get_latency()
            curr_intensity = self.env.get_intensity(normalized=True)   # observe current CI

            latencies.append(curr_latency)
            intensities.append(curr_intensity)
            last_intensities.append(curr_intensity) # add CI to queue

            state = self.create_state(progress, last_intensities, curr_latency)

            action = self.choose_action(state, curr_latency)

            s_prime, loss, is_done = self.env.step(action)

            # Update the weights
            if train:
                self.update_weights(state, action, loss, s_prime, episode)

            # Update tracking
            if action == self.env.run:  # losses are incurred when running
                cum_loss += loss
            progress = s_prime

    
        cum_loss = np.round(cum_loss, 2)
        regret, optimal_loss = self._calculate_regret(cum_loss, intensities, self.env.nS)

        return cum_loss, optimal_loss, regret, latencies, intensities, self.env.time
    
    def _calculate_regret(self, loss, intensities, job_size):
        optimal_loss = np.sum(np.sort(intensities)[:job_size])
        regret = loss - optimal_loss
        return regret, np.round(optimal_loss, 2)

    def save_weights(self, path=None):
        DATA_DIR = Path(__file__).parent
        if path:
            DATA_DIR = path
        
        filename = os.path.join(DATA_DIR, "linq_weights.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self.w, f)
    
    def reset(self):
        self.w = np.zeros((self.action_dim, self.state_dim))