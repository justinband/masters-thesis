import numpy as np
from algorithms import LearningAlg
import matplotlib.pyplot as plt

class QLearn(LearningAlg):

    def __init__(self, env, epsilon = 0.1, alpha = 1e-6, latency=0):
        super().__init__(env, epsilon, alpha, latency)
        self.q = np.random.rand(self.env.nS, self.env.nA) 

    def choose_action(self, q, state, time, curr_latency):
        if (self.max_latency == 0) or (curr_latency >= time): # Mandatory run
            return self.env.run
        elif np.random.rand() < self.epsilon: # Explore
            return np.random.choice(self.env.nA)
        else: # Exploit
            return np.argmin(q[state]) # Choose smallest Q-val
        
    def update_q_value(self, s, a, loss, s_prime):
        # Q(s, a) = Q(s, a) + a(loss + min_a' Q(s', a') - Q(s, a))
        delta = np.min(self.q[s_prime]) - self.q[s, a]
        return self.q[s, a] + self.alpha * (loss + delta)

    def train_episode(self, start_idx):
        state = self.env.reset(start_idx)
        is_done = False
        episode_losses = []

        while not is_done:
            curr_latency = self.env.get_latency()
            action = self.choose_action(self.q, state, self.env.time, curr_latency)
            next_state, loss, is_done = self.env.step(action)

            self.q[state, action] = self.update_q_value(state, action, loss, next_state)

            episode_losses.append(loss)
            state = next_state

        return episode_losses, self.env.time
