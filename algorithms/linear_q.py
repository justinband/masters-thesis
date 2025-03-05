import numpy as np
from algorithms import LearningAlg

class LinearQ(LearningAlg):

    def __init__(self, env, state_dim, action_dim = 2, alpha = 1e-6, epsilon = 0.1, latency=0):
        super().__init__(env, epsilon, alpha, latency)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.w = np.zeros((self.action_dim, self.state_dim))  # Linear weights
        # self.w = np.zeros((self.action_dim, self.state_dim))  # Linear weights

    def create_state(self, progress, curr_intensity):
        '''
        This function incorporates a variety of things into the state.

        For example, job progress and the current trend may be included.
        '''
        return np.array([progress, curr_intensity])
    
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

    def train_episode(self, start_idx):
        progress = self.env.reset(start_idx)
        is_done = False
        episode_losses = []

        while not is_done:
            curr_intensity = self.env.get_loss()
            curr_latency = self.env.get_latency()

            state = self.create_state(progress, curr_intensity)

            action = self.choose_action(state, curr_latency)

            next_state, loss, is_done = self.env.step(action)

            self.update_weights(state, action, loss, next_state)

            episode_losses.append(loss)
            state = next_state

        return episode_losses, self.env.time
    
    def reset(self):
        self.w = np.zeros((self.action_dim, self.state_dim))