import numpy as np


class LinearQ():

    def __init__(self, env, state_dim, action_dim = 2, alpha = 1e-6, epsilon = 0.1):
        self.env = env
        self.alpha = alpha      # Learning rate
        self.epsilon = epsilon  # Exploration-rate

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.w = np.zeros((action_dim, state_dim))  # Linear weights
        # self.w = np.zeros((action_dim, state_dim))  # Linear weights

    def create_state(self, progress, curr_intensity):
        '''
        This function incorporates a variety of things into the state.

        For example, job progress and the current trend may be included.
        '''
        return np.array([progress, curr_intensity])
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_vals = np.dot(self.w, state)
            return np.argmin(q_vals)
        

    def update_weights(self, state, action, loss, s_prime):
        curr_q = np.dot(self.w[action], state)    
        next_q = np.min(np.dot(self.w, s_prime))
        error = (loss + 0.99 * next_q) - curr_q
        error = loss - curr_q
        self.w[action] += self.alpha * error * state

    def train_episode(self, data):
        progress = self.env.reset(energy_df=data)
        is_done = False
        episode_loss = 0
        n_steps = 0

        while not is_done:
            curr_intensity = self.env.get_loss()
            state = self.create_state(progress, curr_intensity)

            action = self.choose_action(state)

            next_state, loss, is_done = self.env.step(action)

            self.update_weights(state, action, loss, next_state)

            episode_loss += loss
            n_steps += 1
            state = next_state

        return episode_loss, n_steps