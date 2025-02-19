import numpy as np

class QLearn():

    def __init__(self, env, epsilon = 0.1, alpha = 1e-6, beta = 0.01):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.q = np.random.rand(self.env.nS, self.env.nA) 

    def choose_action(self, q, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.nA)
        else:
            return np.argmin(q[state]) # Choose smallest Q-val
        
    def update_q_value(self, s, a, loss, s_prime):
        # Q(s, a) = Q(s, a) + a(loss + min_a' Q(s', a') - Q(s, a))
        delta = np.min(self.q[s_prime]) - self.q[s, a]
        # print(f"Loss = {loss}")
        # print(f"Delta = {delta}")
        self.q[s, a] = self.q[s, a] + self.alpha * (loss + delta)
        return self.q[s, a]
    

    def train_episode(self, data):
        state = self.env.reset(energy_df=data)
        is_done = False
        episode_loss = 0
        n_steps = 0

        while not is_done:
            action = self.choose_action(self.q, state)
            next_state, loss, is_done = self.env.step(action)

            self.update_q_value(state, action, loss, next_state)

            episode_loss += loss
            n_steps += 1
            state = next_state

        return episode_loss, n_steps
