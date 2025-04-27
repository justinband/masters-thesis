import numpy as np
from mdps import JobEnv

class RunAgent():
    def __init__(self, env: JobEnv):
        self.env = env
    
    def evaluate(self, start_idx=None):
        self.env.test()
        if start_idx is None:
            start_idx = self.env.get_random_index()
        self.env.reset(start_idx)

        done = False
        total_loss = 0
        total_carbon = 0

        intensity_history = []
        state_history = np.arange(0, self.env.job_size) # Known since we always run
        action_history = np.ones(self.env.job_size) # Known since we always run
        loss_history = []
    
        while not done:
            curr_intenisty = self.env.get_carbon()
            intensity_history.append(curr_intenisty)

            action = self.env.run

            _, loss, done = self.env.step(action)
            loss_history.append(loss)

            total_loss += loss
            total_carbon += curr_intenisty
            
        return total_loss, action_history, intensity_history, state_history, loss_history, [], total_carbon
