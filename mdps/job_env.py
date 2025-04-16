import numpy as np
import math
from datasets import DataLoader

class JobEnv():
    def __init__(self, job_size, alpha, dataloader: DataLoader, train_size=0.75, start_idx=0):
        self.job_size = job_size  # num states
        self.nA = 2     # num actions
        self.job_state = 0      # intitial state
        self.job_state_tracking = []
        self.time = 0   # time tracker
        self.curr_idx = start_idx    # current energy index
        self.dataloader = dataloader
        df = self.dataloader.data

        # Actions
        self.pause = 0
        self.run = 1

        # Calculate Lambda
        self.alpha = alpha
        self.lamb = self._get_lambda()

        self.complete = False
        self.is_normal = None
        self.is_train = True

        # Energy data
        self.train_data, self.test_data = self._train_test_split(df, train_size)

    def _train_test_split(self, data, train_size):
        split_idx = int(len(data) * train_size)
        train_df = data.iloc[:split_idx].reset_index(drop=True)
        test_df = data.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df
    
    def _get_dataset(self):
        energy_type = self._get_energy_type()
        return self.train_data[energy_type] if self.is_train else self.test_data[energy_type]
            
    def _get_energy_type(self):
        assert self.is_normal is not None, "JobEnv not correctly initialized"
        return 'normalized' if self.is_normal else 'carbon_intensity'
    
    def _get_lambda(self):
        carbon_alpha = self.dataloader.get_quantile_from_data(self.alpha)
        return 1/self.alpha * carbon_alpha * self.job_size

    def _calculate_loss(self, action, tradeoff, idx):
        data = self._get_dataset()
        intensity = data.iloc[idx]

        # if action == self.run:
        #     if self.time == 0:
        #         loss = intensity
        #     else:
        #         # Numer is [prev time (time - 1)] - state in previous time
        #         #   This does not mean state - 1! It means what was the state at the last time step
        #         prev_time = self.time - 1
        #         prev_state = self.job_state_tracking[prev_time]

        #         if prev_state > 0:
        #             offset = 0
        #         else:
        #             offset = -tradeoff * (prev_time - prev_state) / (prev_state**2 + prev_state)

        #         loss = intensity + offset
        # elif action == self.pause:
        #     if self.time == 0
        if action == self.run:  # Run
            time_t = self.time - 1
            state_t = self.job_state - 1
            offset = -tradeoff * (time_t - state_t) / (state_t**2 + state_t) if state_t > 0 else 0
            loss = intensity + offset
        elif action == self.pause:
            state_t = self.job_state - 1
            # TODO: We do tradeoff in the else case. Good or bad?
            offset = tradeoff * (1/ state_t) if state_t > 0 else tradeoff
            loss = offset

        return loss
    
    def get_random_index(self):
        data = self.train_data if self.is_train else self.test_data
        return np.random.randint(len(data)) 
    
    def get_optimal_carbon(self, idx, tradeoff):
        data = self._get_dataset()

        max_indices = np.arange(idx, idx + self.job_size) % len(data)
        max_carbon = data.iloc[max_indices].sum()

        if tradeoff > 0:
            bound = (self.job_size / tradeoff) * max_carbon + self.job_size
        else:
            bound = self.job_size
        T = math.ceil(bound)
        T_optimal_indices = np.arange(idx, idx + T) % len(data)
        T_optimal_carbon = data.iloc[T_optimal_indices].nsmallest(self.job_size).sum()

        return T_optimal_carbon
    
    def get_carbon(self):
        data = self._get_dataset()
        return data.iloc[self.curr_idx]
    
    def step(self, action, tradeoff):
        loss = self._calculate_loss(action, tradeoff, self.curr_idx)

        if action == self.run and self.job_state == self.job_size - 1:
            self.complete = True
            return self.job_state, loss, self.complete
        
        self.job_state_tracking.append(self.job_state) # Adds

        if action == self.run:
            self.job_state += 1

        self.time += 1
        self.curr_idx = (self.curr_idx + 1) % len(self._get_dataset())

        return self.job_state, loss, self.complete
    
    def train(self):
        self.is_train = True

    def test(self):
        self.is_train = False
  
    def reset(self, start_idx, is_normal):
        self.job_state = 0
        self.job_state_tracking = []
        self.time = 0
        self.curr_idx = start_idx
        self.complete = False
        self.is_normal = is_normal
        return self.job_state
