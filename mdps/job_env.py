import numpy as np
import math
import matplotlib.pyplot as plt
from datasets import DataLoader

class JobEnv():
    def __init__(self, job_size, alpha, dataloader: DataLoader, normalize, train_size=0.75, start_idx=0):
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
        self.carbon_alpha = self.dataloader.get_quantile_from_data(alpha)
        self.lambdas = []

        self.complete = False
        self.is_normal = normalize
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
    
    def _get_lambda(self, state):
        return 1/self.alpha * self.carbon_alpha * (state + 1)
        # return 1/self.alpha * self.carbon_alpha * self.job_size
    
    def _get_T_alpha(self, sum_data):
        # T <= [alpha * (1/carbon_alpha) * sum^N_{i=1} c_i] + N
        N = self.job_size
        T = (self.alpha * (1/self.carbon_alpha) * sum_data) + N
        return T

    def calc_opt_carbon(self, start_idx):
        data = self._get_dataset()
        job_size = self.job_size

        if self.alpha == 0: # Running job in same time
            T = job_size
        else:
            first_n_indices = np.arange(start_idx, start_idx + job_size) % len(data)
            first_n_carbon = data.iloc[first_n_indices].sum()
            T = self._get_T_alpha(first_n_carbon)
            T = math.ceil(T)
        
        # Calculate T upper bound
        T_indices = np.arange(start_idx, start_idx + T) % len(data)
        T_carbon = data.iloc[T_indices].reset_index(drop=True)
        run_indices = T_carbon.nsmallest(job_size).index

        ## Calculate losses for the optimal actions
        state_tracking = []
        state = 0
        optimal_carbon = 0
        optimal_loss = 0

        for t in range(T):
            intensity = T_carbon[t]
            T_prev = t - 1
            N_prev = state_tracking[-1] if state_tracking else 0
            lamb = self._get_lambda(state=N_prev)

            if t in run_indices:
                loss = self._calc_run_loss(lamb, T_prev, N_prev, intensity)
                optimal_carbon += intensity

                state_tracking.append(state)
                state += 1
            else:
                loss = self._calc_pause_loss(lamb, N_prev)
                state_tracking.append(state)

            optimal_loss += loss

            if state >= job_size:
                break

        return optimal_loss, optimal_carbon, T

    def _calculate_loss(self, action, idx, track=False):
        data = self._get_dataset()
        intensity_t = data.iloc[idx]
        T_prev = self.time - 1
        N_prev = self.job_state_tracking[-1] if self.job_state_tracking else 0
        lamb = self._get_lambda(state=N_prev)

        if track:
            self.lambdas.append(lamb)

        if action == self.run:
            return self._calc_run_loss(lamb, T_prev, N_prev, intensity_t)
        elif action == self.pause:
            return self._calc_pause_loss(lamb, N_prev)
        else:
            raise Exception("Loss Calculation: Neither run or pause action were performed")
    
    def _calc_pause_loss(self, lamb, state):
        if state == 0:
            return 0
        else:
            return lamb/state

    def _calc_run_loss(self, lamb, time, state, intensity):
        if state == 0:
            return intensity
        
        numer = time - state
        denom = (state + 1) * state
        penalty = -lamb * (numer/denom)
        return intensity + penalty
    
    def get_random_index(self):
        data = self.train_data if self.is_train else self.test_data
        return np.random.randint(len(data)) 
    
    def get_carbon(self):
        data = self._get_dataset()
        return data.iloc[self.curr_idx]
    
    def step(self, action, track=False):
        loss = self._calculate_loss(action, self.curr_idx, track)

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
  
    def reset(self, start_idx):
        self.job_state = 0
        self.job_state_tracking = []
        self.time = 0
        self.curr_idx = start_idx
        self.complete = False
        return self.job_state
