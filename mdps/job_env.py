import numpy as np
import math

class JobEnv():
    def __init__(self, job_size, df, start_idx = 0):
        self.job_size = job_size  # num states
        self.nA = 2     # num actions
        self.job_state = 0      # intitial state
        self.job_state_tracking = []
        self.time = 0   # time tracker
        self.curr_idx = start_idx    # current energy index

        self.pause = 0
        self.run = 1

        self.complete = False
        self.is_normal = None

        # Energy data
        self.data = df

    def _get_energy_type(self, is_normal):
        assert is_normal is not None, AssertionError("JobEnv not correctly initialized")
        return 'normalized' if is_normal else 'carbon_intensity'

    def _calculate_loss(self, action, tradeoff, idx):
        intensity = self.data[self._get_energy_type(self.is_normal)].iloc[idx]

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
    
    def get_optimal_carbon(self, normalize, idx, tradeoff):
        data = self.data[self._get_energy_type(normalize)]

        max_indices = np.arange(idx, idx + self.job_size) % len(self.data)
        max_carbon = data.iloc[max_indices].sum()

        if tradeoff > 0:
            bound = (self.job_size / tradeoff) * max_carbon + self.job_size
        else:
            bound = self.job_size
        T = math.ceil(bound)
        T_optimal_indices = np.arange(idx, idx + T) % len(self.data)
        T_optimal_carbon = data.iloc[T_optimal_indices].nsmallest(self.job_size).sum()

        return T_optimal_carbon

    
    def get_carbon(self, normalize):
        idx = self.curr_idx
        return self.data[self._get_energy_type(normalize)].iloc[idx]
    
    def step(self, action, tradeoff):
    
        loss = self._calculate_loss(action, tradeoff, self.curr_idx)

        if (action == self.run and self.job_state == self.job_size - 1):
            self.complete = True
            return self.job_state, loss, self.complete
        
        self.job_state_tracking.append(self.job_state) # Adds

        if action == self.run:
            self.job_state += 1

        self.time += 1

        self.curr_idx = (self.curr_idx + 1) % len(self.data)

        return self.job_state, loss, self.complete
  
    def reset(self, start_idx, is_normal):
        self.job_state = 0
        self.job_state_tracking = []
        self.time = 0
        self.curr_idx = start_idx
        self.complete = False
        self.is_normal = is_normal
        return self.job_state
