import numpy as np
import math

class JobMDP():
    def __init__(self, job_size, normalized, energy_df=None):
        # MDP Parameters
        self.nS = job_size  # num states
        self.nA = 2         # num actions
        self.s = 0          # initial state
        self.time = 0       # current run time
        self.idx = 0        # current index in energy data
        self.normalized = normalized

        # Actions
        self.pause = 0
        self.run = 1

        # Flags
        self.complete = False

        # Energy Data
        self.energy = energy_df
        self.curr_loss = None

        # Transiton Matrix
        self.p = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            # Pause
            self.p[s, self.pause, s] = 1

            # Run
            if s < self.nS - 1:
                self.p[s, self.run, s+1] = 1
            else:
                self.p[s, self.run, s] = 1 # We go to 'terminal state'

    def _get_energy_type(self):
        return 'normalized' if self.normalized else 'carbon_intensity'

    def get_next_state(self):
        return np.argmax(self.p[self.s, self.run]) # Assumes that states are known

    def get_latency(self):
        if self.s == 0: # No runs performed
            latency = self.time
        else: # Some runs performed, calc latency
            latency = (self.time - self.s) / self.s
            
        return latency
    
    def get_intensity(self, idx):
        energy_type = self._get_energy_type()
        return self.energy[energy_type].iloc[idx]
    
    def get_intensity(self):
        energy_type = self._get_energy_type()
        return self.energy[energy_type].iloc[self.idx]
    
    def get_loss(self, a=None):
        energy_type = self._get_energy_type()
        return self.energy[energy_type].iloc[self.idx]
        # if (a is not None) and (a == self.pause): # Get latency on pause action
        #     return self.get_latency()
        # else:
        #     return self.energy[energy_type].iloc[self.idx]

    def step(self, a):
        """
        Execute run/pause action in the MDP.

        Running in the last state advances the MDP to a complete/terminal state.
        """
        if self.idx == len(self.energy) - 1: # Data wrap around
            self.idx = 0

        self.curr_loss = self.get_loss(a)

        # Terminating Action - running in last state reaches terminal state
        if ((a == self.run) and (self.s == self.nS - 1)) or self.complete: # Terminal state
            self.complete = True
            return self.s, self.curr_loss, self.complete
        
        # Non-terminating action
        self.s = np.argmax(self.p[self.s, a]) # deterministic selection
        self.time += 1
        self.idx += 1

        return self.s, self.curr_loss, self.complete
    
    def get_optimal(self, start_idx, tradeoff):
        job_size = self.nS
        energy_type = self._get_energy_type()
        df = self.energy[energy_type]

        # T = N; therefore idx + N
        max_indices = np.arange(start_idx, start_idx + job_size) % len(self.energy)
        max_carbon = df.iloc[max_indices].sum()

        # Calculate T upper bound
        T = math.ceil((job_size/tradeoff) * max_carbon + job_size)
        # T is >= N; therefore idx + T
        T_indices = np.arange(start_idx, start_idx + T) % len(self.energy)

        opt_values = df.iloc[T_indices].nsmallest(job_size)
        opt_indices = opt_values.index.to_list()
        opt_carbon = opt_values.sum()

        return opt_carbon, opt_values, opt_indices
        
    def reset(self, start_idx):
        self.s = 0
        self.time = 0
        self.complete = False
        self.idx = start_idx
        self.curr_loss = self.get_loss()
        return self.s