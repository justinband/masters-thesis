import numpy as np

class JobMDP():
    def __init__(self, job_size, energy_df=None):
        # MDP Parameters
        self.nS = job_size  # num states
        self.nA = 2         # num actions
        self.s = 0          # initial state
        self.time = 0       # current run time
        self.idx = 0        # current index in energy data

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

    def get_next_state(self):
        return np.argmax(self.p[self.s, self.run]) # Assumes that states are known

    def get_latency(self):
        if self.s == 0: # No runs performed
            latency = self.time
        else: # Some runs performed, calc latency
            latency = (self.time - self.s) / self.s
            
        return latency
    
    def get_intensity(self, normalized=False):
        if normalized:
            return self.energy['normalized'].iloc[self.idx]
        else:
            return self.energy['carbon_intensity'].iloc[self.idx]
    
    def get_loss(self, a=None):
        if (a is not None) and (a == self.pause): # Get latency on pause action
            return self.get_latency()
        else:
            return self.energy['normalized'].iloc[self.idx]

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
        
    def reset(self, start_idx):
        self.s = 0
        self.time = 0
        self.complete = False
        self.idx = start_idx
        self.curr_loss = self.get_loss()
        return self.s