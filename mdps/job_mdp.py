from pandas import DataFrame
from datasets import DataLoader
import numpy as np

class JobMDP():
    def __init__(self, job_size, energy_df):
        # MDP Parameters
        self.nS = job_size  # num states
        self.nA = 2         # num actions
        self.s = 0          # initial state
        self.time = 0       # current run time

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

    def step(self, a):
        """
        Execute run/pause action in the MDP.

        Running in the last state advances the MDP to a complete/terminal state.
        """
        ### If deadline reached
        if self.time >= len(self.energy):
            print("Deadline has been reached.")
            self.complete = True
            return -1, -1, self.complete 
        
        ### Otherwise, execute action
        self.curr_loss = self.energy.iloc[self.time]

        if ((a == self.run) and (self.s == self.nS - 1)) or self.complete: # Goes to terminal state or completed
            self.complete = True
            print("Job is complete:")
        else: # Perform action
            self.s = np.argmax(self.p[self.s, a]) # deterministic selection

        self.time += 1

        return self.s, self.curr_loss, self.complete
        
    def reset(self, energy_df=None):
        self.s = 0
        self.time = 0
        self.curr_loss = self.energy.iloc[self.time]
        self.complete = False
        if energy_df:
            self.energy = energy_df

        return self.s