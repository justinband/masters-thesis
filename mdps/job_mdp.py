import numpy as np

class JobMDP():
    def __init__(self, job_size, energy_df=None):
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

    def get_latency(self):
        latency = (self.s + 1) / (self.time + 1)
        # print(f"Latency = {1 - latency:.3f} : {(self.s+1)}/{self.time+1}")
        return 1 - latency
    
    def get_loss(self, a=None): 
        if (a is not None) and (a == self.pause):
            return self.get_latency()
        else:
            return self.energy['normalized'].iloc[self.time]


    def step(self, a):
        """
        Execute run/pause action in the MDP.

        Running in the last state advances the MDP to a complete/terminal state.
        """
        self.curr_loss = self.get_loss(a)

        # Terminating Action - running in last state reaches terminal state
        if ((a == self.run) and (self.s == self.nS - 1)) or self.complete: # Terminal state
            self.complete = True
            return self.s, self.curr_loss, self.complete
        
        # Non-terminating action
        self.s = np.argmax(self.p[self.s, a]) # deterministic selection
        self.curr_loss = self.get_loss(a)
        self.time += 1

        # Deadline reached - last possible action
        if self.time == len(self.energy): 
            self.complete = True

        return self.s, self.curr_loss, self.complete
        
    def reset(self, energy_df=None):
        self.s = 0
        self.time = 0
        self.complete = False
        if energy_df is not None:
            self.energy = energy_df
            self.curr_loss = self.get_loss()

        return self.s