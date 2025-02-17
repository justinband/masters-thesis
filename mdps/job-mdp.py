import numpy as np

class JobMDP():
    def __init__(self, job_size):
        # MDP Parameters
        self.nS = job_size  # num states
        self.nA = 2         # num actions
        self.s = 0          # initial state

        # Actions
        self.pause = 0
        self.run = 1

        # Flags
        self.complete = False

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
        if ((a == self.run) and (self.s == self.nS - 1)) or self.complete:
            self.complete = True
            print(f"Complete - s={self.s}")
            return self.s, self.complete
        else:
            self.s = np.argmax(self.p[self.s, a]) # Deterministic selection.
            return self.s, self.complete
        
    def reset(self):
        self.s = 0
        self.complete = False
        return self.s