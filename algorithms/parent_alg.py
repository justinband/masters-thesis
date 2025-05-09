import numpy as np
from abc import ABC, abstractmethod
from mdps import JobEnv

class LearningAlg():
    def __init__(self, env: JobEnv, lr, epsilon, epsilon_min, epsilon_decay):
        self.env = env
        self.lr = lr      # Exploration-rate
        self.epsilon = epsilon  # Learning rate
        self.epsilon_max = epsilon
        self.epsilon_min = epsilon_min
        self.decay_factor = epsilon_decay # 0.001
    @abstractmethod
    def train_episode(self, data):
        """
        Must be implemented to train algorithms on the environment.
        """
        pass

    @abstractmethod
    def evaluate(self, start_idx=None):
        """
        Must be implemented for algorithmic evaluation.
        """
        pass

    def decay_epsilon(self, episode):
        """
        Decays epsilon according to a standardized approach
        """
        if self.epsilon > self.epsilon_min:
            # Inverse Time Decay
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) / (1 + self.decay_factor * episode)

    