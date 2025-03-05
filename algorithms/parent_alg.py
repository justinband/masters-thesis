from abc import ABC, abstractmethod

class LearningAlg():
    def __init__(self, env, epsilon, alpha, latency=0):
        self.env = env
        self.epsilon = epsilon  # Learning rate
        self.alpha = alpha      # Exploration-rate
        self.max_latency = latency

    @abstractmethod
    def train_episode(self, data):
        """
        Must be implemented to train algorithms on the environment.
        """
        pass


    