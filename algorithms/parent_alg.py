from abc import ABC, abstractmethod

class LearningAlg():
    def __init__(self, env, epsilon, lr):
        self.env = env
        self.epsilon = epsilon  # Learning rate
        self.lr = lr      # Exploration-rate

    @abstractmethod
    def train_episode(self, data):
        """
        Must be implemented to train algorithms on the environment.
        """
        pass


    