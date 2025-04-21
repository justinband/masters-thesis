from abc import ABC, abstractmethod

class LearningAlg():
    def __init__(self, env, epsilon, alpha, tradeoff):
        self.env = env
        self.epsilon = epsilon  # Learning rate
        self.alpha = alpha      # Exploration-rate
        self.tradeoff = tradeoff

    @abstractmethod
    def train_episode(self, data):
        """
        Must be implemented to train algorithms on the environment.
        """
        pass


    