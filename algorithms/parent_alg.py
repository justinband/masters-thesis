from abc import ABC, abstractmethod

class LearningAlg():
    def __init__(self, env, lr, epsilon, epsilon_min, epsilon_decay):
        self.env = env
        self.lr = lr      # Exploration-rate
        self.epsilon = epsilon  # Learning rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    @abstractmethod
    def train_episode(self, data):
        """
        Must be implemented to train algorithms on the environment.
        """
        pass

    def decay_epsilon(self):
        """
        Decays epsilon according to a standardized approach
        """
        # FIXME: Play around with the decaying
        if self.epsilon > self.epsilon_min:
            # self.epsilon = max(self.epsilon_min, self.epsilon - ((1-self.epsilon_min)/4000)) # Constant rate
            # self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-0.0001 * episode) # "Inverse step decay", faster to start and slows down.
            self.epsilon *= self.epsilon_decay


    