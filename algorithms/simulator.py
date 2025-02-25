from datasets import DataLoader
from mdps import JobMDP
from algorithms import QLearn, LinearQ
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np


class Simulator():
    def __init__(self, alg, job_size, deadline, episodes, seeds=None):
        self.energy = DataLoader()
        self.alg = alg # Alg should be initialized with MDP
        self.job_size = job_size
        self.deadline = deadline
        self.episodes = episodes
        self.samples, self.seeds = self.energy.get_n_samples(deadline, episodes, seeds)

        self.losses = []

    def train(self):
        for _ in tqdm(range(self.episodes)):
            data = self.samples[0]
            episode_losses, _ = self.alg.train_episode(data)
            self.losses.append(np.sum(episode_losses))

    def plot_losses(self, title=""):
        plt.plot(self.losses)
        plt.title(title)
        plt.show()


job_size = 10
deadline = 48 # hours
episodes = 100000

mdp = JobMDP(job_size)

# q_learn = QLearn(mdp) #s Should update the hyper-params
# sim = Simulator(q_learn, job_size, deadline, episodes)
# sim.train()
# sim.plot_training("Q-Learn")
# sim.plot_losses("Q-Learn Incurred")

mdp.reset()
linear_q = LinearQ(mdp, state_dim=2, action_dim=2)
sim = Simulator(linear_q, job_size, deadline, episodes)
sim.train()
sim.plot_losses("Linear Q")