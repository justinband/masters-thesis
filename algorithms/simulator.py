from datasets import DataLoader
from mdps import JobMDP
from algorithms import QLearn
import matplotlib.pyplot as plt

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
        for i in range(self.episodes):
            data = self.samples[0]
            episode_loss, n_steps = self.alg.train_episode(data)

            self.losses.append(episode_loss)
            
            print(f"Iteration: {i}: Loss = {(episode_loss/n_steps):.3f}, Steps = {n_steps}")

    def plot_training(self):
        plt.plot(self.losses)
        plt.show()
            
job_size = 10
deadline = 48
episodes = 1000

mdp = JobMDP(job_size)
q_learn = QLearn(mdp) #s Should update the hyper-params
sim = Simulator(q_learn, job_size, deadline, episodes)

sim.train()
sim.plot_training()