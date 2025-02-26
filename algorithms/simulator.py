from datasets import DataLoader
from mdps import JobMDP
from algorithms import QLearn, LinearQ
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import argparse


class Simulator():
    def __init__(self, alg, job_size, deadline, episodes, seeds=None):
        self.energy = DataLoader()

        if alg == "ql":
            self.alg = QLearn(JobMDP(job_size))
        elif alg == "linq":
            state_dim = 2
            action_dim = 2
            self.alg = LinearQ(JobMDP(job_size), state_dim=state_dim, action_dim=action_dim)

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


if __name__ == "__main__":
    job_size = 10
    deadline = 48 # hours
    episodes = 100000

    parser = argparse.ArgumentParser(description="Run simulator with a specified algorithm.")
    parser.add_argument("algorithm", type=str, help="Algorithm to run. Options: ql, linq")
    parser.add_argument("-e", "--episodes", type=int, default=episodes, help="Number of episodes to train on")
    parser.add_argument("-j", "--job-size", type=int, default=job_size, help="Size of a job")
    parser.add_argument("-d", "--deadline", type=int, default=deadline, help="Deadline jobs must complete by")
    args = parser.parse_args()

    sim = Simulator(args.algorithm, args.job_size, args.deadline, args.episodes)
    sim.train()
    sim.plot_losses(title=args.algorithm)
