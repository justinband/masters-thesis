from datasets import DataLoader
from mdps import JobMDP
from algorithms import QLearn, InformedQL, LinearQ
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import argparse


class Simulator():
    def __init__(self, alg, job_size, deadline, episodes, latency, seed=None):
        self.energy = DataLoader(seed=seed)

        mdp = JobMDP(job_size)
        if alg == "ql":
            self.alg = QLearn(mdp)
        elif alg == "informed-ql":
            self.alg = InformedQL(mdp)
        elif alg == "linq":
            state_dim = 2
            action_dim = 2
            self.alg = LinearQ(mdp, state_dim=state_dim, action_dim=action_dim)

        self.job_size = job_size
        self.deadline = deadline
        self.episodes = episodes
        self.latency = latency
        self.samples = self.energy.get_n_samples(deadline, episodes)

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
    latency = 1
    episodes = 100000

    parser = argparse.ArgumentParser(description="Run simulator with a specified algorithm.")
    parser.add_argument("algorithm", type=str, help="Algorithm to run. Options: ql, informed-ql, linq")
    parser.add_argument("-e", "--episodes", type=int, default=episodes, help="Number of episodes to train on")
    parser.add_argument("-j", "--job-size", type=int, default=job_size, help="Size of a job")
    parser.add_argument("-d", "--deadline", type=int, default=deadline, help="Deadline jobs must complete by")
    parser.add_argument("-l", "--latency", type=int, default=latency, help="Amount of latency we're willing to incur. Latency=0 means no latency. Latency=1 effectively doubles runtime")
    parser.add_argument("-s", "--seed", type=int, help="Defines a seed. Useful for reproduction.")
    args = parser.parse_args()

    sim = Simulator(args.algorithm, args.job_size, args.deadline, args.episodes, args.latency, args.seed)
    sim.train()
    sim.plot_losses(title=args.algorithm)
