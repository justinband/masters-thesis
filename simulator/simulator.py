from datasets import DataLoader
from mdps import JobMDP
from algorithms import QLearn, InformedQL, LinearQ
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
import pandas as pd

class Simulator():
    def __init__(self, algs, job_size, episodes, latency, iterations, verbose, seed=None):
        self.seed = np.random.randint(0, 2**36 - 1) if seed is None else seed
        
        self.energy = DataLoader(seed=seed)
        self.job_size = job_size
        self.episodes = episodes
        self.latency = latency
        self.iterations = iterations
        self.verbose = verbose

        self.train_data, self.val_data = self.energy.split_data(train_size=0.8)
        self.losses = []

        mdp = JobMDP(job_size, self.train_data)
        self.algs = {}
        for a in algs:
            self.algs[a] = self.create_alg(a, mdp)

        self.losses = np.empty((len(self.algs), self.iterations, self.episodes))

    def create_alg(self, alg_name, env):
        if alg_name == "ql":
            title = "Q-Learning"
            algorithm = QLearn(env, latency=self.latency)
        elif alg_name == "iql":
            title = "Informed Q-Learning"
            algorithm = InformedQL(env, latency=self.latency)
        elif alg_name == "linq":
            title = "Linear Q-Learning"
            state_dim = 2
            action_dim = 2
            algorithm = LinearQ(env, state_dim=state_dim, action_dim=action_dim, latency=self.latency)

        return {'title': title, 'alg': algorithm}

    def train(self):
        for alg_i, (_, alg_dict) in enumerate(self.algs.items()):
            print(f"[{alg_dict['title']}] Begin on {self.episodes} episodes...")

            for iter in range(self.iterations):
                if self.verbose: print(f"Iteration {iter}")

                # TODO: This may need to change in the future. We'll need Q/w values for performance testing.
                alg_dict['alg'].reset() # Reset the algorithm for the given iteration

                episodes = tqdm(range(self.episodes)) if self.verbose else range(self.episodes)
                for ep_i in episodes:
                    start_idx = np.random.randint(len(self.train_data))
                    episode_losses, _ = alg_dict['alg'].train_episode(start_idx)
                    self.losses[alg_i][iter][ep_i] = np.sum(episode_losses)

            if self.verbose: print(f"[{alg_dict['title']}] End")

        self.losses = np.array([np.mean(self.losses[alg_i], axis=0) for alg_i in range(len(self.algs))])

    def plot_losses(self):
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, 6))
        palette = sns.color_palette("mako_r", n_colors=len(self.algs))
        palette = sns.color_palette("tab10", n_colors=len(self.algs))
        
        for alg_i, alg_dict in enumerate(self.algs.values()):
            window_size = 500
            rolling = pd.Series(self.losses[alg_i]).rolling(window=window_size, min_periods=1)
            smoothed_losses = rolling.mean()
            std_dev_losses = rolling.std()
            sns.lineplot(
                smoothed_losses,
                label=f'{alg_dict['title']} average (window = {window_size})',
                color = palette[alg_i],
                zorder = 2
            )

            min_episode = 0
            episodes = np.arange(min_episode, min_episode + len(self.losses[alg_i]))
            plt.fill_between(
                episodes,
                smoothed_losses - std_dev_losses,
                smoothed_losses + std_dev_losses,
                color = palette[alg_i],
                alpha = 0.5,
                label = '±1 Std Dev',
                zorder = 1
            )

        if len(self.algs.keys()) == 1:
            # FIXME: This is pretty nasty
            title = f"Carbon Intensities from {list(self.algs.values())[0]['title']} on a {self.job_size}-State MDP" 
        else:
            title = f"Carbon Intensities from algorithms on a {self.job_size}-State MDP"
            # title = f"[{', '.join(alg_dict['title'] for alg_dict in self.algs.values())}] Training with Seed {self.seed}"

        plt.title(title, fontsize=14)
        plt.ylabel("Normalized Carbon Intensity", fontsize=12)
        plt.xlabel("# Episodes", fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.show()
