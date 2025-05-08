from datasets import DataLoader
from mdps import JobEnv
from algorithms import QLearn, LinearQLearning, RunAgent
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import wandb
from . import evaluator
from . import utils

class Simulator():
    def __init__(self, algs, job_size, episodes, alpha, lr, iterations, verbose, normalized, seed=None):
        self.seed = np.random.randint(0, 2**36 - 1) if seed is None else seed
        self.job_size = job_size
        self.episodes = episodes
        self.alpha = alpha
        self.lr = lr
        self.iterations = iterations
        self.verbose = verbose
        self.normalize = normalized
        self.train_size = 0.8
        
        # Flags
        self.wandb_log = False

        # Environment
        dataloader = DataLoader(seed=seed, train_size=self.train_size, alpha=alpha)
        self.env = JobEnv(job_size, alpha, dataloader, self.normalize, self.train_size)

        # Algorithms
        self.algs = {}
        for a in algs:
            self.algs[a] = self._create_alg(a, self.env, self.lr)

        # Trackers
        num_algs = len(self.algs)
        self.losses = np.empty((num_algs, episodes))
        self.carbons = np.empty((num_algs, episodes))
        self.optimal_losses = np.empty((num_algs, episodes))
        self.optimal_carbons = np.empty((num_algs, episodes))
        self.regrets = np.empty((num_algs, episodes))

        self.intensities = np.empty((len(self.algs)), dtype=object)
        self.max_time = 0

        self.cum_regret = np.zeros((len(self.algs), self.episodes))
        self.temp_cum_regret = np.zeros((len(self.algs), self.episodes))
        self.instant_regret = np.zeros((len(self.algs), self.episodes))

    def _create_alg(self, alg_name, env, lr):
        alg_map = {
            "ql": ("Q-Learning", QLearn),
            "lfa-ql": ("Linear Q-Learning", LinearQLearning),
            "run-only": ("Run-Only Agent", RunAgent)
        }
        title, algorithm_class = alg_map.get(alg_name)
        algorithm = algorithm_class(env, lr=lr)
        return {'title': title, 'alg': algorithm}
    
    def _pad_data(self, data, max_length):
        """
        Data represents a list of episodes.

        Each episode contains a list of carbon intensities or latencies.
        """
        return np.array([
            np.pad(item, (0, max_length - len(item)), constant_values=item[-1])
            for item in data
        ])

    def train(self):
        print("Start training...")
        for alg_i, (alg_title, alg_dict) in enumerate(self.algs.items()):
            agent = alg_dict['alg']
            config = {
                    "learning_rate": agent.lr,
                    "episodes": self.episodes,
                    "job_size": self.job_size,
                    "alpha": self.alpha
                } 
            if self.wandb_log:
                wandb.init(
                    project="carbon-scheduling",
                    name=f'{alg_title}_job{self.job_size}_alpha{self.alpha}',
                    config=config
                )

            print(f"[{alg_dict['title']}] {self.episodes} episodes. LR = {self.lr}")

            episodes = tqdm(range(self.episodes)) if self.verbose else range(self.episodes)

            for e in episodes:
                loss, carbon, opt_carbon, regret = agent.train_episode(e)

                self.losses[alg_i][e] = loss
                self.carbons[alg_i][e] = carbon
                self.optimal_carbons[alg_i][e] = opt_carbon
                self.regrets[alg_i][e] = regret

                if self.wandb_log:
                    wandb.log({
                        "episode": e,
                        "train_loss": loss,
                        "carbon": carbon,
                        "optimal_carbon": opt_carbon,
                        "regret": regret
                    })
        
            if self.verbose:
                print(f"[{alg_dict['title']}] End")

            utils.save_model(agent, alg_title, config)
        print("End training...")

    def evaluate(self, iterations=6000):
        self.env.test()
        config = {
            'job_size': self.job_size,
            'alpha': self.alpha,
            'learning_rate': self.lr,
            'episodes': self.episodes
        }
        evaluator.evaluate(self.env, iterations, self.algs, config) 


    def plot_losses(self):
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, 6))
        palette = sns.color_palette("mako_r", n_colors=len(self.algs))
        palette = sns.color_palette("tab10", n_colors=len(self.algs))
        
        for alg_i, alg_dict in enumerate(self.algs.values()):
            window_size = 500
            rolling = pd.Series(self.losses[alg_i]).rolling(window=window_size, min_periods=1)
            smoothed_losses = rolling.mean()
            std_dev_losses = rolling.std().fillna(0) # fillna needed for when in pos 0 and no other data for std calc

            # plt.plot(self.optimal_losses[alg_i], label='optimal', color='red')

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
        ylabel = "Normalized Carbon Intensity" if self.normalize else "Carbon Intensity"
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel("# Episodes", fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.show()