from datasets import DataLoader
from mdps import JobEnv
from algorithms import NewQLearn, InformedQL, LinearQ
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
import pandas as pd

class Simulator():
    def __init__(self, algs, job_size, episodes, tradeoff, learning_rate, iterations, verbose, normalized, seed=None):
        self.seed = np.random.randint(0, 2**36 - 1) if seed is None else seed
        
        self.energy = DataLoader(seed=seed)
        self.job_size = job_size
        self.episodes = episodes
        self.tradeoff = tradeoff
        self.lr = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        self.normalize = normalized
        self.train_size = 0.8

        # self.train_data, self.val_data = self.energy.split_data(train_size=0.8)
        self.losses = []

        mdp = JobEnv(job_size, self.energy.data, self.train_size)
        self.algs = {}
        for a in algs:
            self.algs[a] = self._create_alg(a, mdp)

        self.losses = np.empty((len(self.algs), self.iterations, self.episodes))
        self.optimal_losses = np.empty((len(self.algs), self.episodes))

        self.intensities = np.empty((len(self.algs)), dtype=object)
        self.max_time = 0

        self.cum_regret = np.zeros((len(self.algs), self.episodes))
        self.temp_cum_regret = np.zeros((len(self.algs), self.episodes))
        self.instant_regret = np.zeros((len(self.algs), self.episodes))

    def _create_alg(self, alg_name, env):
        alg_map = {
            "ql": ("Q-Learning", NewQLearn),
            "iql": ("Informed Q-Learning", InformedQL),
            "linq": ("Linear Q-Learning", LinearQ),
        }
        title, algorithm_class = alg_map.get(alg_name)
        algorithm = algorithm_class(env, lr=self.lr, tradeoff=self.tradeoff)
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
        for alg_i, (_, alg_dict) in enumerate(self.algs.items()):
            print(f"[{alg_dict['title']}] {self.episodes} episodes. LR = {self.lr}")

            iter_intensities = []

            for iter in range(self.iterations):
                if self.verbose: 
                    print(f"Iteration {iter}")

                ### Episode Loop
                episodes = tqdm(range(self.episodes)) if self.verbose else range(self.episodes)
                intensities = np.empty(self.episodes, dtype=object)

                for ep_i in episodes:
                    # ep_loss, opt_loss, regret, ep_intensities, ep_time = alg_dict['alg'].run_episode(start_idx, ep_i, train=True)
                    ep_loss, ep_carbon, opt_carbon, ep_time = alg_dict['alg'].train_episode(self.normalize, ep_i)

                    if ep_time > self.max_time: # update max time, used for padding
                        self.max_time = ep_time

                    ### Trackers
                    intensities[ep_i] = ep_loss
                    self.losses[alg_i][iter][ep_i] = ep_loss

                    # self.losses[alg_i][iter][ep_i] = ep_loss
                    # self.optimal_losses[alg_i, ep_i] = opt_loss

                iter_intensities.append(intensities)

                # Pad intensities for current iteration
                # padded_intensities = self._pad_data(intensities, self.max_time + 1)
                # iter_intensities.append(np.nanmean(padded_intensities, axis=0))

            if self.verbose:
                print(f"[{alg_dict['title']}] End")

            # self.intensities[alg_i] = self._pad_data(iter_intensities, self.max_time + 1)
        
        # Average over iterations
        self.losses = np.array([np.mean(self.losses[alg_i], axis=0) for alg_i in range(len(self.algs))])
        # self.intensities = np.array([np.nanmean(self.intensities[alg_i], axis=0) for alg_i in range(len(self.algs))])

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