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
            self.algs[a] = self._create_alg(a, mdp)

        self.losses = np.empty((len(self.algs), self.iterations, self.episodes))
        self.optimal_losses = np.empty((len(self.algs), self.episodes))

        self.latencies = np.empty((len(self.algs)), dtype=object)
        self.intensities = np.empty((len(self.algs)), dtype=object)
        self.max_time = 0

    def _create_alg(self, alg_name, env):
        alg_map = {
            "ql": ("Q-Learning", QLearn),
            "iql": ("Informed Q-Learning", InformedQL),
            "linq": ("Linear Q-Learning", LinearQ),
        }
        title, algorithm_class = alg_map.get(alg_name)
        algorithm = algorithm_class(env, latency=self.latency)
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
            print(f"[{alg_dict['title']}] Begin on {self.episodes} episodes...")

            iter_latencies = []
            iter_intensities = []

            for iter in range(self.iterations):
                if self.verbose: 
                    print(f"Iteration {iter}")

                # TODO: This may need to change in the future. We'll need Q/w values for performance testing.
                alg_dict['alg'].reset() # Reset the algorithm for the given iteration

                ### Episode Loop
                episodes = tqdm(range(self.episodes)) if self.verbose else range(self.episodes)
                latencies = []
                intensities = []

                for ep_i in episodes:
                    start_idx = np.random.randint(len(self.train_data))
                    episode_losses, episode_latencies, ep_intensities, ep_time = alg_dict['alg'].run_episode(start_idx, train=True)

                    ### Trackers
                    self.losses[alg_i][iter][ep_i] = np.sum(episode_losses)
                    latencies.append(episode_latencies)
                    intensities.append(ep_intensities)

                    ep_intensities.sort()
                    optimal_loss = np.sum(ep_intensities[:self.job_size])
                    self.optimal_losses[alg_i, ep_i] = optimal_loss
                      
                    if ep_time > self.max_time:
                        self.max_time = ep_time


                # Pad latencies for current iteration
                padded_latencies = self._pad_data(latencies, self.max_time + 1)
                iter_latencies.append(np.nanmean(padded_latencies, axis=0))

                padded_intensities = self._pad_data(intensities, self.max_time + 1)
                iter_intensities.append(np.nanmean(padded_intensities, axis=0))

            if self.verbose:
                print(f"[{alg_dict['title']}] End")

            # Store padded latencies
            self.latencies[alg_i] = self._pad_data(iter_latencies, self.max_time + 1)
            self.intensities[alg_i] = self._pad_data(iter_intensities, self.max_time + 1)
        
        # Average over iterations
        self.losses = np.array([np.mean(self.losses[alg_i], axis=0) for alg_i in range(len(self.algs))])
        self.latencies = np.array([np.nanmean(self.latencies[alg_i], axis=0) for alg_i in range(len(self.algs))])
        self.intensities = np.array([np.nanmean(self.intensities[alg_i], axis=0) for alg_i in range(len(self.algs))])

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
        plt.ylabel("Normalized Carbon Intensity", fontsize=12)
        plt.xlabel("# Episodes", fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.show()


    def plot_latency(self):
        sns.set_theme(style='darkgrid')
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # Plot Latency
        for alg_i, alg_dict in enumerate(self.algs.values()):
            ax1.plot(self.latencies[alg_i], label=f'Latency', color='red')

        for alg_i, alg_dict in enumerate(self.algs.values()):
            ax2.plot(self.intensities[alg_i], label=f'CI', color='blue')
            
        ax1.set_ylabel("Average Latency")
        ax1.legend(loc="best")
        ax2.set_ylabel("Carbon Intensity")
        ax2.legend(loc="best")

        plt.title("Latencies", fontsize=14)
        plt.xlabel("Time", fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.show()

        # Plot carbon intensity over time

        # Plot latency over time, averaged over each episode
