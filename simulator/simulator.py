from datasets import DataLoader
from mdps import JobEnv
from algorithms import QLearn, LinearQLearning, RunAgent
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
import pandas as pd
import wandb
import joblib
import os
from utils import plotting
import pprint
from utils import generic

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
        dataloader = DataLoader(seed=seed)
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
            if self.wandb_log:
                config = {
                    "learning_rate": agent.lr,
                    "episodes": self.episodes,
                    "job_size": self.job_size,
                    "alpha": self.alpha
                }
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

            self._save_model(agent, alg_title)
        print("End training...")

    def _add_baseline_alg(self):
        agent = RunAgent(self.env)
        key = 'run-agent'
        value = {'title': 'Run-Only Agent', 'alg': agent}
        self.algs.setdefault(key, value)

    def evaluate(self):
        ##### TESTING
        ##### FIXME: THIS SHOULD BE REMOVED
        start_idx = 6500
        #####
        #####

        # Add run-only alg to the algorithms
        self._add_baseline_alg()
        print(self.algs)

        results = {}

        for alg_i, (alg_title, alg_dict) in enumerate(self.algs.items()):
            print(f"Evaluating {alg_title}...")
            agent = alg_dict['alg']
            total_loss, action_history, intensity_history, state_history, loss_history, q_vals_history, total_carbon = agent.evaluate(start_idx)

            plotting.plot_evaluation_results(
                actions=action_history,
                intensities=intensity_history,
                losses=loss_history,
                q_vals=q_vals_history,
                title=alg_dict['title']
            )
            print(f"[{alg_title}] Total loss: {total_loss:.2f}")
            print(f"[{alg_title}] Total Carbon: {total_carbon:.5f}")
            print(f"[{alg_title}] Job completed in {len(action_history)} hours")
            results[alg_title] = {
                'loss': np.round(total_loss, 2),
                'carbon': np.round(total_carbon, 5),
                'hours': len(action_history)}

        print("--- Results ---")
        pprint.pprint(results)
    
        baseline = 'run-agent'
        for key in results:
            if key != baseline:
                carbon_diff = generic.calculate_diff(results[baseline]['carbon'], results[key]['carbon'])
                loss_diff = generic.calculate_diff(results[baseline]['loss'], results[key]['loss'])
                # time_diff = generic.calculate_diff(results[baseline]['hours'], results[key]['hours'])
                print(f'Loss difference from run-agent to {key} = {loss_diff}%')
                print(f'Carbon difference from run-agent to {key} = {carbon_diff}%')
                # print(f'Time difference from run-agent to {key} = {time_diff}')

        plt.show()

    def _save_model(self, model, model_name):
        model_dir = f"models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        
        # Dump model
        weights = model.get_weights()
        joblib.dump(weights, model_path)

    def _load_model(self, model, model_name):
        model_path = f"models/{model_name}.pkl"
        if os.path.exists(model_path):
            weights = joblib.load(model_path)
            model.load_weights(weights)
            print(f"Loaded saved {model_name} model")

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