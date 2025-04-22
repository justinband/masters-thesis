import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from algorithms import LearningAlg
from mdps import JobEnv
from datasets import DataLoader
from utils import generic, plotting

class QLearn(LearningAlg):

    def __init__(self, env: JobEnv, lr=0.01, epsilon=1):
        self.env = env
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.q = np.zeros((self.env.job_size, self.env.nA))

        self.action_dim = [0, 1]

    def _choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            min_q = np.where(self.q[state] == np.min(self.q[state]))[0] # handle duplicates
            return np.random.choice(min_q)
        
    def _update_q_value(self, state, action, loss, s_prime):
        # Loss should come from job_env
        q_delta = np.min(self.q[s_prime]) - self.q[state, action]
        self.q[state, action] += self.lr * (loss + q_delta)

    def train_episode(self, normalize, episode):
        self.env.train()
        start_idx = self.env.get_random_index()
        state = self.env.reset(start_idx, normalize)
        done = False

        total_loss = 0
        total_carbon = 0

        while not done:
            curr_intenisty = self.env.get_carbon()
            action = self._choose_action(state)

            if action == self.env.run:
                total_carbon += curr_intenisty

            next_job_state, loss, done = self.env.step(action)

            self._update_q_value(state, action, loss, next_job_state)
            
            total_loss += loss
            state = next_job_state

        optimal_loss, optimal_carbon, optimal_time = self.env.calc_opt_carbon(start_idx)

        regret = generic.calculate_regret(total_loss, optimal_loss)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Loss: {total_loss:.2f}, Epsilon: {self.epsilon:.2f}")

        return total_loss, total_carbon, optimal_carbon, regret
    
    def evaluate(self, normalize):
        self.env.test()
        start_idx = self.env.get_random_index()
        state = self.env.reset(start_idx, normalize)

        done = False
        total_loss = 0
        total_carbon = 0
        intensity_history = []
        state_history = []
        action_history = []
        q_vals_history = []
        loss_history = []

        for s in range(self.env.job_size):
            print(f'Q in {s}: {self.q[s]}')

        max_time = 1500

        while not done:
            curr_intenisty = self.env.get_carbon()
            intensity_history.append(curr_intenisty)

            state_history.append(state)

            q_vals = self.q[state]
            q_vals_history.append(q_vals)

            action = np.argmin(q_vals)
            action_history.append(action)

            _, loss, done = self.env.step(action)
            loss_history.append(loss)
            total_loss += loss
            if action == self.env.run:
                total_carbon += curr_intenisty

            if self.env.time >= max_time:
                print(f"Evaluation took longer than {max_time} hours")
                done = True

        return total_loss, action_history, intensity_history, state_history, loss_history, q_vals_history, total_carbon

    def get_weights(self):
        return self.q
    
    def load_weights(self, weights):
        self.q = weights

def main():
    print("Running QL")
    # Hyper Parameters
    job_size = 10
    # seed = 100
    alpha = 1e-5
    tradeoff = 0.05 # This proudces interesting results w/ normalize=False
    tradeoff = 0.05
    episodes = 1250
    normalize = True

    # Environment Parameters
    train_size = 0.8
    energy_df = DataLoader().data
    # energy_df = DataLoader(seed=seed).data
    # np.random.seed(seed)
    env = JobEnv(job_size, energy_df, train_size)

    # Agent
    agent = QLearn(env, lr=alpha, tradeoff=tradeoff)

    # Tracking
    losses = []
    carbons = []
    optimal_carbons = []
    regrets = []

    ## Training
    for e in tqdm(range(episodes)):
        loss, carbon, opt_carbon, regret = agent.train_episode(normalize, e)
        losses.append(loss)
        carbons.append(carbon)
        optimal_carbons.append(opt_carbon)
        regrets.append(regret)

    ## Evaluate
    total_loss, action_history, intensity_history, state_history, loss_history, q_vals_history = agent.evaluate(normalize)

    plotting.plot_training_progress(losses)
    plt.show()

    plotting.plot_training_carbons(carbons, optimal_carbons)
    plt.show()

    plotting.plot_evaluation_results(action_history, intensity_history, loss_history, q_vals_history)
    print(f"Total loss: {total_loss:.2f}")
    print(f"Job completed in {len(action_history)} hours")
    plt.show()

if __name__ == '__main__':
    main()