from algorithms import QLearn

class InformedQL(QLearn):
    def __init__(self, env, epsilon = 0.1, alpha = 1e-6, latency=0):
        super().__init__(env, epsilon, alpha, latency)

    def train_episode(self, start_idx):
        state = self.env.reset(start_idx)
        is_done = False
        episode_losses = []
        episode_latencies = []
        episode_intensities = []


        while not is_done:
            # (0) get latency
            curr_latency = self.env.get_latency()
            episode_latencies.append(curr_latency)
            episode_intensities.append(self.env.get_intensity())

            # (1) observe the next state
            s_prime = self.env.get_next_state()

            # (2) what would happen if we performed each action?
            actions = [self.env.pause, self.env.run]
            pause_loss = self.env.get_loss(a=self.env.pause)
            run_loss = self.env.get_loss(a=self.env.run)
            losses = [pause_loss, run_loss]

            # (3) Copy and Update Q-Values given some context
            q_copy = self.q.copy()
            for (a, l) in zip(actions, losses):
                q_copy[state, a] = self.update_q_value(state, a, l, s_prime)
            
            # (4) Choose action based on updated context and current state
            action = self.choose_action(q_copy, state, curr_latency)
            next_state, loss, is_done = self.env.step(action)

            # (5) Update Q-Val based on chosen action
            self.q[state, action] = q_copy[state, action]
            
            # (6) Update counters
            episode_losses.append(loss)
            state = next_state

        return episode_losses, episode_latencies, episode_intensities, self.env.time

        