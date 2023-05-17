import multiprocessing
import multiprocessing as mp
import torch
import numpy as np
from torch.distributions import MultivariateNormal


class RolloutProcess(mp.Process):
    def __init__(self, env, queue, return_queue, timesteps_per_episode, gamma):
        super().__init__()

        self.env = env
        self.queue = queue
        self.return_queue = return_queue
        self.timesteps_per_episode = timesteps_per_episode
        self.gamma = gamma

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def run(self) -> None:
        while True:
            actor = self.queue.get()

            t = 0

            batch_obs = []
            batch_acts = []
            batch_log_probs = []
            batch_rews = []
            batch_rtgs = []
            batch_lens = []

            for i in range(3):
                ep_t = 0
                ep_rews = []
                ep_r = 0
                obs = self.env.reset()
                done = False

                while ep_t < self.timesteps_per_episode and not done:
                    batch_obs.append(obs)

                    action, log_prob = self.get_action(actor, obs)
                    obs, rew, done, _ = self.env.step(action)

                    ep_rews.append(rew)
                    ep_r += rew
                    batch_acts.append(action)
                    batch_log_probs.append(log_prob)
                    # t += 1
                    ep_t += 1

                batch_lens.append(ep_t)
                batch_rews.append(ep_rews)
            # batch_obs = np.array(batch_obs)
            # batch_obs = torch.tensor(batch_obs, dtype=torch.float)
            # batch_acts = torch.tensor(batch_acts, dtype=torch.float)
            # batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
            batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

            # Log the episodic returns and episodic lengths in this batch.
            # self.logger['batch_rews'] = batch_rews
            # self.logger['batch_lens'] = batch_lens

            self.return_queue.put((batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews))

    def get_action(self, actor, obs):
        mean = actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs


if __name__ == '__main__':
    multiprocessing.freeze_support()
    exit(1)