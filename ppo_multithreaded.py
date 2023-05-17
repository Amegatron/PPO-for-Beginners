import multiprocessing

import numpy as np
import gym
import torch
import torch.nn as nn
from ppo import PPO
from rollout import RolloutProcess
import multiprocessing as mp


class PPOMultithreaded(PPO):
    def __init__(self, policy_class, env, **hyperparameters):
        self.rollout_processes = []
        self.queue = mp.Queue()
        self.return_queue = mp.Queue()
        self.processes = 4
        super().__init__(policy_class, env, **hyperparameters)

    def _init_hyperparameters(self, hyperparameters):
        super()._init_hyperparameters(hyperparameters)

    def learn(self, total_timesteps):
        rollout_processes = []
        self.queue = mp.Queue(maxsize=self.processes)
        self.return_queue = mp.Queue(maxsize=self.processes)

        print(f"Starting {self.processes} child processes...")
        for i in range(self.processes):
            p = RolloutProcess(self.env, self.queue, self.return_queue, self.max_timesteps_per_episode, self.gamma)
            rollout_processes.append(p)
            p.start()

        self._learn(total_timesteps)

        print("Stopping child processes...", end='')
        for i in range(self.processes):
            rollout_processes[i].terminate()

        print("")

    def _learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        queue = mp.Queue(1)
        # queue.put(self.actor)

        print("Launching render thread...")
        self.renderer_process = mp.Process(
            target=self.render_env, args=[self.env, queue]
        )
        self.renderer_process.start()

        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            self.actor.to(self.device)
            self.critic.to(self.device)

            obs_cuda = batch_obs.to(self.device)
            acts_cuda = batch_acts.to(self.device)

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(obs_cuda, acts_cuda)
            A_k = batch_rtgs - V.detach()

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(obs_cuda, acts_cuda)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            self.actor.to(self.device_cpu)
            self.critic.to(self.device_cpu)

            try:
                tmp_actor = queue.get_nowait()
            except:
                pass

            queue.put(self.actor)

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def rollout(self):
        if not self.queue.empty():
            raise ChildProcessError("Queue is not empty after previous run")

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        for i in range(self.processes):
            self.queue.put(self.actor)

        for i in range(self.processes):
            result = self.return_queue.get()

            batch_obs.extend(result[0])
            batch_acts.extend(result[1])
            batch_log_probs.extend(result[2])
            batch_rtgs.extend(result[3])
            batch_lens.extend(result[4])
            batch_rews.extend(result[5])

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        # batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


if __name__ == '__main__':
    multiprocessing.freeze_support()
    exit(1)