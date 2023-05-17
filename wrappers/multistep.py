from collections import deque

import gym
import numpy as np


class MultiStepWrapper(gym.Env):
    def __init__(self, base_env, steps: int):
        self.base_env = base_env

        assert steps > 0
        self.steps = steps
        self.step_obs = deque([])  # Just for type-hinting

    def step(self, action):
        obs, rew, done, info = self.base_env.step(action)
        self.step_obs.append(obs)

        return np.array(self.step_obs), rew, done, info

    def reset(self):
        obs = self.base_env.reset()
        self.step_obs = deque([obs] * self.steps, maxlen=self.steps)

        return np.array(self.step_obs)

    def render(self, mode='human'):
        self.base_env.render(mode=mode)
