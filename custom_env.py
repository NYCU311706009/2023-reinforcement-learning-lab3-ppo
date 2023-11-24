from typing import Tuple
import gym
from gym.core import Env
import torch


class ReacherRewardWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminate, truncate, info = super().step(action)
        if action == 1:
            reward += 0.01
        return observation, reward, terminate, truncate, info
