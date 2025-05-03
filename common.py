# this works like a header right
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

import gym
from gym.spaces import Box

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

# Fake frames let's goooooo
class NvidiaFrameGeneration(gym.Wrapper):
    def __init__(self, env, n_frame):
        super().__init__(env)
        self.skip = n_frame

    def step(self, action):
        total_reward = 0
        # make fake frames (where we do the same thing)
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

# state: [240, 256, 3] -> [240, 256]
class Grayscaler(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low = 0, high = 255, shape = obs_shape, dtype = np.uint8)

    def reorient(self, observation):
        # Why did they put it in this order
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.reorient(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

# [240, 256] -> [91, 91]
class Downscaler(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape

        self.observation_space = Box(low = 0, high = 255, shape = shape, dtype = np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias = True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

# [91, 91] -> [4, 91, 91]
class FrameStacker(gym.Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen = num_stack)

        obs_shape = (num_stack, *self.env.observation_space.shape)
        self.observation_space = gym.spaces.Box(
            low = 0, high = 255,
            shape = obs_shape,
            dtype = np.uint8
        )

    def reset(self):
        state = self.env.reset()
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(state)
        return np.stack(self.frames, axis = 0)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.frames.append(state)
        return np.stack(self.frames, axis = 0), reward, done, info

# It's time to d-d-d-d-duel
class Yugi(nn.Module):
    def __init__(self, obs, act):
        super().__init__()

        ch, _, _ = obs

        self.convolute = nn.Sequential(
            nn.Conv2d(in_channels = ch, out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.stt_layer = nn.Linear(512, 1)
        self.adv_layer = nn.Linear(512, act)

    def forward(self, x):
        features = self.convolute(x)
        stt = self.stt_layer(features)
        adv = self.adv_layer(features)
        return stt + adv - adv.mean(dim = 1, keepdim = True)

# Fenwick Tree
class Fenwick:
    def __init__(self, capacity):
        self.root = np.zeros(capacity, dtype = np.float64) # original values
        self.tree = np.zeros(capacity, dtype = np.float64)
        self.cap  = capacity

    # sum[0:n+1]
    def sum(self, n):
        total = 0
        while n >= 0:
            total += self.tree[n]
            n = (n & (n + 1)) - 1
        return total

    # Just use this
    def assign(self, n, value):
        current = self.root[n]
        self.increase(n, value - current)
        self.root[n] = value

    # updates the Fenwick tree
    def increase(self, n, amount):
        while n < self.cap:
            self.tree[n] += amount
            n = n | (n + 1)


# to be enhanced
class ReplayBuffer:
    def __init__(self, capacity, batch_size = 128):
        self.buffer = TensorDictReplayBuffer(storage = LazyMemmapStorage(capacity), batch_size = batch_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, act, next_state, reward, done):

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        td = TensorDict({
            "states": torch.tensor(state),
            "acts": torch.tensor([act]),
            "states2": torch.tensor(next_state),
            "rewards": torch.tensor([reward]),
            "dones": torch.tensor([done]),
        }, batch_size=[])

        self.buffer.add(td)

    def sample(self):
        return self.buffer.sample()
