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
import gc

import gym
from gym.spaces import Box

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Let ε<0
MATH_EPSILON = -1e-6

# Fake frames
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
        # technically false but whatever
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
        self.observation_space = Box(low = 0, high = 255, shape = obs_shape, dtype = np.uint8)

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
        self.root = np.zeros(capacity, dtype = np.float32) # original values
        self.tree = np.zeros(capacity, dtype = np.float32)
        self.cap  = capacity
        self.size = 0

    # sum[0:n+1]
    def sum(self, n):
        total = 0
        while n >= 0:
            total += self.tree[n]
            n = (n & (n + 1)) - 1
        return total

    # get raw values
    def fetch(self, n):
        return self.root[n]

    # Just use this
    def assign(self, n, value):
        current = self.root[n]
        if current == 0.0:
            self.size += 1
            assert self.size <= self.cap
        self.increase(n, value - current)
        self.root[n] = value

    # updates the Fenwick tree
    def increase(self, n, amount):
        while n < self.cap:
            self.tree[n] += amount
            n = n | (n + 1)

    # binary search
    def retrieve(self, upperbound):
        left, right = 0, self.size - 1
        while left < right:
            mid = (left + right) // 2
            if self.sum(mid) < upperbound:
                left = mid + 1
            else:
                right = mid
        return left

class ReplayBuffer:
    def __init__(self, capacity, obs_size, batch_size = 128,
        # PER
        alpha = 0.6
    ):
        self.obs        = np.zeros([capacity, *obs_size], dtype=np.float32)
        self.next_obs   = np.zeros([capacity, *obs_size], dtype=np.float32)
        self.acts       = np.zeros([capacity], dtype=np.float32)
        self.rewards    = np.zeros([capacity], dtype=np.float32)
        self.dones      = np.zeros(capacity, dtype=np.float32)

        self.capacity   = capacity
        self.ptr        = 0
        self.size       = 0
        self.batch_size = batch_size

        self.priorities   = Fenwick(capacity)
        self.max_priority = 1.0
        self.alpha        = alpha

    def __len__(self):
        return self.size

    def add(self, obs, act, next_obs, reward, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.next_obs[self.ptr] = next_obs
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.priorities.assign(self.ptr, self.max_priority ** self.alpha)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, beta = 0.4):
        assert len(self) >= self.batch_size

        p_total = self.priorities.sum(len(self) - 1)

        # get some indices with weights
        def sample_proportional():
            indices = []
            segment = p_total / self.batch_size

            for i in range(self.batch_size):
                a = segment * i
                b = segment * (i + 1)
                upperbound = random.uniform(a, b)
                idx = self.priorities.retrieve(upperbound)
                assert idx < len(self)
                indices.append(idx)

            return indices

        indices  = sample_proportional()

        obs      = self.obs[indices]
        next_obs = self.next_obs[indices]
        acts     = self.acts[indices]
        rewards  = self.rewards[indices]
        dones    = self.dones[indices]

        ps       = self.priorities.root[indices] / p_total
        weights  = (len(self) * ps) ** (-1 * beta)
        weights  = weights / weights.max()

        return dict(
            obs      = obs,
            next_obs = next_obs,
            acts     = acts,
            rewards  = rewards,
            dones    = dones,
            weights  = weights,
            indices  = indices,
        )

    def update(self, indices, priorities):
        for i, pr in zip(indices, priorities):
            new_pr = pr ** self.alpha
            self.priorities.assign(i, new_pr)
            self.max_priority = max(self.max_priority, new_pr)