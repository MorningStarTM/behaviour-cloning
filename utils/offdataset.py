import os
import random
from tqdm import tqdm

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset

@dataclass
class DataLoader:
    def __init__(self, dataset, gamma=1.0, epsilon=1e-5):
        self.dataset = dataset
        self.epsilon = epsilon
        self.gamma = gamma

        self.states = []
        trajectories_length = []
        for obs in self.dataset["observations"]:
            self.states.append(obs)
            trajectories_length.append(len(obs))
        self.states = np.vstack(self.states)
        self.length = self.states.shape[0]

        mean = np.mean(self.states, axis=0)
        std = np.std(self.states, axis=0) + self.epsilon
        self.states = (self.states - mean) / std

        self.actions = []
        for acts in self.dataset["actions"]:
            self.actions.append(acts)
        self.actions = np.vstack(self.actions)


    def __call__(self, batch_size):
        batch_idxs = np.random.choice(
            self.length,
            size=batch_size,
            replace=True,
        )

        s = self.states[batch_idxs]
        a = self.actions[batch_idxs]
        s = torch.from_numpy(np.vstack(s)).float()
        a = torch.from_numpy(np.vstack(a)).float()

        return {
            "states": s,
            "actions": a,
        }

