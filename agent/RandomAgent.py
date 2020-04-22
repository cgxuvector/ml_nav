import torch
from torch import nn
import numpy as np


class RandomAgent(nn.Module):
    def __init__(self, action_space, random_seed):
        super(RandomAgent, self).__init__()
        self.action_space = action_space
        self.seed_rnd = random_seed
        self.policy_net = torch.nn.Linear(2, 1)

    def get_action(self, state, goal):
        return self.action_space.sample()

    def train_one_batch(self, batch):
        pass

    def forward(self, x):
        return self.policy_net(x)


class RandomAgentDM(object):
    def __init__(self, random_seed, stride):
        # set the seed
        np.random.seed(random_seed)
        # define the action space
        self.action_space = [
            np.array([-stride,  0, 0]),  # up
            np.array([stride,  0, 0]),  # down
            np.array([0, -stride, 0]),  # left
            np.array([0,  stride, 0])   # right
        ]

    def get_random_action(self):
        return np.random.choice(self.action_space, 1).item()

    def get_specific_action(self, act_name):
        if act_name == "up":
            return self.action_space[0]
        elif act_name == "down":
            return self.action_space[1]
        elif act_name == "left":
            return self.action_space[2]
        elif act_name == "right":
            return self.action_space[3]
        else:
            raise Exception(f"Invalid action name. Expected 'up', 'down', 'left', 'right', but get {act_name}")