import torch
from torch import nn


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
