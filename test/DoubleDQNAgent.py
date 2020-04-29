import torch
from torch import nn
import random
import numpy as np
import IPython.terminal.debugger as Debug

# exponentially decay vs linearly decay


# customized weight initialization
def customized_weights_init(model):
    # compute the gain
    gain = nn.init.calculate_gain('relu')
    # init the params
    classname = model.__class__.__name__
    if classname.find('Linear') != -1:
        # init the weights
        nn.init.xavier_uniform_(model.weight, gain=gain)
        # init the bias
        nn.init.constant_(model.bias, 0)


# create the model
class DeepQNet(nn.Module):
    # init function
    def __init__(self):
        super(DeepQNet, self).__init__()
        # feed forward network
        self.fcNet = nn.Sequential(
                nn.Linear(3, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 4),
                nn.Identity()
        )

    # forward function
    def forward(self, state):
        x = self.fcNet(state)
        return x

# # test code
# testModel = DeepQNet()
# testModel.apply(customized_weights_init)
# for param in testModel.parameters():
#     print(param.size(), param)
# test_input = torch.randn((1, 3), dtype=torch.float32)
# test_output = testModel(test_input)
# print(test_output, test_output.size())


class DQNAgent(object):
    def __init__(self,
                 dqn_mode='vanilla',
                 target_update_frequency=2000,
                 policy_update_frequency=4,
                 use_target_soft_update=False,
                 use_gradient_clip=False,
                 gamma=0.995,
                 learning_rate=1e-3,
                 device="cpu"
                 ):
        self.device = torch.device(device)
        """ DQN configurations"""
        # create the policy network and target network
        self.policy_net = DeepQNet()
        self.target_net = DeepQNet()
        # init the weights of policy network and target network
        self.policy_net.apply(customized_weights_init)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # move the networks to device
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)
        # DQN mode: vanilla or double
        self.dqn_mode = dqn_mode
        """ Training configurations """
        self.gamma = gamma
        self.tau = 0.05  # parameters for soft target update
        self.soft_update = use_target_soft_update
        self.freq_update_target = target_update_frequency
        self.freq_update_policy = policy_update_frequency
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate,
                                          weight_decay=5e-4)
        self.criterion = torch.nn.MSELoss()
        self.clip_gradient = use_gradient_clip
        """ Saving and plotting configurations"""
        self.returns = []
        self.lengths = []

    # select an action based on the policy network
    def get_action(self, input_state, eps):
        if random.uniform(0, 1) < eps:  # with probability 1 - epsilon, the agent selects a random action
            action = random.sample(range(4), 1)[0]
        else:  # with probability epsilon, the agent selects a greedy action
            input_state = torch.tensor(input_state).float().to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(input_state).view(1, -1)
                action = q_values.max(dim=1)[1].item()
        return action

    # update the target network
    def update_target_net(self):
        # hard update
        if not self.soft_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:  # soft update
            for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
                target_param.data.copy_((1-self.tau) * param.data + self.tau * target_param.data)

    # update the policy network
    def update_policy_net(self, batch_data):
        # convert the batch from numpy to tensor
        state, action, next_state, reward, done = self.convert2tensor(batch_data)
        # compute the Q_policy(s, a)
        sa_values = self.policy_net(state).gather(dim=1, index=action)
        # compute the TD target r + gamma * max_a' Q_target(s', a')
        if self.dqn_mode == "vanilla":  # update the policy network using vanilla DQN
            # compute the : max_a' Q_target(s', a')
            max_next_sa_values = self.target_net(next_state).max(1)[0].detach().view(-1, 1)
            # if s' is terminal, then change Q(s', a') = 0
            terminal_mask = (torch.ones(done.size(), device=self.device) - done)
            max_next_sa_values = max_next_sa_values * terminal_mask
            # compute the r + gamma * max_a' Q_target(s', a')
            td_target = reward + self.gamma * max_next_sa_values
        else:  # update the policy network using double DQN
            # select the maximal actions using greedy policy network: argmax_a Q_policy(S_t+1)
            estimate_next_sa_values = self.policy_net(next_state).detach()
            next_action = estimate_next_sa_values.max(dim=1)[1].view(-1, 1)
            # compute the Q_target(s', argmax_a)
            next_sa_values = self.target_net(next_state).gather(dim=1, index=next_action).view(-1, 1)
            # convert the value of the terminal states to be zero
            terminal_mask = (torch.ones(done.size(), device=self.device) - done)
            max_next_state_q_values = next_sa_values * terminal_mask
            # compute the TD target
            td_target = reward + self.gamma * max_next_state_q_values

        # compute the loss using MSE error: TD error
        loss = self.criterion(sa_values, td_target)
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        # why clip the gradient
        if self.clip_gradient:
            for params in self.policy_net.parameters():
                params.grad.data.clamp(-1, 1)
        self.optimizer.step()

    def train_one_batch(self, t, batch):
        # update the policy network
        if not np.mod(t + 1, self.freq_update_policy):
            self.update_policy_net(batch)

        # update the target network
        if np.mod(t + 1, self.freq_update_target):
            self.update_target_net()

    def convert2tensor(self, batch):
        if len(batch._fields) == 5:
            state = torch.stack(batch.state).float().to(self.device)
            action = torch.stack(batch.action).long().view(-1, 1).to(self.device)
            reward = torch.stack(batch.reward).float().view(-1, 1).to(self.device)
            next_state = torch.stack(batch.next_state).float().to(self.device)
            done = torch.stack(batch.done).view(-1, 1).to(self.device)
            return state, action, next_state, reward, done
        elif len(batch._fields) == 6:
            state = torch.stack(batch.state).float().to(self.device)
            action = torch.stack(batch.action).long().view(-1, 1).to(self.device)
            reward = torch.stack(batch.reward).float().view(-1, 1).to(self.device)
            next_state = torch.stack(batch.next_state).float().to(self.device)
            goal = torch.stack(batch.goal).float().to(self.device)
            done = torch.stack(batch.done).view(-1, 1).to(self.device)
            return state, action, next_state, reward, goal, done