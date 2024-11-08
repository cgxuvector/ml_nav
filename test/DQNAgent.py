"""
    Double DQN agent implementation notes:
        - The estimation of the terminal state from the policy network should be mask 0.
        - For double DQN, we have to detach both the target network and the policy network.
        - For hard update target network, we have to update it every fix number of steps. (defaul 2000)
        - For soft update target network, we have to update it every step.
"""
import torch
from torch import nn
import random
import numpy as np
import torch.nn.functional as F
import IPython.terminal.debugger as Debug


# customized weight initialization
def customized_weights_init(m):
    # compute the gain
    gain = nn.init.calculate_gain('relu')
    # init the convolutional layer
    if isinstance(m, nn.Conv2d):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    # init the linear layer
    if isinstance(m, nn.Linear):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)


# create the model
class DeepQNet(nn.Module):
    # init function
    def __init__(self, use_small_obs, use_true_state):
        super(DeepQNet, self).__init__()
        # set the model configuration
        self.use_small_obs = use_small_obs
        self.use_true_state = use_true_state
        if not use_true_state:
            if not self.use_small_obs:
                self.conv_qNet = nn.Sequential(
                    # 3 x 64 x 64 --> 32 x 31 x 31
                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),

                    # 32 x 31 x 31 --> 64 x 14 x 14
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),

                    # 64 x 14 x 14 --> 128 x 6 x 6
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),

                    # 128 x 6 x 6 --> 256 x 2 x 2
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),

                    # 256 x 2x 2 --> 256 x 1 x 1
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU()
                )

                # define the q network
                self.fcNet = nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 4)
                )
            else:
                # if the convolutional flag is enabled.
                self.conv_qNet = nn.Sequential(
                    # 3 x 32 x 32 --> 32 x 14 x 14
                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),

                    # 32 x 14 x 14 --> 64 x 6 x 6
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),

                    # 64 x 6 x 6 --> 128 x 2 x 2
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                )

                # define the q network
                self.fcNet = nn.Sequential(
                    nn.Linear(1024 * 4, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 4),
                    nn.Identity()
                )
        else:
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
        # check whether use the true state
        if not self.use_true_state:
            # check whether use the small observation
            if not self.use_small_obs:
                # compute the state feature
                state_feature = self.conv_qNet(state).view(-1, 1 * 8 * 256)
            else:
                # compute the state feature
                state_feature = self.conv_qNet(state).view(-1, 1 * 8 * 128 * 4)
            x = self.fcNet(state_feature)
        else:
            x = self.fcNet(state)
        return x


# define the learning model
class DQNAgent(object):
    # init the DQN agent
    def __init__(self,
                 dqn_mode='vanilla',
                 target_update_frequency=2000,
                 policy_update_frequency=4,
                 use_small_obs=False,
                 use_true_state=False,
                 use_target_soft_update=False,
                 use_gradient_clip=False,
                 gamma=0.99,
                 learning_rate=1e-3,
                 device="cpu"
                 ):
        """
        Function is used to initialize the DQN agent.
        :param dqn_mode: model of DQN agent (vanilla or double)
        :param target_update_frequency: frequency of updating the target network
        :param policy_update_frequency: frequency of updating the policy network
        :param use_target_soft_update: flag of using soft update
        :param use_gradient_clip: flag of using gradient clip
        :param gamma: discounted factor
        :param learning_rate: learning rate
        :param device: device for computing
        """
        """ Set the device """
        self.device = torch.device(device)
        """ DQN configurations"""
        # create the policy network and target network
        self.policy_net = DeepQNet(use_small_obs, use_true_state)
        self.target_net = DeepQNet(use_small_obs, use_true_state)
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
        self.use_true_state = use_true_state
        self.soft_update = use_target_soft_update
        self.clip_gradient = use_gradient_clip
        self.freq_update_target = target_update_frequency
        self.freq_update_policy = policy_update_frequency
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate,
                                          weight_decay=5e-4)
        self.criterion = torch.nn.MSELoss()
        """ Saving and plotting configurations"""
        self.returns = []
        self.lengths = []

    # select an action based on the policy network
    def get_action(self, input_state, eps):
        if random.uniform(0, 1) < eps:  # with probability epsilon, the agent selects a random action
            action = random.sample(range(4), 1)[0]
        else:  # with probability 1 - epsilon, the agent selects a greedy action
            input_state = self.toTensor(input_state)
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
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # update the policy network
    def update_policy_net(self, batch_data):
        # convert the batch from numpy to tensor
        state, action, next_state, reward, done = self.convert2tensor(batch_data)
        # compute the Q_policy(s, a)
        sa_values = self.policy_net(state).gather(dim=1, index=action)
        # compute the TD target r + gamma * max_a' Q_target(s', a')
        if self.dqn_mode == "vanilla":  # update the policy network using vanilla DQN
            # compute the : max_a' Q_target(s', a')
            max_next_sa_values = self.target_net(next_state).max(1)[0].view(-1, 1).detach()
            # if s' is terminal, then change Q(s', a') = 0
            terminal_mask = (torch.ones(done.size(), device=self.device) - done)
            max_next_sa_values = max_next_sa_values * terminal_mask
            # compute the r + gamma * max_a' Q_target(s', a')
            td_target = reward + self.gamma * max_next_sa_values
        else:  # update the policy network using double DQN
            # select the maximal actions using greedy policy network: argmax_a Q_policy(S_t+1)
            estimated_next_action = self.policy_net(next_state).max(dim=1)[1].view(-1, 1).detach()
            # compute the Q_target(s', argmax_a)
            next_sa_values = self.target_net(next_state).gather(dim=1, index=estimated_next_action).detach().view(-1, 1)
            # convert the value of the terminal states to be zero

            terminal_mask = (torch.ones(done.size(), device=self.device) - done)
            max_next_state_q_values = next_sa_values * terminal_mask
            # compute the TD target
            td_target = reward + self.gamma * max_next_state_q_values

        # compute the loss using MSE error: TD error
        #loss = self.criterion(sa_values, td_target)
        loss = F.smooth_l1_loss(sa_values, td_target)
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        # why clip the gradient
        if self.clip_gradient:
            for params in self.policy_net.parameters():
                params.grad.data.clamp(-1, 1)
        self.optimizer.step()

    # train the agent for one mini-batch
    def train_one_batch(self, t, batch):
        # update the policy network
        if not np.mod(t + 1, self.freq_update_policy):
            self.update_policy_net(batch)

        # update the target network
        if not self.soft_update:
            if not np.mod(t + 1, self.freq_update_target):
                self.update_target_net()
        else:
            self.update_target_net()

    # convert data type into tensor
    def convert2tensor(self, batch):
        if not self.use_true_state:
            if len(batch._fields) == 5:
                state = torch.cat(batch.state, dim=0).float().to(self.device)
                action = torch.cat(batch.action, dim=0).long().to(self.device)
                reward = torch.cat(batch.reward, dim=0).float().to(self.device)
                next_state = torch.cat(batch.next_state, dim=0).float().to(self.device)
                done = torch.cat(batch.done, dim=0).to(self.device)
                return state, action, next_state, reward, done
            elif len(batch._fields) == 6:
                state = torch.cat(batch.state, dim=0).float().to(self.device)
                action = torch.cat(batch.action, dim=0).long().to(self.device)
                reward = torch.cat(batch.reward, dim=0).float().to(self.device)
                next_state = torch.cat(batch.next_state, dim=0).float().to(self.device)
                goal = torch.cat(batch.goal, dim=0).float().to(self.device)
                done = torch.cat(batch.done, dim=0).to(self.device)
                return state, action, next_state, reward, goal, done
        else:
            if len(batch._fields) == 5:
                state = torch.stack(batch.state).float().to(self.device)
                action = torch.stack(batch.action).long().view(-1, 1).to(self.device)
                reward = torch.stack(batch.reward).float().view(-1, 1).to(self.device)
                next_state = torch.stack(batch.next_state).float().to(self.device)
                done = torch.stack(batch.done).long().view(-1, 1).to(self.device)
                return state, action, next_state, reward, done
            elif len(batch._fields) == 6:
                state = torch.stack(batch.state).float().to(self.device)
                action = torch.stack(batch.action).long().view(-1, 1).to(self.device)
                reward = torch.stack(batch.reward).float().view(-1, 1).to(self.device)
                next_state = torch.stack(batch.next_state).float().to(self.device)
                goal = torch.stack(batch.goal).float().to(self.device)
                done = torch.stack(batch.done).long().view(-1, 1).to(self.device)
                return state, action, next_state, reward, goal, done

    def toTensor(self, state):
        if not self.use_true_state:
            state_obs = torch.tensor(np.array(state).transpose(0, 3, 1, 2), dtype=torch.float32, device=self.device)
        else:
            state_obs = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        return state_obs

    # auxiliary function
    @staticmethod
    def rolling_average(data, window_size):
        assert data.ndim == 1
        kernel = np.ones(window_size)
        smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
        return smooth_data[: -window_size + 1]
