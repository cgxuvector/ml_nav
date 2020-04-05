import torch
from torch import nn
import numpy as np
from utils import mapper
import IPython.terminal.debugger as Debug 

class GoalDeepQNet(nn.Module):
    """
        Define the Q network:
            - No image / image feature input: fully-connected (3 layer implemented)
            - Image input: convolutional (not implemented)
    """
    def __init__(self):
        super(GoalDeepQNet, self).__init__()
        # if the convolutional flag is enabled.
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
            nn.Linear(2048 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4)
        )

    def forward(self, state, goal):
        # compute state embedding
        state_fea = self.conv_qNet(state).view(-1, 1 * 8 * 256)
        goal_fea = self.conv_qNet(goal).view(-1, 1 * 8 * 256)
        # concatenate the tensor
        state_goal_fea = torch.cat((state_fea, goal_fea), dim=1)
        # concatenate the goal with state
        x = self.fcNet(state_goal_fea)
        return x


class GoalDQNAgent(object):
    def __init__(self,
                 target_update_frequency,
                 policy_update_frequency,
                 soft_target_update_tau=0.01,
                 learning_rate=1e-3,
                 dqn_mode="vanilla",
                 gamma=1.0,
                 gradient_clip=True,
                 device="cpu"
                 ):
        """
        Init the DQN agent
        :param target_update_frequency: frequency of updating the target net (time steps)
        :param policy_update_frequency: frequency of updating the policy net (time steps)
        :param soft_target_update_tau: soft update params
        :param learning_rate: learning rate
        :param dqn_mode: dqn mode: vanilla, double
        :param gamma: gamma
        :param gradient_clip: if true, gradient clip will be applied
        :param device: device to use
        """
        self.device = torch.device(device)
        """ DQN configurations"""
        # create the policy network and target network
        self.policy_net = GoalDeepQNet()
        self.target_net = GoalDeepQNet()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)
        # DQN mode: vanilla or double
        self.dqn_mode = dqn_mode
        """ Training configurations """
        self.gamma = gamma
        self.tau = soft_target_update_tau  # parameters for soft target update
        self.soft_update = True if soft_target_update_tau else False  # flag for soft update
        self.freq_update_target = target_update_frequency
        self.freq_update_policy = policy_update_frequency
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate,
                                          weight_decay=5e-4)
        self.criterion = torch.nn.MSELoss()
        self.clip_gradient = gradient_clip
        """ Saving and plotting configurations"""
        self.returns = []
        self.lengths = []

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
        state, action, next_state, reward, goal, done = self.convert2tensor(batch_data)
        # compute the Q_policy(s, a)
        state_goal_q_values = self.policy_net(state, goal).gather(dim=1, index=action)
        # compute the TD target r + gamma * max_a' Q_target(s', a')
        if self.dqn_mode == "vanilla":  # update the policy network using vanilla DQN
            with torch.no_grad():
                # compute the : max_a' Q_target(s', a')
                max_next_state_goal_q_values = self.target_net(next_state, goal).max(1)[0].view(-1, 1)
                # if s' is terminal, then change Q(s', a') = 0
                terminal_mask = (torch.ones(done.size(), device=self.device) - done)
                max_next_state_goal_q_values = max_next_state_goal_q_values * terminal_mask
                # compute the r + gamma * max_a' Q_target(s', a')
                td_target = (reward + self.gamma * max_next_state_goal_q_values)
        else:  # update the policy network using double DQN
            with torch.no_grad():
                # select the maximal actions using greedy policy network: argmax_a Q_policy(S_t+1)
                estimate_next_state_goal_q_values = self.policy_net(next_state, goal)
                next_action_data = estimate_next_state_goal_q_values.max(1)[1].view(-1, 1)
                # compute the Q_target(s', argmax_a)
                next_state_goal_q_values = self.target_net(next_state, goal).gather(dim=1, index=next_action_data)
                # convert the value of the terminal states to be zero
                terminal_mask = (torch.ones(done.size(), device=self.device) - done)
                max_next_state_goal_q_values = next_state_goal_q_values * terminal_mask
                # compute the TD target
                td_target = reward + self.gamma * max_next_state_goal_q_values

        # compute the loss using MSE error: TD error
        loss = self.criterion(state_goal_q_values, td_target)
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        # why clip the gradient
        if self.clip_gradient:
            for params in self.policy_net.parameters():
                params.grad.data.clamp(-1, 1)
        self.optimizer.step()

    # select an action based on the policy network
    def get_action(self, input_state, goal):
        input_state = input_state.float()
        goal = goal.float()
        input_state = input_state.to(self.device)
        goal = goal.to(self.device)
        with torch.no_grad():
            goal_q_values = self.policy_net(input_state, goal)
            # action = q_values.max(0)[1].item()
            action = goal_q_values.max(1)[1].item()
        return action

    def train_one_batch(self, t, batch):
        # update the policy network
        if not np.mod(t + 1, self.freq_update_policy):
            # sample a batch to train policy net
            self.update_policy_net(batch)

        # update the target network
        if np.mod(t, self.freq_update_target):
            self.update_target_net()

    def convert2tensor(self, batch):
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

    @staticmethod
    def rolling_average(data, window_size):
        assert data.ndim == 1
        kernel = np.ones(window_size)
        smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
        return smooth_data[: -window_size + 1]








