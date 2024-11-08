"""
    Goal-conditioned double DQN agent implementation notes:
        - The estimation of the terminal state from the policy network should be mask 0.
        - For double DQN, we have to detach both the target network and the policy network.
        - For hard update target network, we have to update it every fix number of steps. (default 2000)
        - For soft update target network, we have to update it every step.

    Note: We use the same Double DQN model both in our model and the baseline SoRB (not discounted returnhengguang
    )

    We add distributional DQN
"""
import torch
from torch import nn
import numpy as np
import random
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


class GoalDeepQNet(nn.Module):
    """
        Define the Q network:
            - No image / image feature input: fully-connected (3 layer implemented)
            - Image input: convolutional (not implemented)
    """
    def __init__(self, small_obs=False, true_state=False, distributional_rl=False, atoms=51):
        super(GoalDeepQNet, self).__init__()
        # set the small observation
        self.small_obs = small_obs
        # set the state to be true
        self.true_state = true_state
        # use distributional RL
        self.use_distributional = distributional_rl
        # num of atoms
        self.atoms = atoms
        # if the convolutional flag is enabled.
        if not self.true_state:
            if not self.small_obs:
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
                if not self.use_distributional:
                    self.fcNet = nn.Sequential(
                        nn.Linear(2048 * 2, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 4)
                    )
                else:  # for distributional DQN
                    self.fcNet = nn.Sequential(
                        nn.Linear(2048 * 2, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 4 * self.atoms)
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
                if not self.use_distributional:
                    self.fcNet = nn.Sequential(
                        nn.Linear(1024 * 2 * 4, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 4)
                    )
                else:  # for distributional DQN
                    self.fcNet = nn.Sequential(
                        nn.Linear(1024 * 2 * 4, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 4 * self.atoms)
                    )
        else:
            # for normal DQN
            if not self.use_distributional:
                self.fcNet = nn.Sequential(
                    nn.Linear(6, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 4),
                    nn.Identity()
                )
            else:  # for distributional DQN
                self.fcNet = nn.Sequential(
                    nn.Linear(6, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 4 * self.atoms),
                    nn.Identity()
                )

    def forward(self, state, goal):
        if not self.true_state:
            if not self.small_obs:
                # compute state embedding
                state_fea = self.conv_qNet(state).contiguous().view(-1, 1 * 8 * 256)
                goal_fea = self.conv_qNet(goal).contiguous().view(-1, 1 * 8 * 256)
            else:
                # compute state embedding
                state_fea = self.conv_qNet(state).contiguous().view(-1, 1 * 8 * 128 * 4)
                goal_fea = self.conv_qNet(goal).contiguous().view(-1, 1 * 8 * 128 * 4)
            
            # concatenate the tensor
            state_goal_fea = torch.cat((state_fea, goal_fea), dim=1)
            # concatenate the goal with state
            x = self.fcNet(state_goal_fea)
            # compute the SoftMax for distributional RL
            if self.use_distributional:
                x = F.softmax(x.view(-1, self.atoms), dim=1).view(-1, 4, self.atoms)
        else:
            # convert the dimension
            state = state.view(-1, 3)
            goal = goal.view(-1, 3)
            state_goal_fea = torch.cat([state, goal], dim=1)
            x = self.fcNet(state_goal_fea)
            # compute the SoftMax for distributional RL
            if self.use_distributional:
                x = F.softmax(x.view(-1, self.atoms), dim=1).view(-1, 4, self.atoms)
        return x


class GoalDQNAgent(object):
    def __init__(self,
                 dqn_mode="vanilla",
                 target_update_frequency=2000,
                 policy_update_frequency=4,
                 use_small_obs=False,
                 use_true_state=False,
                 use_target_soft_update=False,
                 use_gradient_clip=False,
                 gamma=0.99,
                 learning_rate=1e-3,
                 device="cpu",
                 use_distributional=False,
                 support_atoms=2,
                 batch_size=8,
                 use_rescale=False
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
        # pixel values
        self.use_rescale = use_rescale
        # parameters for distributional DQN
        self.use_distributional = use_distributional
        # number of atoms
        self.support_atoms = support_atoms
        # min and max possible values
        self.min_val, self.max_val = -1 * (self.support_atoms - 1), 0
        # generate the possible values
        self.support_atoms_values = torch.linspace(self.min_val, self.max_val, self.support_atoms, device=self.device).view(-1, 1)
        self.delta_z = (self.max_val - self.min_val) / self.support_atoms
        self.batch_size = batch_size
        # create the policy network and target network
        self.policy_net = GoalDeepQNet(use_small_obs, use_true_state, use_distributional, support_atoms)
        self.target_net = GoalDeepQNet(use_small_obs, use_true_state, use_distributional, support_atoms)
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
        if not self.use_distributional:  # use MSE loss when using normal DQN
            self.criterion = torch.nn.MSELoss()
        else:  # use cross entropy loss when using distributional DQN
            self.criterion = torch.nn.CrossEntropyLoss()
        """ Saving and plotting configurations"""
        self.returns = []
        self.lengths = []

    # select an action based on the policy network
    def get_action(self, input_state, goal, eps):
        if random.uniform(0, 1) < eps:  # with probability epsilon, the agent selects a random action
            max_action = random.sample(range(4), 1)[0]
        else:  # with probability 1 - epsilon, the agent selects the greedy action
            input_state = self.toTensor(input_state)
            goal = self.toTensor(goal) 
            with torch.no_grad():
                if not self.use_distributional:  # for normal DQN
                    goal_q_values = self.policy_net(input_state, goal).view(1, -1)
                    max_action = goal_q_values.max(dim=1)[1].item()
                else:  # for distributional DQN
                    goal_q_distributions = self.policy_net(input_state, goal)
                    goal_q_distributions = goal_q_distributions.squeeze(0)
                    goal_q_values = torch.mm(goal_q_distributions, self.support_atoms_values)
                    max_action = goal_q_values.max(dim=0)[1].item()
        return max_action

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
        if not self.use_distributional:
            # convert the batch from numpy to tensor
            state, action, next_state, reward, goal, done = self.convert2tensor(batch_data)
            # compute the Q_policy(s, a)
            q_sa_values = self.policy_net(state, goal).gather(dim=1, index=action)
            # compute the TD target r + gamma * max_a' Q_target(s', a')
            if self.dqn_mode == "vanilla":  # update the policy network using vanilla DQN
                # compute the : max_a' Q_target(s', a')
                max_next_q_sa_values = self.target_net(next_state, goal).max(1)[0].view(-1, 1).detach()
                # if s' is terminal, then change Q(s', a') = 0
                terminal_mask = (torch.ones(done.size(), device=self.device) - done)
                max_next_q_sa_values = max_next_q_sa_values * terminal_mask
                # compute the r + gamma * max_a' Q_target(s', a')
                td_target = reward + self.gamma * max_next_q_sa_values
            else:  # update the policy network using double DQN
                # select the maximal actions using greedy policy network: argmax_a Q_policy(S_t+1)
                estimated_next_action = self.policy_net(next_state, goal).max(dim=1)[1].view(-1, 1).detach()
                # compute the Q_target(s', argmax_a)
                next_q_sa_values = self.target_net(next_state, goal).gather(dim=1, index=estimated_next_action).detach().view(-1, 1)
                # convert the value of the terminal states to be zero
                terminal_mask = (torch.ones(done.size(), device=self.device) - done)
                max_next_q_sa_values = (next_q_sa_values * terminal_mask).clamp(max=0)
                # compute the TD target
                td_target = reward + self.gamma * max_next_q_sa_values

            # compute the loss using MSE error: TD error
            loss = self.criterion(q_sa_values, td_target)
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            # why clip the gradient
            if self.clip_gradient:
                for params in self.policy_net.parameters():
                    params.grad.data.clamp(-1, 1)
            self.optimizer.step()
        else:
            # sampled transitions
            state, action, next_state, reward, goal, done = self.convert2tensor(batch_data)
            # compute the current distributional
            action = action.unsqueeze(-1).expand(self.batch_size, 1, self.support_atoms)
            current_q_distributions = self.policy_net(state, goal).gather(dim=1, index=action)
            # compute the target distribution
            with torch.no_grad():
                target_q_distributions = self.project_distributions(next_state, reward, done, goal, self.support_atoms_values, 'shift')
            
            # code the cross entropy loss
            loss = (-1 * target_q_distributions.squeeze(dim=1) * current_q_distributions.squeeze(dim=1).log()).sum(-1).mean()

            # update
            self.optimizer.zero_grad()
            loss.backward()
            # why clip the gradient
            if self.clip_gradient:
                for params in self.policy_net.parameters():
                    params.grad.data.clamp(-1, 1)
            self.optimizer.step()

    def project_distributions(self, next_state, reward, done, goal, support, project_mode='raw'):
        # compute the next distributions
        next_q_distributions = self.target_net(next_state, goal).detach()
        # compute the expected values
        next_expected_q_values = torch.mm(next_q_distributions.view(-1, self.support_atoms), support).view(-1, 4)
        # extract the maximal action
        max_next_action = next_expected_q_values.max(dim=1)[1].unsqueeze(-1).unsqueeze(-1).expand(self.batch_size, 1, self.support_atoms)
        # extract maximal distributions
        max_next_q_distribution = next_q_distributions.gather(dim=1, index=max_next_action).squeeze(dim=1)

        if project_mode == 'raw':
            # expand reward, done, and support
            reward = reward.expand_as(max_next_q_distribution)
            done = done.expand_as(max_next_q_distribution)
            support = support.squeeze(-1).unsqueeze(0).expand_as(max_next_q_distribution)

            # compute the target
            Tz = reward + (1 - done) * self.gamma * support
            Tz = Tz.clamp(min=self.min_val, max=self.max_val)
            b = (Tz - self.min_val) / self.delta_z
            l = b.floor().clamp(min=0, max=self.support_atoms - 1).long()
            u = b.ceil().clamp(min=0, max=self.support_atoms - 1).long()

            # compute the same indices
            row_indices, col_indices = torch.where(l == u)
            u_new = u.clone()
            for r, c in zip(row_indices.tolist(), col_indices.tolist()):
                u_new[r, c] += 1

            # distribute the probability
            offset = torch.linspace(0, (self.batch_size - 1) * self.support_atoms, self.batch_size).long()\
                .unsqueeze(1).expand(self.batch_size, self.support_atoms)
            # initialize the projected distributions
            project_distribution = torch.zeros(max_next_q_distribution.size())
            project_distribution.view(-1).index_add_(0, (l + offset).view(-1), (max_next_q_distribution * (u_new.float() - b)).view(-1))
            project_distribution.view(-1).index_add_(0, (u + offset).view(-1), (max_next_q_distribution * (b - l.float())).view(-1))
        elif project_mode == 'shift':
            # get the batch size
            batch_size = self.batch_size
            support_num = support.size(0)
            # compute the terminal mask tensor
            one_hot_mask = F.one_hot(torch.ones(batch_size, dtype=torch.int64, device=self.device) * (support_num - 1), support_num).float()
            # generate the first column
            col_1 = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)
            # generate the middle part
            col_middle = max_next_q_distribution[:, 2:]
            # sum the end part
            col_last = torch.sum(max_next_q_distribution[:, 0:2], dim=1).view(-1, 1)
            # combine the shifted q distribution
            shifted_q_distribution_target = torch.cat([col_last, col_middle, col_1], dim=1)
            # mask the target q distribution
            mask_condition = done.bool().expand(batch_size, support_num)
            masked_shifted_q_distribution_target = torch.where(mask_condition, one_hot_mask, shifted_q_distribution_target)
            project_distribution = masked_shifted_q_distribution_target
        else:
            raise Exception(f"Unexpected projection mode. Expected raw or shift, but get {project_mode} ")

        return project_distribution.unsqueeze(dim=1)

    def train_one_batch(self, t, batch):
        # update the policy networkse
        if not np.mod(t + 1, self.freq_update_policy):
            # sample a batch to train policy net
            self.update_policy_net(batch)
            if self.soft_update:
                self.update_target_net()

        # update the target network
        if not self.soft_update:
            if not np.mod(t + 1, self.freq_update_target):
                self.update_target_net()

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
                if not self.use_rescale:
                    return state, action, next_state, reward, goal, done
                else:
                    return state / 255.0, action, next_state / 255.0, reward, goal / 255.0, done
        else:
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

    def toTensor(self, state):
        if not self.use_true_state:
            if not self.use_rescale:
                state_obs = torch.tensor(np.array(state).transpose(0, 3, 1, 2), dtype=torch.float32, device=self.device)
            else:
                state_obs = torch.tensor(np.array(state).transpose(0, 3, 1, 2) / 255.0, dtype=torch.float32, device=self.device)
        else:
            state_obs = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        return state_obs

    @staticmethod
    def rolling_average(data, window_size):
        assert data.ndim == 1
        kernel = np.ones(window_size)
        smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
        return smooth_data[: -window_size + 1]
