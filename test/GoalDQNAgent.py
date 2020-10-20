"""
    Goal-conditioned double DQN agent implementation notes:
        - The estimation of the terminal state from the policy network should be mask 0.
        - For double DQN, we have to detach both the target network and the policy network.
        - For hard update target network, we have to update it every fix number of steps. (defaul 2000)
        - For soft update target network, we have to update it every step.
"""
import time
import torch
from torch import nn
import numpy as np
import random
import IPython.terminal.debugger as Debug
import torch.nn.functional as F

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
    def __init__(self, small_obs=False, true_state=False, use_state_est=False):
        super(GoalDeepQNet, self).__init__()
        # set the small observation
        self.small_obs = small_obs
        # set the state to be true
        self.true_state = true_state
        # set the state estimation
        self.estimate_state = use_state_est
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
                self.fcNet = nn.Sequential(
                    nn.Linear(2048 * 2, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 4)
                )

                # if use the state estimation network
                if self.estimate_state:
                    self.fcEstNet = nn.Sequential(
                        nn.Linear(2048 * 2, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 2),
                        nn.LogSoftmax(dim=1)
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
                    nn.Linear(1024 * 2 * 4, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 4)
                )

                if self.estimate_state:
                    self.fcEstNet = nn.Sequential(
                        nn.Linear(1024 * 2 * 4, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 2),
                        nn.LogSoftmax(dim=1)
                    )
        else:
            self.fcNet = nn.Sequential(
                nn.Linear(6, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 4),
                nn.Identity()
            )

            if self.estimate_state:
                self.fcEstNet = nn.Sequential(
                    nn.Linear(6, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2),
                    nn.LogSoftmax(dim=1)
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
            val_x = self.fcNet(state_goal_fea)
            # if use state estimation, one more head
            if self.estimate_state:
                est_x = self.fcEstNet(state_goal_fea)
        else:
            # convert the dimension
            state = state.view(-1, 4)
            goal = goal.view(-1, 2) 
            #state = state.view(-1, 3)
            #goal = goal.view(-1, 3)
            state_goal_fea = torch.cat([state, goal], dim=1)
            val_x = self.fcNet(state_goal_fea)
            # if use state estimation, one more head
            if self.estimate_state:
                est_x = self.fcEstNet(state_goal_fea)
        
        if not self.estimate_state:
            return val_x
        else:
            return val_x, est_x


class GoalDQNAgent(object):
    def __init__(self,
                 dqn_mode="vanilla",
                 target_update_frequency=2000,
                 policy_update_frequency=4,
                 use_small_obs=False,
                 use_true_state=False,
                 use_target_soft_update=False,
                 use_gradient_clip=False,
                 use_rescale=False,
                 gamma=0.99,
                 learning_rate=1e-3,
                 device="cpu",
                 use_state_est=False,
                 alpha=1.0,
                 n_step_num=1
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
        self.policy_net = GoalDeepQNet(use_small_obs, use_true_state, use_state_est)
        self.target_net = GoalDeepQNet(use_small_obs, use_true_state, use_state_est)
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
        self.n_step_num = n_step_num
        self.alpha = alpha
        self.tau = 0.05  # parameters for soft target update
        self.use_true_state = use_true_state
        self.use_rescale = use_rescale
        self.soft_update = use_target_soft_update
        self.clip_gradient = use_gradient_clip
        self.freq_update_target = target_update_frequency
        self.freq_update_policy = policy_update_frequency
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate,
                                          weight_decay=5e-4)
        self.criterion = torch.nn.MSELoss()
        self.use_state_est = use_state_est
        if self.use_state_est:
            self.est_criterion = torch.nn.CrossEntropyLoss()
        """ Saving and plotting configurations"""
        self.returns = []
        self.lengths = []
        self.total_loss = []
        self.state_esti_loss = []
        
        self.current_total_loss = np.inf
        self.current_state_loss = np.inf

    # select an action based on the policy network
    def get_action(self, input_state, goal, eps):
        if random.uniform(0, 1) < eps:  # with probability epsilon, the agent selects a random action
            action = random.sample(range(4), 1)[0]
        else:  # with probability 1 - epsilon, the agent selects the greedy action
            input_state = self.toTensor(input_state)
            goal = self.toTensor(goal)
            with torch.no_grad():
                if not self.use_state_est:
                    goal_q_values = self.policy_net(input_state, goal).view(1, -1)
                else:
                    goal_q_values, _ = self.policy_net(input_state, goal)
                    goal_q_values = goal_q_values.view(1, -1)
                action = goal_q_values.max(dim=1)[1].item()
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
        state, action, next_state, reward, goal, done = self.convert2tensor(batch_data)
        
        # compute the Q_policy(s, a)
        if not self.use_state_est:
            goal_q_sa_values = self.policy_net(state, goal).gather(dim=1, index=action)
        else:
            goal_q_sa_values, _ = self.policy_net(state, goal)
            goal_q_sa_values = goal_q_sa_values.gather(dim=1, index=action) 
        # compute the TD target r + gamma * max_a' Q_target(s', a')
        if self.dqn_mode == "vanilla":  # update the policy network using vanilla DQN
            # compute the : max_a' Q_target(s', a')
            if not self.use_state_est:
                max_next_goal_q_sa_values = self.target_net(next_state, goal).max(1)[0].view(-1, 1).detach()
            else:
                max_next_goal_q_sa_values, state_goal_estimation = self.target_net(next_state, goal)
                max_next_goal_q_sa_values = max_next_goal_q_sa_values.max(1)[0].view(-1, 1).detach()
            # if s' is terminal, then change Q(s', a') = 0
            masked_max_next_goal_q_sa_values = max_next_goal_q_sa_values * (1 - done)
            # compute the r + gamma * max_a' Q_target(s', a')
            td_target = reward + self.gamma * masked_max_next_goal_q_sa_values
        else:  # update the policy network using double DQN
            # select the maximal actions using greedy policy network: argmax_a Q_policy(S_t+1)
            if not self.use_state_est:
                estimated_next_goal_action = self.policy_net(next_state, goal).max(dim=1)[1].view(-1, 1).detach()
            else:
                estimated_next_goal_action, state_goal_estimation = self.policy_net(next_state, goal)
                estimated_next_goal_action = estimated_next_goal_action.max(dim=1)[1].view(-1, 1).detach()
            # compute the Q_target(s', argmax_a)
            if not self.use_state_est:
                next_goal_q_sa_values = self.target_net(next_state, goal).gather(dim=1, index=estimated_next_goal_action).detach().view(-1, 1)
            else:
                next_goal_q_sa_values, _ = self.target_net(next_state, goal)
                next_goal_q_sa_values = next_goal_q_sa_values.gather(dim=1, index=estimated_next_goal_action).detach().view(-1, 1)
            # convert the value of the terminal states to be zero
            masked_next_goal_q_sa_values = next_goal_q_sa_values * (1 - done)
            # compute the TD target
            td_target = reward + self.gamma * masked_next_goal_q_sa_values

        if not self.use_state_est:
            # compute the loss using MSE error: TD error
            # loss = self.criterion(sa_goal_values, td_target)
            loss = F.smooth_l1_loss(goal_q_sa_values, td_target)
        else:
            # compute the first component using MSE Loss: TD error
            #loss_1 = self.criterion(sa_goal_values, td_target) 
            loss_1 = F.smooth_l1_loss(goal_q_sa_values, td_target)
            # compute the second component using Cross Entropy Loss: Estimation error 
            loss_label = torch.cat((torch.ones_like(done) - done, done), dim=1)
            loss_2 = (-1 * loss_label * state_goal_estimation).sum(-1).mean() 
            # combine together
            loss = loss_1 + self.alpha * loss_2

        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        # why clip the gradient
        if self.clip_gradient:
            for params in self.policy_net.parameters():
                params.grad.data.clamp(-1, 1)
        self.optimizer.step()
        
        if not self.use_state_est:
            return loss.item()
        else:
            return loss.item(), loss_2.item()

    def train_one_batch(self, t, batch):
        # update the policy network
        if not np.mod(t + 1, self.freq_update_policy):
            # sample a batch to train policy net
            if self.use_state_est:
                loss, loss_1 = self.update_policy_net(batch)
                self.total_loss.append(loss)
                self.state_esti_loss.append(loss_1)
                self.current_total_loss = loss
                self.current_state_loss = loss_1
            else:
                loss = self.update_policy_net(batch)
                self.current_total_loss = loss
                self.total_loss.append(loss)

        # update the target network
        if not self.soft_update:
            if not np.mod(t + 1, self.freq_update_target):
                self.update_target_net()
        else:
            self.update_target_net()

    def convert2tensor(self, batch):
        if not self.use_true_state:
            if len(batch._fields) == 5:
                state = torch.cat(batch.state, dim=0).float().to(self.device)
                action = torch.cat(batch.action, dim=0).long().to(self.device)
                reward = torch.cat(batch.reward, dim=0).float().to(self.device)
                next_state = torch.cat(batch.next_state, dim=0).float().to(self.device)
                done = torch.cat(batch.done, dim=0).to(self.device)
                if self.use_rescale:
                    return state/255, action, next_state/255, reward, done
                else:
                    return state, action, next_state, reward, done
            elif len(batch._fields) == 6 or len(batch._fields) == 7:
                state = torch.cat(batch.state, dim=0).float().to(self.device)
                action = torch.cat(batch.action, dim=0).long().to(self.device)
                reward = torch.cat(batch.reward, dim=0).float().to(self.device)
                next_state = torch.cat(batch.next_state, dim=0).float().to(self.device)
                goal = torch.cat(batch.goal, dim=0).float().to(self.device)
                done = torch.cat(batch.done, dim=0).to(self.device)
                if self.use_rescale:
                    return state/255, action, next_state/255, reward, goal/255, done
                else:
                    return state, action, next_state, reward, goal, done
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
            state_obs = torch.tensor(np.array(state).transpose(0, 3, 1, 2), dtype=torch.float32, device=self.device)
            if self.use_rescale:
                state_obs = state_obs / 255
        else:
            state_obs = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        return state_obs

    @staticmethod
    def rolling_average(data, window_size):
        assert data.ndim == 1
        kernel = np.ones(window_size)
        smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
        return smooth_data[: -window_size + 1]








