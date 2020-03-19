""" This is the traditional critic method!
    Implementation of Double DQN method that includes the followings:
        - Definition of the action-value function: deep neural network architecture
        - Definition of the memory: replay buffer
        - Update functions
"""
import torch
from torch import nn
import numpy as np
from collections import deque
from collections import namedtuple
import tqdm
from utils import ml_schedule
DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "next_state", "reward", "done"])


class ReplayMemory(object):
    """
        Define the experience replay buffer
        Note: currently, the replay store numpy arrays
    """

    def __init__(self, max_memory_size, transition=DEFAULT_TRANSITION):
        """
        Initialization function
        :param max_memory_size: maximal size of the replay memory
        :param transition: transition type defined as a named tuple. Default version is
                    "
                    Transition(state, action, next_state, reward, done)
                    "
        """
        # memory params
        self.max_size = max_memory_size
        self.size = 0

        # transition params
        self.TRANSITION = transition

        # memory data
        self.data_buffer = deque(maxlen=self.max_size)

    def __len__(self):
        """ Return current memory size. """
        return self.size

    def add(self, single_transition):
        """
        Add one transition to the replay memory
        :param trans: should be an instance of the namedtuple defined in self.TRANSITION
        :return: None
        """
        # check the fields compatibility
        assert (single_transition._fields == self.TRANSITION._fields), f"A valid transition should contain " \
                                                                       f"{self.TRANSITION._fields}" \
                                                                       f" but currently got {single_transition._fields}"
        # add the data into buffer
        self.data_buffer.append(single_transition)

        # track the current buffer size and index
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample one batch from the replay memory
        :param batch_size: size of the sampled batch
        :return: batch
        """
        assert batch_size > 0, f"Invalid batch size. Expect batch size > 0, but get {batch_size}"
        # sample the indices: if buffer is smaller than batch size,then return the whole buffer,
        # otherwise return the batch
        sampled_indices = np.random.choice(self.size, min(self.size, batch_size))
        # obtain the list of named tuples, each is a transition
        sampled_transition_list = [self.data_buffer[idx] for idx in sampled_indices]
        # convert the list of transitions to transition of list-like data
        # *sampled_list: unpack the list into elements
        # zip(*sampled_list): parallel iterate the sub-elements in each unpacked element
        # trans_type(*zip(*sample_list)): construct the batch
        sampled_transitions = self.TRANSITION(*zip(*sampled_transition_list))
        return sampled_transitions


class DeepQNet(nn.Module):
    """
        Define the Q network:
            - No image / image feature input: fully-connected (3 layer implemented)
            - Image input: convolutional (not implemented)
    """
    def __init__(self, state_dim, action_num, hidden_dim=512):
        super(DeepQNet, self).__init__()

        # Q network fc layers configs
        self.input_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = action_num

        # define the q network
        self.qNet = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, state):
        return self.qNet(state)


class DQN(object):
    def __init__(self,
                 env,
                 state_dim,  # Q network configs
                 action_num,
                 hidden_dim,
                 buffer_size,  # replay memory configs
                 batch_size,
                 max_time_steps,  # training params
                 start_update_step,
                 target_update_frequency,
                 policy_update_frequency,
                 soft_target_update=False,
                 transition_config=DEFAULT_TRANSITION,  # other params with default values
                 learning_rate=1e-3,
                 dqn_mode="vanilla",
                 eps_start=1.0,
                 eps_end=0.01,
                 eps_schedule="linear",
                 gamma=1.0,
                 gradient_clip=True,
                 save_name=""
                 ):
        """
        """
        # environment
        self.env = env
        """ DQN configurations"""
        # create the policy network and target network
        self.policy_net = DeepQNet(state_dim, action_num, hidden_dim)
        self.target_net = DeepQNet(state_dim, action_num, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # create the replay buffer
        self.TRANSITION = transition_config
        self.batch_size = batch_size
        self.replay_buffer = ReplayMemory(buffer_size, transition=transition_config)
        # DQN mode: vanilla or double
        self.dqn_mode = dqn_mode
        # Epsilon schedule
        self.schedule = ml_schedule.LinearSchedule(eps_start, eps_end, max_time_steps) if eps_schedule == "linear" else ml_schedule.ConstantSchedule(eps_start)

        """ Training configurations """
        self.gamma = gamma
        self.max_time_steps = max_time_steps
        self.learning_rate = learning_rate
        self.step_update_start = start_update_step
        self.tau = 0.01  # parameters for soft target update
        self.soft_update = soft_target_update  # flag for soft update
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
        self.save_name = save_name

    # update the target network
    def update_target_net(self):
        # hard update
        if not self.soft_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # update the policy network
    def update_policy_net(self, batch_data):
        # convert the batch from numpy to tensor
        state, action, next_state, reward, done = self.convert2tensor(batch_data)
        # compute the Q_policy(s, a)
        state_q_values = self.policy_net(state).gather(dim=1, index=action)
        # compute the TD target r + gamma * max_a' Q_target(s', a')
        if self.dqn_mode == "vanilla":  # update the policy network using vanilla DQN
            with torch.no_grad():
                # compute the : max_a' Q_target(s', a')
                max_next_state_q_values = self.target_net(next_state).max(1)[0]
                # if s' is terminal, then change Q(s', a') = 0
                terminal_mask = (torch.ones(done.size()) - done)
                max_next_state_q_values = max_next_state_q_values * terminal_mask
                # compute the r + gamma * max_a' Q_target(s', a')
                td_target = (reward + self.gamma * max_next_state_q_values).view(-1, 1)
        else:  # update the policy network using double DQN
            with torch.no_grad():
                # select the maximal actions using greedy policy network: argmax_a Q_policy(S_t+1)
                estimate_next_state_q_values = self.policy_net(next_state)
                next_action_data = estimate_next_state_q_values.max(1)[1].view(-1, 1)
                # compute the Q_target(s', argmax_a)
                next_state_q_values = self.target_net(next_state).gather(dim=1, index=next_action_data)
                # convert the value of the terminal states to be zero
                terminal_mask = (torch.ones(done.size()) - done).view(-1, 1)
                max_next_state_q_values = next_state_q_values * terminal_mask
                # compute the TD target
                td_target = reward.view(-1, 1) + self.gamma * max_next_state_q_values

        # compute the loss using MSE error: TD error
        loss = self.criterion(state_q_values, td_target)
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        # why clip the gradient
        if self.clip_gradient:
            for params in self.policy_net.parameters():
                params.grad.data.clamp(-1, 1)
        self.optimizer.step()

    # select an action based on epsilon greedy
    def select_action(self, input_state, eps):
        # select a random action with probability epsilon
        if np.random.sample() < eps:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.policy_net(input_state)
                action = q_values.max(0)[1].item()
        return action

    def train(self):
        # episode params
        episode_idx = 0
        episode_t = 0
        rewards = []
        # reset the environment
        state = self.env.reset()
        # loop all time steps
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # obtain the epsilon
            eps = self.schedule.get_value(t)
            # select an action using epsilon greedy strategy
            action = self.select_action(torch.tensor(state).float(), eps)
            # interact with the environment
            next_state, reward, done, _ = self.env.step(action)
            # add the sample into the replay buffer
            sample = self.TRANSITION(state=state,
                                     action=action,
                                     next_state=next_state,
                                     reward=reward,
                                     done=done)
            self.replay_buffer.add(sample)

            # collecting samples
            if t < self.step_update_start:
                if done:
                    state = self.env.reset()
                else:
                    state = next_state
                continue

            # update the policy network
            if not np.mod(t+1, self.freq_update_policy):
                # sample a batch to train policy net
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.update_policy_net(sampled_batch)

            # update the target network
            if np.mod(t, self.freq_update_target):
                self.update_target_net()

            # compute the training statistics
            if done:
                # compute the returns
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G
                self.returns.append(G)
                # compute the length
                self.lengths.append(episode_t)
                # track the number of episodes
                episode_idx += 1

                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {self.returns[-1]} | Epsilon: {eps}'
                )
                # reset
                rewards = []
                episode_t = 0
                state = self.env.reset()
            else:
                # record rewards
                rewards.append(reward)
                # increase time step for one episode
                episode_t += 1
                # next state
                state = next_state

        np.save("./results/return/" + self.save_name + "_return.npy", np.array(self.returns))
        np.save("./results/return/" + self.save_name + "_length.npy", np.array(self.lengths))
        torch.save(self.policy_net.state_dict(), "./results/model/" + self.save_name + ".pt")

    def eval(self, eval_mode, model_name):
        # evaluation offline: evaluate the policy after training
        if eval_mode == "offline":
            self.policy_net.load_state_dict(torch.load(model_name))
            self.policy_net.eval()

        episode_num = 1
        step_num = 100_000

        for ep in range(episode_num):
            state = self.env.reset()
            for t in range(step_num):
                self.env.render()
                act = self.select_action(torch.tensor(state).float(), 0)
                next_state, _, done, _ = self.env.step(act)
                if done:
                    break
                state = next_state

    @staticmethod
    def convert2tensor(batch):
        state = torch.tensor(batch.state).float()
        action = torch.tensor(batch.action).view(-1, 1).long()
        reward = torch.tensor(batch.reward).float()
        next_state = torch.tensor(batch.next_state).float()
        done = torch.tensor(batch.done).float()
        return state, action, next_state, reward, done

    @staticmethod
    def rolling_average(data, window_size):
        assert data.ndim == 1
        kernel = np.ones(window_size)
        smooth_data = np.convolve(data, kernel) / np.convolve(np.ones_like(data), kernel)
        return smooth_data[: -window_size + 1]








