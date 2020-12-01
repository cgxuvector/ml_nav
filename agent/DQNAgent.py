import torch
from torch import nn
import numpy as np
import gym
import tqdm
import abc
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


# define the abstract base class
class Schedule(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_value(self, time):
        pass


# define the linear schedule
class LinearSchedule(Schedule):
    """ This schedule returns the value linearly"""
    def __init__(self, start_value, end_value, duration):
        self._start_value = start_value
        self._end_value = end_value
        self._duration = duration
        self._schedule_amount = end_value - start_value

    def get_value(self, time):
        return self._start_value + self._schedule_amount * min(1.0, time * 1.0 / self._duration)


# class of deep neural network model
class DeepQNet(nn.Module):
    # initialization
    def __init__(self, obs_dim, act_dim):
        # inherit from nn module
        super(DeepQNet, self).__init__()
        # feed forward network
        self.fc_layer = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Identity()
        )

    # forward function
    def forward(self, state):
        x = self.fc_layer(state)
        return x


class DQNAgent(object):
    # initialize the agent
    def __init__(self,
                 env_params=None,
                 agent_params=None,
                 ):
        # save the parameters
        self.env_params = env_params
        self.agent_params = agent_params

        # environment parameters
        self.action_space = np.linspace(0, env_params['act_num'], env_params['act_num'], endpoint=False).astype('uint8')
        self.action_dim = env_params['act_num']
        self.obs_dim = env_params['obs_dim']

        # create behavior policy and target networks
        self.dqn_mode = agent_params['dqn_mode']
        self.use_obs = agent_params['use_obs']
        self.behavior_policy_net = DeepQNet(self.obs_dim, self.action_dim)
        self.target_policy_net = DeepQNet(self.obs_dim, self.action_dim)

        # initialize target network with behavior network
        self.behavior_policy_net.apply(customized_weights_init)
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

        # send the agent to a specific device: cpu or gpu
        self.device = torch.device(agent_params['device'])
        self.behavior_policy_net.to(self.device)
        self.target_policy_net.to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.behavior_policy_net.parameters(), lr=self.agent_params['lr'])

    # get action
    def get_action(self, obs, eps):
        if np.random.random() < eps:  # with probability eps, the agent selects a random action
            action = np.random.choice(self.action_space, 1)[0]
        else:  # with probability 1 - eps, the agent selects a greedy policy
            obs = self._arr_to_tensor(obs).view(1, -1)
            with torch.no_grad():
                q_values = self.behavior_policy_net(obs)
                action = q_values.max(dim=1)[1].item()
        return action

    # update behavior policy
    def update_behavior_policy(self, batch_data):
        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']

        # compute the q value estimation using the behavior network
        pred_q_value = self.behavior_policy_net(obs_tensor)
        pred_q_value = pred_q_value.gather(dim=1, index=actions_tensor)

        # compute the TD target using the target network
        if self.dqn_mode == 'vanilla':
            # compute the TD target using vanilla method: TD = r + gamma * max a' Q(s', a')
            # no gradient should be tracked
            with torch.no_grad():
                max_next_q_value = self.target_policy_net(next_obs_tensor).max(dim=1)[0].view(-1, 1)
                td_target_value = rewards_tensor + self.agent_params['gamma'] * (1 - dones_tensor) * max_next_q_value
        else:
            # compute the TD target using double method: TD = r + gamma * Q(s', argmaxQ_b(s'))
            with torch.no_grad():
                max_next_actions = self.behavior_policy_net(next_obs_tensor).max(dim=1)[1].view(-1, 1).long()
                max_next_q_value = self.target_policy_net(next_obs_tensor).gather(dim=1, index=max_next_actions).view(-1, 1)
                td_target_value = rewards_tensor + self.agent_params['gamma'] * (1 - dones_tensor) * max_next_q_value
                td_target_value = td_target_value.detach()

        # compute the loss
        td_loss = torch.nn.functional.mse_loss(pred_q_value, td_target_value)

        # minimize the loss
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    # update update target policy
    def update_target_policy(self):
        if self.agent_params['use_soft_update']:  # update the target network using polyak average (i.e., soft update)
            # polyak ~ 0.95
            for param, target_param in zip(self.behavior_policy_net.parameters(), self.target_policy_net.parameters()):
                target_param.data.copy_((1 - self.agent_params['polyak']) * param + self.agent_params['polyak'] * target_param)
        else:  # hard update
            self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

    # load trained model
    def load_model(self, model_file):
        # load the trained model
        self.behavior_policy_net.load_state_dict(torch.load(model_file, map_location=self.device))
        self.behavior_policy_net.eval()

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        # get the numpy arrays
        obs_arr, action_arr, reward_arr, next_obs_arr, done_arr = batch_data
        # convert to tensors
        batch_data_tensor['obs'] = torch.tensor(obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['action'] = torch.tensor(action_arr, dtype=torch.int).long().view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        batch_data_tensor['next_obs'] = torch.tensor(next_obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)

        return batch_data_tensor


# re-implement the replay buffer based on openai baseline
class ReplayBuffer(object):
    def __init__(self, buffer_size):
        # total size of the replay buffer
        self.total_size = buffer_size

        # create a list to store the transitions
        self._data_buffer = []
        self._next_idx = 0

    def __len__(self):
        return len(self._data_buffer)

    def add(self, obs, act, reward, next_obs, done):
        # create a tuple
        trans = (obs, act, reward, next_obs, done)

        # interesting implementation
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trans)
        else:
            self._data_buffer[self._next_idx] = trans

        # increase the index
        self._next_idx = (self._next_idx + 1) % self.total_size

    def _encode_sample(self, indices):
        # lists for transitions
        obs_list, actions_list, rewards_list, next_obs_list, dones_list = [], [], [], [], []

        # collect the data
        for idx in indices:
            # get the single transition
            data = self._data_buffer[idx]
            obs, act, reward, next_obs, d = data
            # store to the list
            obs_list.append(np.array(obs, copy=False))
            actions_list.append(np.array(act, copy=False))
            rewards_list.append(np.array(reward, copy=False))
            next_obs_list.append(np.array(next_obs, copy=False))
            dones_list.append(np.array(d, copy=False))
        # return the sampled batch data as numpy arrays
        return np.array(obs_list), np.array(actions_list), np.array(rewards_list), np.array(next_obs_list), np.array(
            dones_list)

    def sample_batch(self, batch_size):
        # sample indices with replaced
        indices = [np.random.randint(0, len(self._data_buffer)) for _ in range(batch_size)]
        return self._encode_sample(indices)


class Experiment(object):
    def __init__(self, env_params, agent_params, trn_params):
        # initialize the parameters
        self.env_params = env_params
        self.agent_params = agent_params
        self.trn_params = trn_params

        # initialize the environment
        self.env_trn = self.env_params['env_trn']
        self.env_tst = self.env_params['env_tst']

        # initialize the agent
        self.agent = self.agent_params['agent']

        # initialize the memory
        self.memory = ReplayBuffer(self.trn_params['total_time_steps'])

        # initialize the schedule
        self.schedule = LinearSchedule(1, 0.01, self.trn_params['total_time_steps'] / 2)

        # training statistics
        self.returns = []

    def train_agent(self):
        # reset the environment
        obs, rewards = self.env_trn.reset(), []

        episode_t = 0
        pbar = tqdm.trange(self.trn_params['total_time_steps'])
        for t in pbar:
            # get one action
            eps = self.schedule.get_value(t)
            action = self.agent.get_action(obs, eps)

            # interact with the environment
            next_obs, reward, done, _ = self.env_trn.step(action)

            # add to memory
            self.memory.add(obs, action, reward, next_obs, done)

            # store the data
            rewards.append(reward)
            episode_t += 1
            obs = next_obs

            # check termination
            if done or episode_t % self.trn_params['episode_time_steps'] == 0:
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G += self.agent_params['gamma'] * r
                self.returns.append(G)
                episode_idx = len(self.returns)

                # print information
                pbar.set_description(
                    f'Episode: {episode_idx} | '
                    f'Steps: {episode_t} |'
                    f'Return: {G:2f} | '
                    f'Eps: {eps} | '
                    f'Buffer: {len(self.memory)}'
                )

                # reset the environment
                obs, rewards, episode_t = self.env_trn.reset(), [], 0

            # train the agent
            if t > self.trn_params['start_train_step']:
                # sample a mini-batch
                batch_data = self.memory.sample_batch(self.trn_params['batch_size'])
                # update the behavior policy
                if not np.mod(t, self.trn_params['update_policy_freq']):
                    self.agent.update_behavior_policy(batch_data)
                # update the target policy
                if not np.mod(t, self.trn_params['update_target_freq']):
                    self.agent.update_target_policy()


# init the environment
env_trn = gym.make('CartPole-v1')
env_tst = gym.make('CartPole-v1')

# initialize parameters for environment
env_params = {
    'env_trn': env_trn,
    'env_tst': env_tst,
    'act_num': env_trn.action_space.n,
    'obs_dim': env_trn.observation_space.shape[0]
}

# initialize parameters for agent
agent_params = {
    'agent': None,
    'dqn_mode': 'vanilla',
    'use_obs': False,
    'polyak': 0.95,
    'device': 'cpu',
    'lr': 1e-3,
    'gamma': 0.995,
    'use_soft_update': False
}

# initialize parameters for training
train_params = {
    'memory_size': 50000,
    'batch_size': 128,
    'total_time_steps': 500000,
    'episode_time_steps': 100,
    'start_train_step': 1000,
    'update_target_freq': 1000,
    'update_policy_freq': 4,
    'lr': 1e-3
}

# create the agent
my_agent = DQNAgent(env_params, agent_params)
agent_params['agent'] = my_agent

# create the experiment
my_experiment = Experiment(env_params, agent_params, train_params)

# run the experiment
my_experiment.train_agent()