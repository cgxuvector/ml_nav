"""
    Implementation of the TD3 method for discrete action space
"""
from collections import namedtuple
from collections import deque

import torch
from torch import nn

import numpy as np
import copy
import tqdm

from utils import ml_schedule
DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "next_state", "reward", "done"])


def env_test(env):
    max_time_steps = 10_000
    env.reset()
    for t in range(max_time_steps):
        env.render()
        action = env.action_space.sample()
        _, _, _, done = env.step(action)
        if done:
            env.reset()


""" Experience Replay Memory """


class ReplayMemory(object):
    def __init__(self, max_memory_size, transition=DEFAULT_TRANSITION):
        # memory params
        self.max_size = max_memory_size  # maximal size
        self.size = 0  # current size
        # transition type
        self.TRANSITION = transition  # transition type as a namedtuple. See DEFAULT_TRANSITION above
        # memory data buffer
        self.data_buffer = deque(maxlen=self.max_size)

    def __len__(self):
        # get the current memory size
        return self.size

    def add(self, single_transition):
        # check the transition
        assert (single_transition._fields == self.TRANSITION._fields), f"Invalid transition. Expect transition with " \
                                                                       f"fields {self.TRANSITION._fields}, but get" \
                                                                       f"fields {single_transition._fields}"
        # add the transition
        self.data_buffer.append(single_transition)
        # update the current memory size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # check the input
        assert batch_size > 0, f"Invalid batch size. Expect batch_size > 0, but get {batch_size}."
        # sampled indices for the data
        sampled_indices = np.random.choice(self.size, min(self.size, batch_size))
        # load the sampled transitions based on indices
        sampled_transitions_list = [self.data_buffer[idx] for idx in sampled_indices]
        # convert the list of transitions to transition of list
        sampled_transitions = self.TRANSITION(*zip(*sampled_transitions_list))
        # return the sampled transitions
        return sampled_transitions

    def pre_populate(self, env, transition_num):
        # check the input
        assert transition_num >= 0, f"Invalid number of prepopulated transitions. Expect transition_num >= 0" \
                                    f"but get {transition_num}"
        # reset the environment
        state = env.reset()
        for t in range(transition_num):
            # randomly sample one action
            action = env.action_space.sample()
            # step 1 in the environment
            next_state, reward, done, _ = env.step(action)
            # add the transition
            self.data_buffer.append(self.TRANSITION(state, action, next_state, reward, done))
            # check the termination and update the state
            if done:
                state = env.reset()
            else:
                state = next_state


""" Actor-Critic Models """


class Actor(nn.Module):
    """The Actor model"""
    def __init__(self, state_dim, action_num, hidden_dim=256):
        super(Actor, self).__init__()

        # Actor params
        self.input_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = action_num

        # Policy network
        self.policyNet = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, state) -> torch.Tensor:
        return self.policyNet(state)


class Critic(nn.Module):
    """The Critic model"""
    def __init__(self, state_dim, action_num, hidden_dim=256):
        super(Critic, self).__init__()

        # Critic params
        self.input_dim = state_dim + 1
        self.hidden_dim = hidden_dim
        self.output_dim = 1

        # Critic network
        self.criticNet = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),

            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, state, action) -> torch.Tensor:
        state_action = torch.cat((state, action.float()), dim=1)
        return self.criticNet(state_action)


""" TD3 Alg """


class TD3(object):
    def __init__(self,
                 env,
                 state_dim,
                 hidden_dim,
                 action_num,
                 start_time_steps,
                 max_time_steps,
                 buffer_size,
                 population_num,
                 batch_size,
                 actor_update_frequency,
                 critic_update_frequency,
                 learning_rate=1e-3,
                 transition_config=DEFAULT_TRANSITION,
                 gamma=0.99,
                 eps_schedule="linear",
                 eps_start=1.0,
                 eps_end=0.01,
                 tau=0.005,
                 save_name=""):
        """
        """
        self.env = env
        """ Actor-Critic framework"""
        # actor networks
        self.actor = Actor(state_dim, action_num, hidden_dim)
        self.actor_target = copy.deepcopy(self.actor)
        # critic networks
        self.critic_1 = Critic(state_dim, action_num, hidden_dim)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2 = Critic(state_dim, action_num, hidden_dim)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        """ Replay memory """
        self.TRANSITION = transition_config
        self.batch_size = batch_size
        self.pre_pop_num = population_num
        self.replay_buffer = ReplayMemory(buffer_size)
        """ Epsilon schedule """
        self.schedule = ml_schedule.LinearSchedule(eps_start, eps_end, max_time_steps) if eps_schedule == "linear" else ml_schedule.ConstantSchedule(eps_start)
        """ Training params"""
        self.tau = tau
        self.discount = gamma
        self.start_time_steps = start_time_steps
        self.max_time_steps = max_time_steps
        self.freq_update_actor = actor_update_frequency
        self.freq_update_critic = critic_update_frequency
        """ Optimizers """
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=learning_rate,
                                                weight_decay=5e-4)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=learning_rate,
                                                   weight_decay=5e-4)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=learning_rate,
                                                   weight_decay=5e-4)
        """ Loss functions"""
        self.criterion = torch.nn.MSELoss()

        """ Training statistics """
        self.returns = []
        self.lengths = []
        self.save_name = save_name

    # select an action based on epsilon greedy
    def select_action(self, state, eps):
        if np.random.sample() < eps:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                action_prob = self.actor(torch.tensor(state).float().view(1, -1))
                action = action_prob.max(1)[1].item()
        return action

    def update_actor_critic(self, batch_data, t_total):
        # convert batch data to tensor
        state, action, next_state, reward, done = self.convert2tensor(batch_data)
        # compute the current Q estimations
        state_q1_values = self.critic_1(state, action)
        state_q2_values = self.critic_2(state, action)
        # compute the two target values
        with torch.no_grad():
            # select an action for the next_state
            next_action = self.actor_target(next_state).max(dim=1)[1].view(-1, 1)
            # compute the critic targets
            next_state_q1_values = self.critic_1_target(next_state, next_action)
            next_state_q2_values = self.critic_2_target(next_state, next_action)
            min_next_state_q_values = torch.min(next_state_q1_values, next_state_q2_values)
            # compute the terminal mask
            terminal_mask = (torch.ones(done.size()) - done).view(-1, 1)
            td_target_q_values = reward.view(-1, 1) + self.discount * min_next_state_q_values * terminal_mask

        # compute the loss
        critic_1_loss = self.criterion(state_q1_values, td_target_q_values)
        critic_2_loss = self.criterion(state_q2_values, td_target_q_values)
        # optimization
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        if np.mod(t_total, self.freq_update_actor):
            self.update_actor(state)
            self.update_target()

    def update_actor(self, state):
        # compute the actor loss
        action = self.actor(state).max(1)[1].view(-1, 1)
        actor_loss = -1 * self.critic_1(state, action).mean()
        # optimization
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # update target
        self.update_target()

    def update_target(self):
        # self.actor_target.load_state_dict(self.actor.state_dict())
        # self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        # self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self):
        rewards = []
        episode_counter = 0
        episode_t_counter = 0
        pbar = tqdm.trange(self.max_time_steps)
        state = self.env.reset()
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)
            # obtain an action
            action = self.select_action(state, eps)
            # take 1 step in the environment
            next_state, reward, done, _ = self.env.step(action)
            # add sample to replay buffer
            self.replay_buffer.add(self.TRANSITION(state, action, next_state, reward, done))

            # start training when the buffer has enough transitions
            if t < self.start_time_steps:
                # update the states
                if done:
                    state = self.env.reset()
                else:
                    state = next_state
                continue

            # update the actor_critic
            if not np.mod(t, self.freq_update_critic):
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.update_actor_critic(sampled_batch, t)

            if done:
                # compute the returns
                G = 0
                for r in reversed(rewards):
                    G = r + self.discount * G
                # store the length and returns
                self.returns.append(G)
                self.lengths.append(episode_t_counter)

                # increase episode
                episode_counter += 1

                # show the information in the bar
                pbar.set_description(
                    f'Episode: {episode_counter} | Steps: {episode_t_counter} | Return: {G} | Epsilon: {eps}'
                )

                # reset the environment
                state = self.env.reset()
                episode_t_counter = 0
                rewards = []
            else:
                state = next_state
                episode_t_counter += 1
                rewards.append(reward)

        np.save("./results/return/" + self.save_name + "_return.npy", np.array(self.returns))
        np.save("./results/return/" + self.save_name + "_lengths.npy", np.array(self.lengths))
        torch.save(self.actor.state_dict(), "./results/model/" + self.save_name + ".pt")

    @staticmethod
    def convert2tensor(batch):
        state = torch.tensor(batch.state).float()
        action = torch.tensor(batch.action).long().view(-1, 1)
        next_state = torch.tensor(batch.next_state).float()
        reward = torch.tensor(batch.reward).float()
        done = torch.tensor(batch.done).float()
        return state, action, next_state, reward, done
