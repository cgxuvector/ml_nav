from utils import mapper
import numpy as np
from collections import namedtuple
import tqdm
from utils import memory
from utils import ml_schedule
import random
import torch
import os
import IPython.terminal.debugger as Debug

DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "reward", "next_state", "goal", "done"])


class Experiment(object):
    def __init__(self,
                 env,
                 maze_list,
                 seed_list,
                 agent,
                 max_time_steps,
                 max_time_steps_per_episode=100,
                 start_train_step=10,
                 buffer_size=None,
                 transition=DEFAULT_TRANSITION,
                 learning_rate=1e-3,
                 batch_size=64,
                 gamma=0.99,
                 save_dir=None,
                 model_name=None,
                 random_seed=1234,
                 use_replay=False,
                 sampled_goal=5,
                 eval_frequency=100,
                 eps_start=1.0,
                 eps_end=0.01):
        # environment
        self.env = env
        self.maze_list = maze_list
        self.seed_list = seed_list
        # agent
        self.agent = agent
        # training configurations
        self.max_time_steps = max_time_steps
        self.max_steps_per_episode = max_time_steps_per_episode
        self.start_train_step = start_train_step
        self.learning_rate = learning_rate
        self.eval_fre = eval_frequency
        # replay buffer configurations
        if buffer_size:
            self.replay_buffer = memory.ReplayMemory(buffer_size, transition)
            self.TRANSITION = transition
        self.batch_size = batch_size
        self.use_relay_buffer = use_replay
        # rl related configuration
        self.gamma = gamma
        self.schedule = ml_schedule.LinearSchedule(eps_start, eps_end, max_time_steps)
        # results statistics
        self.distance = []
        self.returns = []
        self.lengths = []
        # saving settings
        self.model_name = model_name
        self.save_dir = save_dir
        # random setting
        self.seed_rnd = random_seed
        self.sampled_goal = sampled_goal

    def run(self):
        # running statistics
        rewards = []
        episode_t = 0
        sampled_goal_count = self.sampled_goal
        # set a maze and obtain the start and goal positions
        np.random.seed(self.seed_rnd)
        size = np.random.choice(self.maze_list)
        seed = np.random.choice(self.seed_list)
        env_map = mapper.RoughMap(size, seed, 3)
        pos_params = env_map.get_start_goal_pos()
        # reset the environment
        state, goal = self.env.reset(size, seed, pos_params)
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)
            if np.random.sample() < eps:
                action = self.env.action_space.sample()
            else:
                # get action from the agent
                action = self.agent.get_action(self.toTensor(state), self.toTensor(goal))
            # apply the action
            next_state, reward, done, dist, _ = self.env.step(action)
            # store the replay buffer and convert the data to tensor
            if self.use_relay_buffer:
                trans = self.TRANSITION(state=self.toTensor(state),
                                        action=torch.tensor(action).long().view(-1, 1),
                                        reward=torch.tensor(reward).float().view(-1, 1),
                                        next_state=self.toTensor(next_state),
                                        goal=self.toTensor(goal),
                                        done=torch.tensor(done).view(-1, 1))
                self.replay_buffer.add(trans)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G
                self.returns.append(G)
                episode = len(self.returns)
                self.lengths.append(len(rewards))
                self.distance.append(dist)
                pbar.set_description(
                    f'Episode: {episode} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f}'
                )
                # reset
                rewards = []
                episode_t = 0
                if sampled_goal_count > 0:
                    size, seed, pos_params, env_map = self.map_sampling(env_map, self.maze_list, self.seed_list, True)
                    sampled_goal_count -= 1
                else:
                    size, seed, pos_params, env_map = self.map_sampling(env_map, self.maze_list, self.seed_list, False)
                    sampled_goal_count = self.sampled_goal
                state, goal = self.env.reset(size, seed, pos_params)
            else:
                state = next_state
                rewards.append(reward)
                episode_t += 1

            # train the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save the model and the statics
        model_save_path = os.path.join(self.save_dir, self.model_name) + ".pt"
        distance_save_path = os.path.join(self.save_dir, self.model_name + "_distance.npy")
        torch.save(self.agent.policy_net.state_dict(), model_save_path)
        np.save(distance_save_path, self.distance)


    @staticmethod
    def map_sampling(env_map, maze_list, seed_list, sample_pos=False):
        if not sample_pos:
            size = np.random.choice(maze_list)
            seed = np.random.choice(seed_list)
            env_map = mapper.RoughMap(size, seed, 3)
            # positions
            pos_params = [env_map.raw_pos['init'][0],
                          env_map.raw_pos['init'][1],
                          env_map.raw_pos['goal'][0],
                          env_map.raw_pos['goal'][1],
                          0]  # [init_pos, goal_pos, init_orientation]
        else:
            size = env_map.maze_size
            seed = env_map.maze_seed
            start_idx, end_idx = random.sample(range(len(env_map.valid_pos)), 2)
            start_pos = env_map.valid_pos[start_idx]
            end_pos = env_map.valid_pos[end_idx]
            # positions
            pos_params = [start_pos[0],
                          start_pos[1],
                          end_pos[0],
                          end_pos[1],
                          0]  # [init_pos, goal_pos, init_orientation]

        return size, seed, pos_params, env_map

    @staticmethod
    def toTensor(obs_list):
        state_obs = torch.tensor(np.array(obs_list).transpose(0, 3, 1, 2)).float()
        # print(state_obs.size())
        return state_obs

