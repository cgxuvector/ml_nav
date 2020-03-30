from utils import mapper
import numpy as np
from collections import namedtuple
import tqdm
from utils import memory
from utils import ml_schedule
import torch
import os
import sys
import IPython.terminal.debugger as Debug
import matplotlib.pyplot as plt
from scipy import ndimage

DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "next_state", "reward", "goal", "done"])


class Experiment(object):
    def __init__(self,
                 env,
                 maze_list,
                 seed_list,
                 agent,
                 max_time_steps,
                 fix_maze=True,
                 fix_start=True,
                 fix_goal=True,
                 train_episode_num=10,
                 episode_time_steps=2000,
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
                 sampled_goal=10,
                 eps_start=1.0,
                 eps_end=0.01,
                 device="cpu",
                 use_goal=False):
        # environment
        self.env = env
        self.maze_list = maze_list
        self.seed_list = seed_list
        self.fix_maze = fix_maze
        self.fix_start = fix_start
        self.fix_goal = fix_goal
        # agent
        self.agent = agent
        # training configurations
        self.max_time_steps = max_time_steps
        self.max_steps_per_episode = episode_time_steps
        self.start_train_step = start_train_step
        self.learning_rate = learning_rate
        self.device = torch.device(device)
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
        self.train_episode_num = train_episode_num
        # goal conditioned flag
        self.use_goal = use_goal

    def run(self):
        """
        Function to run the experiment given the environment, agent, and training details
        :return: None. Saved results are as follows:
            - list of final distances for every episode during the training.
            - final model for the agent
        """
        # running statistics
        rewards = []
        episode_t = 0
        sampled_goal_count = self.sampled_goal
        train_episode_count = self.train_episode_num
        # set the random seed
        np.random.seed(self.seed_rnd)
        # select the maze
        size = np.random.choice(self.maze_list)
        seed = np.random.choice(self.seed_list)
        # load the 2D map
        env_map = mapper.RoughMap(size, seed, 3)
        pos_params = env_map.get_start_goal_pos()
        # reset the environment
        state, goal = self.env.reset(size, seed, pos_params)
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)
            # get an action from epsilon greedy
            if np.random.sample() < eps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.get_action(self.toTensor(state)) if not self.use_goal else self.agent.get_action(self.toTensor(state), self.toTensor(goal))
            # step in the environment
            next_state, reward, done, dist, _ = self.env.step(action)
            # store the replay buffer and convert the data to tensor
            if self.use_relay_buffer:
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                self.replay_buffer.add(trans)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G
                # store the return, episode length, and final distance
                self.returns.append(G)
                self.lengths.append(episode_t)
                self.distance.append(dist)
                # compute the episode number
                episode_idx = len(self.returns)
                # print the information
                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | Init: {pos_params[0:2]} | Goal: {pos_params[2:4]}'
                )
                # reset the environments
                rewards = []  # rewards recorder
                episode_t = 0  # episode steps counter
                # for a fixed (start, goal) pair, train it for #train_episode_count
                if train_episode_count > 0:
                    train_episode_count -= 1
                else:
                    # for a fixed maze, sampled #(sampled_goal_count) (start, goal) pairs
                    if sampled_goal_count > 0:
                        size, seed, pos_params, env_map = self.map_sampling(env_map, self.maze_list, self.seed_list, True)
                        sampled_goal_count -= 1
                    else:
                        # then, change to another maze environment
                        size, seed, pos_params, env_map = self.map_sampling(env_map, self.maze_list, self.seed_list, self.fix_maze)
                        sampled_goal_count = self.sampled_goal
                    train_episode_count = self.train_episode_num
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
        returns_save_path = os.path.join(self.save_dir, self.model_name + "_return.npy")
        lengths_save_path = os.path.join(self.save_dir, self.model_name + "_length.npy")
        torch.save(self.agent.policy_net.state_dict(), model_save_path)
        np.save(distance_save_path, self.distance)
        np.save(returns_save_path, self.returns)
        np.save(lengths_save_path, self.lengths)

    def goal_conditioned_run(self):
        """
        Run experiments using goal-conditioned DQN with the nearby goals
        :return: None
        """
        # running statistics
        rewards = []
        episode_t = 0
        sampled_goal_count = self.sampled_goal
        train_episode_count = self.train_episode_num
        # set the random seed
        np.random.seed(self.seed_rnd)
        # select the maze
        size = np.random.choice(self.maze_list)
        seed = np.random.choice(self.seed_list)
        # load the 2D map
        env_map = mapper.RoughMap(size, seed, 3)
        pos_params = env_map.get_start_goal_pos()
        pos_params[2:4] = env_map.sample_path_goal(env_map.init_pos, 1)
        # reset the environment
        state, goal = self.env.reset(size, seed, pos_params)
        pbar = tqdm.trange(self.max_time_steps)
        fig, arrs = plt.subplots(1, 2)
        top_down_img = arrs[0].imshow(ndimage.rotate(self.env.top_down_obs, -90))
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)
            # get an action from epsilon greedy
            if np.random.sample() < eps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.get_action(self.toTensor(state)) if not self.use_goal else self.agent.get_action(
                    self.toTensor(state), self.toTensor(goal))
            # step in the environment
            next_state, reward, done, dist, _ = self.env.step(action)
            # store the replay buffer and convert the data to tensor
            if self.use_relay_buffer:
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                self.replay_buffer.add(trans)

            top_down_img.set_data(ndimage.rotate(self.env.top_down_obs, -90))
            fig.canvas.draw()
            plt.pause(0.0001)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G
                # store the return, episode length, and final distance
                self.returns.append(G)
                self.lengths.append(episode_t)
                self.distance.append(dist)
                # compute the episode number
                episode_idx = len(self.returns)
                # print the information
                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | Init: {pos_params[0:2]} | Goal: {pos_params[2:4]}'
                )
                # reset the environments
                rewards = []  # rewards recorder
                episode_t = 0  # episode steps counter
                # for a fixed (start, goal) pair, train it for #train_episode_count
                if train_episode_count > 0:
                    train_episode_count -= 1
                else:
                    # for a fixed maze, sampled #(sampled_goal_count) (start, goal) pairs
                    if sampled_goal_count > 0:
                        # size, seed, pos_params, env_map = self.map_sampling(env_map, self.maze_list, self.seed_list,
                        #                                                     True)
                        size = env_map.maze_size
                        seed = env_map.maze_seed
                        pos_params[2:4] = env_map.sample_path_goal(env_map.goal_pos, 1)
                        sampled_goal_count -= 1
                    else:
                        # then, change to another maze environment
                        size, seed, pos_params, env_map = self.map_sampling(env_map, self.maze_list, self.seed_list,
                                                                            self.fix_maze)
                        sampled_goal_count = self.sampled_goal
                    train_episode_count = self.train_episode_num
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
        returns_save_path = os.path.join(self.save_dir, self.model_name + "_return.npy")
        lengths_save_path = os.path.join(self.save_dir, self.model_name + "_length.npy")
        torch.save(self.agent.policy_net.state_dict(), model_save_path)
        np.save(distance_save_path, self.distance)
        np.save(returns_save_path, self.returns)
        np.save(lengths_save_path, self.lengths)

    def map_sampling(self, env_map, maze_list, seed_list, sample_pos=False):
        """
        Function is used to sampled start and goal positions in one maze or to sample a new maze.
        :param env_map: The map object of the current maze. This variable is used to sample start and goal positions.
        :param maze_list: List of the valid maze sizes.
        :param seed_list: List of the valid maze seeds.
        :param sample_pos: flag. If true, we only sample start and goal positions instead of sampling a new maze.
        :return: maze size, maze seed, start-goal positions, and map object
        """
        if not sample_pos:
            size = np.random.choice(maze_list)
            seed = np.random.choice(seed_list)
            env_map = mapper.RoughMap(size, seed, 3)
            # positions
            pos_params = env_map.get_start_goal_pos()
        else:
            size = env_map.maze_size
            seed = env_map.maze_seed
            start_pos, end_pos = env_map.sample_start_goal_pos(self.fix_start, self.fix_goal)
            # positions
            pos_params = [start_pos[0],
                          start_pos[1],
                          end_pos[0],
                          end_pos[1],
                          0]  # [init_pos, goal_pos, init_orientation]

        return size, seed, pos_params, env_map

    def toTransition(self, state, action, next_state, reward, goal, done):
        """
        Return the transitions based on goal-conditioned or non-goal-conditioned
        :param state: current state
        :param action: current action
        :param next_state: next state
        :param reward: reward
        :param done: terminal flag
        :param goal: current goal
        :return: A transition.
        """
        if not self.use_goal:
            return self.TRANSITION(state=self.toTensor(state),
                                   action=torch.tensor(action).long().view(-1, 1),
                                   reward=torch.tensor(reward).float().view(-1, 1),
                                   next_state=self.toTensor(next_state),
                                   done=torch.tensor(done).view(-1, 1))
        else:
            return self.TRANSITION(state=self.toTensor(state),
                                   action=torch.tensor(action).long().view(-1, 1),
                                   reward=torch.tensor(reward).float().view(-1, 1),
                                   next_state=self.toTensor(next_state),
                                   goal=self.toTensor(goal),
                                   done=torch.tensor(done).view(-1, 1))

    @staticmethod
    def toTensor(obs_list):
        """
        Function is used to convert the data type. In the current settings, the state obtained from the environment is a
        list of 8 RGB observations (numpy arrays). This function will change the list into a tensor with size
        8 x 3 x 64 x 64.
        :param obs_list: List of the 8 observations
        :return: state tensor
        """
        state_obs = torch.tensor(np.array(obs_list).transpose(0, 3, 1, 2)).float()
        return state_obs


