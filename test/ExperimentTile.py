from utils import mapper
import numpy as np
from collections import namedtuple, defaultdict
import tqdm
from utils import memory
from utils import ml_schedule
import torch
import random
import os
import time
import matplotlib.pyplot as plt
import IPython.terminal.debugger as Debug


# define the global default parameters
DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "next_state", "reward", "goal", "done"])
DEFAULT_ACTION_LIST = ['up', 'down', 'left', 'right']


class Experiment(object):
    def __init__(self,
                 env,  # environment configurations
                 agent,
                 maze_list,
                 seed_list,
                 decal_freq=0.1,
                 fix_maze=True,
                 fix_start=True,
                 fix_goal=True,
                 use_goal=False,
                 goal_dist=np.inf,
                 use_true_state=False,  # state configurations
                 train_local_policy=False,
                 train_episode_num=10,  # training configurations
                 start_train_step=1000,
                 max_time_steps=50000,
                 episode_time_steps=100,
                 eval_policy_freq=10,
                 use_replay=False,
                 use_her=False,  # HER configurations
                 future_k=4,
                 buffer_size=20000,
                 transition=DEFAULT_TRANSITION,
                 learning_rate=1e-3,  # optimization configurations
                 batch_size=64,
                 gamma=0.99,  # RL configurations
                 eps_start=1,  # exploration configurations
                 eps_end=0.01,
                 save_dir=None,  # saving configurations
                 model_name=None
                 ):
        # environment
        self.env = env
        self.env_map = None
        self.maze_size = None
        self.maze_seed = None
        self.maze_size_list = maze_list
        self.maze_seed_list = seed_list
        self.fix_maze = fix_maze
        self.fix_start = fix_start
        self.fix_goal = fix_goal
        self.theme_list = ['MISHMASH']
        self.decal_list = [decal_freq]
        # agent
        self.agent = agent
        # state configurations
        self.use_true_state = use_true_state
        # goal-conditioned configurations
        self.use_goal = use_goal
        self.goal_dist = goal_dist
        # training configurations
        self.train_local_policy = train_local_policy
        self.train_episode_num = train_episode_num
        self.start_train_step = start_train_step
        self.max_time_steps = max_time_steps
        self.max_steps_per_episode = episode_time_steps
        self.eval_policy_freq = eval_policy_freq
        # replay buffer configurations
        if buffer_size:
            self.replay_buffer = memory.ReplayMemory(buffer_size, transition)
            self.TRANSITION = transition
        self.batch_size = batch_size
        self.use_replay_buffer = use_replay
        # HER configurations
        self.use_her = use_her
        self.her_future_k = future_k  # future strategy
        # optimization configurations
        self.learning_rate = learning_rate
        # rl related configuration
        self.gamma = gamma
        # exploration configuration
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.schedule = ml_schedule.LinearSchedule(eps_start, eps_end, max_time_steps/2)
        # results statistics
        self.distance = []
        self.returns = []
        self.lengths = []
        self.policy_returns = []
        # saving settings
        self.model_name = model_name
        self.save_dir = save_dir

    def run_dqn(self):
        """
        Function is used to train the vanilla double DQN agent.
        """
        # set the training statistics
        rewards = []  # list of rewards for one episode
        episode_t = 0  # time step counter for one episode

        # initialize the start and goal positions
        state, goal = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # get action
            action = self.agent.get_action(state, eps)

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)

            # store the replay buffer and convert the data to tensor
            if self.use_replay_buffer:
                # construct the transition
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                # add the transition into the buffer
                self.replay_buffer.add(trans)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                self.lengths.append(episode_t)
                self.distance.append(dist)
                # compute the episode number
                episode_idx = len(self.returns)

                # tdqm bar display function
                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos} | Goal: {self.env.goal_pos} | '
                    f'Eps: {eps:.3f} | Buffer: {len(self.replay_buffer)}'
                )

                # evaluate the current policy
                if (episode_idx - 1) % self.eval_policy_freq == 0:
                    # evaluate the current policy by interaction
                    self.policy_evaluate()

                # reset the environments
                rewards = []
                episode_t = 0
                state, goal = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
            else:
                state = next_state
                rewards.append(reward)
                episode_t += 1

            # start training the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save the results
        self.save_results()

    def run_goal_dqn(self):
        """
        Function is used to train the globally goal-conditioned double DQN.
        """
        # set the training statistics
        rewards = []  # list of rewards for one episode
        episode_t = 0  # time step for one episode

        # update the start-goal positions
        state, goal = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # get action
            action = self.agent.get_action(state, goal, eps)

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)

            # store the replay buffer and convert the data to tensor
            if self.use_replay_buffer:
                # construct the transition
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                # add the transition into the buffer
                self.replay_buffer.add(trans)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                self.lengths.append(episode_t)
                self.distance.append(dist)
                # compute the episode number
                episode_idx = len(self.returns)

                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos} | Goal: {self.env.goal_pos} | '
                    f'Eps: {eps:.3f} | Buffer: {len(self.replay_buffer)}'
                )

                # evaluate the current policy
                if (episode_idx - 1) % self.eval_policy_freq == 0:
                    # evaluate the current policy by interaction
                    self.policy_evaluate()

                # reset the environments
                rewards = []
                episode_t = 0
                state, goal = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
            else:
                state = next_state
                rewards.append(reward)
                episode_t += 1

            # start training the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save the results
        self.save_results()

    def run_random_local_goal_dqn(self):
        """
        Function is used to train the locally goal-conditioned double DQN.
        """
        # set the training statistics
        rewards = []
        episode_t = 0  # time step for one episode
        train_episode_num = self.train_episode_num  # training number for each start-goal pair

        # initialize the state and goal
        state, goal = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # get action
            action = self.agent.get_action(state, goal, eps)

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)
            episode_t += 1

            # store the replay buffer and convert the data to tensor
            if self.use_replay_buffer:
                # construct the transition
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                # add the transition into the buffer
                self.replay_buffer.add(trans)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                self.lengths.append(episode_t)
                self.distance.append(dist)
                # compute the episode number
                episode_idx = len(self.returns)

                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos} | Goal: {self.env.goal_pos} | '
                    f'Eps: {eps:.3f} | Buffer: {len(self.replay_buffer)}'
                )

                # evaluate the current policy
                #if (episode_idx - 1) % self.eval_policy_freq == 0:
                    # evaluate the current policy by interaction
                #    self.policy_evaluate()

                # reset the environments
                rewards = []
                episode_t = 0
                # train a pair of start and goal with fixed number of episodes
                if train_episode_num > 0:
                    # keep the same start and goal
                    self.fix_start = True
                    self.fix_goal = True
                    state, goal = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
                    train_episode_num -= 1
                else:
                    # sample a new pair of start and goal
                    self.fix_start = False
                    self.fix_goal = False
                    state, goal = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
                    train_episode_num = self.train_episode_num
            else:
                state = next_state
                rewards.append(reward)

            # train the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save results
        self.save_results()

    def toTransition(self, state, action, next_state, reward, goal, done):
        """
        Function is used to construct a transition using state, action, next_state, reward, goal, done.
        """
        if not self.use_goal:  # construct non goal-conditioned transition (Default type is int8 to save memory)
            return self.TRANSITION(state=self.toTensor(state),
                                   action=torch.tensor(action, dtype=torch.int8).view(-1, 1),
                                   reward=torch.tensor(reward, dtype=torch.int8).view(-1, 1),
                                   next_state=self.toTensor(next_state),
                                   done=torch.tensor(done, dtype=torch.int8).view(-1, 1))
        else:  # construct goal-conditioned transition
            return self.TRANSITION(state=self.toTensor(state),
                                   action=torch.tensor(action, dtype=torch.int8).view(-1, 1),
                                   reward=torch.tensor(reward, dtype=torch.int8).view(-1, 1),
                                   next_state=self.toTensor(next_state),
                                   goal=self.toTensor(goal),
                                   done=torch.tensor(done, dtype=torch.int8).view(-1, 1))

    def toTensor(self, obs_list):
        """
        Function is used to convert the data type. In the current settings, the state obtained from the environment is a
        list of 8 RGB observations (numpy arrays). This function will change the list into a tensor with size
        8 x 3 x 64 x 64.
        :param obs_list: List of the 8 observations
        :return: state tensor
        """
        if not self.use_true_state:  # convert the state observation from numpy to tensor with correct size
            state_obs = torch.tensor(np.array(obs_list).transpose(0, 3, 1, 2), dtype=torch.uint8)
        else:
            state_obs = torch.tensor(np.array(obs_list), dtype=torch.float32)
        return state_obs

    def update_map2d_and_maze3d(self, set_new_maze=False):
        """
        Function is used to update the 2D map and the 3D maze.
        """
        # set maze configurations
        maze_configs = defaultdict(lambda: None)
        # set new maze flag
        if set_new_maze:
            # randomly select a maze
            self.maze_size = random.sample(self.maze_size_list, 1)[0]
            self.maze_seed = random.sample(self.maze_seed_list, 1)[0]
            # initialize the map 2D
            self.env_map = mapper.RoughMap(self.maze_size, self.maze_seed, 3)
            # initialize the maze 3D
            maze_configs["maze_name"] = f"maze_{self.maze_size}x{self.maze_size}"  # string type name
            maze_configs["maze_size"] = [self.maze_size, self.maze_size]  # [int, int] list
            maze_configs["maze_seed"] = '1234'  # string type number
            maze_configs["maze_texture"] = random.sample(self.theme_list, 1)[0]  # string type name in theme_list
            maze_configs["maze_decal_freq"] = random.sample(self.decal_list, 1)[0]  # float number in decal_list
            maze_configs["maze_map_txt"] = "".join(self.env_map.map2d_txt)  # string type map
            maze_configs["maze_valid_pos"] = self.env_map.valid_pos  # list of valid positions
            # initialize the maze start and goal positions
            maze_configs["start_pos"] = self.env_map.init_pos + [0]  # start position on the txt map [rows, cols, orientation]
            maze_configs["goal_pos"] = self.env_map.goal_pos + [0]  # goal position on the txt map [rows, cols, orientation]
            # initialize the update flag
            maze_configs["update"] = True  # update flag
        else:
            if not self.train_local_policy:
                init_pos, goal_pos = self.env_map.sample_global_start_goal_pos(self.fix_start, self.fix_goal)
            else:
                init_pos, goal_pos = self.env_map.sample_random_start_goal_pos(self.fix_start, self.fix_goal,
                                                                               self.goal_dist)
            # self.env_map.update_mapper(init_pos, goal_pos)
            maze_configs['start_pos'] = init_pos + [0]
            maze_configs['goal_pos'] = goal_pos + [0]
            maze_configs['maze_valid_pos'] = self.env_map.valid_pos
            maze_configs['update'] = False

        # obtain the state and goal observation
        state_obs, goal_obs, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs

    # evaluate the policy during training
    def policy_evaluate(self):
        # reset the environment
        self.fix_start, self.fix_goal = True, True
        state, goal = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
        # record the statistics
        rewards = []
        actions = []
        # run the policy
        for i in range(self.max_steps_per_episode):
            # get one action
            if self.use_goal:
                action = self.agent.get_action(state, goal, 0)
            else:
                action = self.agent.get_action(state, 0)
            actions.append(DEFAULT_ACTION_LIST[action])

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)

            # increase the statistics
            rewards.append(reward)

            # check terminal
            if done:
                break
            else:
                state = next_state

        # compute the discounted return for each time step
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G

        # store the current policy return
        print("Return = {} and {}".format(G, actions[0:30]))
        self.policy_returns.append(G)

    # save the results
    def save_results(self):
        # compute the path for the results
        model_save_path = os.path.join(self.save_dir, self.model_name) + ".pt"
        distance_save_path = os.path.join(self.save_dir, self.model_name + "_distance.npy")
        returns_save_path = os.path.join(self.save_dir, self.model_name + "_return.npy")
        policy_returns_save_path = os.path.join(self.save_dir, self.model_name + "_policy_return.npy")
        lengths_save_path = os.path.join(self.save_dir, self.model_name + "_length.npy")
        # save the results
        torch.save(self.agent.policy_net.state_dict(), model_save_path)
        np.save(distance_save_path, self.distance)
        np.save(returns_save_path, self.returns)
        np.save(lengths_save_path, self.lengths)
        np.save(policy_returns_save_path, self.policy_returns)
