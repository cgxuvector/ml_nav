from utils import mapper
import numpy as np
from collections import namedtuple, defaultdict
import tqdm
from utils import memory
from utils import ml_schedule
import torch
import os
import sys
import IPython.terminal.debugger as Debug
import matplotlib.pyplot as plt
from scipy import ndimage
import random

DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "next_state", "reward", "goal", "done"])

ACTION_LIST = ['up', 'down', 'left', 'right']


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
                 eps_start=1,
                 eps_end=0.01,
                 device="cpu",
                 use_goal=False,
                 goal_dist=1,
                 future_k=4,
                 decal_freq=0.1,
                 use_true_state=False,
                 use_her=False):
        # randomness
        self.random_seed = random_seed
        # environment
        self.env = env
        self.env_map = None
        self.maze_size_list = maze_list
        self.maze_seed_list = seed_list
        self.fix_maze = fix_maze
        self.fix_start = fix_start
        self.fix_goal = fix_goal
        self.theme_list = ['MISHMASH']
        self.decal_list = [decal_freq]
        self.maze_size = None
        self.maze_seed = None
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
        self.use_her = use_her
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
        # recycle
        self.recycle_goal = False
        # last goal
        self.last_goal = None
        # sample goal distance
        self.goal_dist = goal_dist
        # orientation space
        self.init_orientation_space = np.linspace(0, 360, num=37).tolist()
        self.goal_orientation_space = np.linspace(0, 315, num=8).tolist()
        # future strategy
        self.her_future_k = future_k

        # use true state
        self.use_true_state = use_true_state

    # run statistics of the domain
    def run_statistic(self):
        """
                Function is used to run the training of the agent
                """
        # set the random seed
        random.seed(self.random_seed)

        # set the training statistics
        rewards = []  # list of rewards for one episode
        episode_t = 0  # time step for one episode

        # initial reset
        state, goal = self.init_map2d_and_maze3d()
        episode_count = 0
        success_count = 0
        for i in range(10):
            # start the training
            pbar = tqdm.trange(self.max_time_steps)
            for t in pbar:
                # randomly sample an action
                action = np.random.choice(range(4), 1).item()

                # step in the environment
                next_state, reward, done, dist, trans, _, _ = self.env.step(action)

                # check terminal
                if done or (episode_t == self.max_steps_per_episode):
                    episode_count += 1
                    if done:
                        success_count += 1
                    # reset the environments
                    rewards = []
                    episode_t = 0
                    state, goal = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
                else:
                    state = next_state
                    rewards.append(reward)
                    episode_t += 1
        print("Total number of episodes = {}".format(episode_count))
        print("Success number of episodes = {}".format(success_count))
        print("Success rate of navigation using random policy = {}".format(success_count / episode_count))

    def run_dqn(self):
        """
        Function is used to run the training of the agent
        """
        # set the random seed
        random.seed(self.random_seed)

        # set the training statistics
        rewards = []  # list of rewards for one episode
        episode_t = 0  # time step for one episode

        # initial reset
        state, goal = self.init_map2d_and_maze3d()

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)
            # obtain an action from epsilon greedy
            if np.random.sample() < eps:
                action = np.random.choice(range(4), 1).item()
            else:
                action = self.agent.get_action(self.toTensor(state)) if not self.use_goal else \
                         self.agent.get_action(self.toTensor(state), self.toTensor(goal))

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)

            # print(f"State = {state}, action={ACTION_LIST[action]}, next state = {next_state}, dist = {dist}, done={done}")

            # store the replay buffer and convert the data to tensor
            if self.use_relay_buffer:
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

                # reset the environments
                rewards = []
                episode_t = 0
                state, goal = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
            else:
                state = next_state
                rewards.append(reward)
                episode_t += 1

            # train the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        model_save_path = os.path.join(self.save_dir, self.model_name) + ".pt"
        distance_save_path = os.path.join(self.save_dir, self.model_name + "_distance.npy")
        returns_save_path = os.path.join(self.save_dir, self.model_name + "_return.npy")
        lengths_save_path = os.path.join(self.save_dir, self.model_name + "_length.npy")
        torch.save(self.agent.policy_net.state_dict(), model_save_path)
        np.save(distance_save_path, self.distance)
        np.save(returns_save_path, self.returns)
        np.save(lengths_save_path, self.lengths)

    def run_dqn_her(self):
        """
        Function is used to run training of the DQN agent using HER
        Single goal setting
        """
        # set randomness
        random.seed(self.random_seed)  # once set, it has global effect

        # running statistics
        episode_t = 0
        states_buffer = []
        actions_buffer = []
        rewards_buffer = []
        trans_buffer = []
        dones_buffer = []

        # reset the environment
        state, goal = self.init_map2d_and_maze3d()
        states_buffer.append(state)

        # start training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)
            # obtain an action from epsilon greedy
            if random.uniform(0, 1) < eps:
                action = np.random.choice(range(4), 1).item()
            else:
                action = self.agent.get_action(self.toTensor(state)) if not self.use_goal else \
                    self.agent.get_action(self.toTensor(state), self.toTensor(goal))

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)

            """ update the on-policy buffers """
            states_buffer.append(next_state)
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            trans_buffer.append(trans)
            dones_buffer.append(done)

            """ add the transition into the replay buffer """
            # store the replay buffer and convert the data to tensor
            if self.use_relay_buffer:
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                self.replay_buffer.add(trans)

            """ check the terminal """
            if done or (episode_t == self.max_steps_per_episode):
                # compute the return
                G = 0
                for r in reversed(rewards_buffer):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance
                self.returns.append(G)
                self.lengths.append(episode_t)
                self.distance.append(dist)
                # compute the episode number
                episode_idx = len(self.returns)

                # print the information
                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos} | Goal: {self.env.goal_pos} | '
                    f'Eps: {eps:.3f} | Buffer: {len(self.replay_buffer)}'
                )

                """ check if using the HER strategy """
                if self.use_her:
                    self.hindsight_experience_replay(states_buffer,
                                                     actions_buffer,
                                                     rewards_buffer,
                                                     trans_buffer,
                                                     goal,
                                                     dones_buffer)

                episode_t = 0  # reset the time step counter
                states_buffer = []  # reset the states buffer
                actions_buffer = []  # reset the actions buffer
                rewards_buffer = []  # reset the rewards
                trans_buffer = []  # reset the next state buffer
                dones_buffer = []  # reset the dones buffer

                # reset the environment
                state, goal = self.update_map2d_and_maze3d(set_new_maze= not self.fix_maze)
                states_buffer.append(state)
            else:
                state = next_state
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

    def random_goal_conditioned_her_run(self):
        """
               Run experiments using goal-conditioned DQN with the nearby goals using HER
               :return: None

               Question: how to set the exploration?
        """
        # set randomness
        random.seed(self.random_seed)  # once set, it has global effect

        # select a maze
        size = np.random.choice(self.maze_size_list)
        seed = np.random.choice(self.maze_seed_list)

        # select a map
        env_map = mapper.RoughMap(size, seed, 3)

        # randomly select the init and goal positions
        init_pos, goal_pos = env_map.sample_global_start_goal_pos(False, False, self.goal_dist)
        env_map.update_mapper(init_pos, goal_pos)

        # reset the environment
        state, goal = self.init_map2d_and_maze3d()

        # running statistics
        episode_t = 0
        train_episode_count = self.train_episode_num
        states_buffer = [state]
        actions_buffer = []
        rewards_buffer = []
        dones_buffer = []
        trans_buffer = []

        # start training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)
            # obtain an action from epsilon greedy
            if np.random.sample() < eps:
                action = np.random.choice(range(4), 1).item()
            else:
                action = self.agent.get_action(self.toTensor(state)) if not self.use_goal else \
                         self.agent.get_action(self.toTensor(state), self.toTensor(goal))

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)

            """ update the on-policy buffers """
            states_buffer.append(next_state)
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            dones_buffer.append(done)
            trans_buffer.append(trans)

            """ add the transition into the replay buffer """
            # store the replay buffer and convert the data to tensor
            if self.use_relay_buffer:
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                self.replay_buffer.add(trans)

            """ check the terminal """
            if done or (episode_t == self.max_steps_per_episode):
                # compute the return
                G = 0
                for r in reversed(rewards_buffer):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance
                self.returns.append(G)
                self.lengths.append(episode_t)
                self.distance.append(dist)
                # compute the episode number
                episode_idx = len(self.returns)

                # print the information
                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos} | Goal: {self.env.goal_pos} | '
                    f'Eps: {eps:.3f} | Buffer: {len(self.replay_buffer)}'
                )

                """ check if using the HER strategy """
                if not self.use_relay_buffer:
                    self.hindsight_experience_replay(states_buffer,
                                                     actions_buffer,
                                                     rewards_buffer,
                                                     trans_buffer,
                                                     goal,
                                                     dones_buffer)

                # reset the environments
                episode_t = 0  # reset the time step counter
                rewards_buffer = []  # reset the rewards
                states_buffer = []  # reset the states buffer
                actions_buffer = []  # reset the actions buffer
                trans_buffer = []  # reset the transition buffer
                dones_buffer = []  # reset the dones buffer

                # for fixed start and goal positions, train it for #(train_episode_num) episodes
                if train_episode_count > 1:
                    train_episode_count -= 1
                else:
                    state, goal = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
                    # reset the training episodes
                    train_episode_count = self.train_episode_num
                # init the buffers
                states_buffer.append(state)
            else:
                state = next_state
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
            start_pos, end_pos = env_map.sample_global_start_goal_pos(self.fix_start, self.fix_goal, self.goal_dist)
            # sample the init-goal orientations
            init_ori = random.sample(self.init_orientation_space, 1)[0]
            goal_ori = random.sample(self.goal_orientation_space, 1)[0]
            # positions
            pos_params = [start_pos[0],
                          start_pos[1],
                          end_pos[0],
                          end_pos[1],
                          init_ori,
                          goal_ori]  # [init_pos, goal_pos, init_orientation]
        return size, seed, pos_params, env_map

    # adding transition to the buffer using HER
    def hindsight_experience_replay(self, states, actions, rewards, trans, goal, dones):
        states_num = len(states) - 1
        for t in range(states_num):
            state = states[t]  # s_t
            action = actions[t]  # a_t
            reward = rewards[t]  # r_t
            next_state = states[t + 1]  # s_t+1
            next_state_pos = trans[t]  # position of the current next state
            done = dones[t]  # done_t
            transition = self.toTransition(state, action, next_state, reward, goal, done)  # add the current transition
            self.replay_buffer.add(transition)
            # Hindsight Experience Replay
            future_indices = list(range(t+1, len(states)))
            sampled_goals = random.sample(future_indices, self.her_future_k) if len(future_indices) >= self.her_future_k else future_indices
            for idx in sampled_goals:
                # relabel the new goal
                new_goal = states[idx]
                # obtain the new goal position
                new_goal_pos = trans[idx - 1]
                distance = self.env.compute_distance(next_state_pos, new_goal_pos)
                new_reward = self.env.compute_reward(distance)
                new_done = 0 if new_reward == -1 else 1
                transition = self.toTransition(state, action, next_state, new_reward, new_goal, new_done)
                self.replay_buffer.add(transition)

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
                                   action=torch.tensor(action, dtype=torch.int8).view(-1, 1),
                                   reward=torch.tensor(reward, dtype=torch.int8).view(-1, 1),
                                   next_state=self.toTensor(next_state),
                                   done=torch.tensor(done, dtype=torch.int8).view(-1, 1))
        else:
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
        if not self.use_true_state:
            state_obs = torch.tensor(np.array(obs_list).transpose(0, 3, 1, 2), dtype=torch.uint8)
        else:
            state_obs = torch.tensor(np.array(obs_list), dtype=torch.float32)
        return state_obs

    def init_map2d_and_maze3d(self):
        # randomly select a maze
        self.maze_size = random.sample(self.maze_size_list, 1)[0]
        self.maze_seed = random.sample(self.maze_seed_list, 1)[0]
        # initialize the map 2D
        self.env_map = mapper.RoughMap(self.maze_size, self.maze_seed, 3)
        # initialize the maze 3D
        maze_configs = defaultdict(lambda: None)
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
        # obtain the state and goal observation
        state_obs, goal_obs, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs

    def update_map2d_and_maze3d(self, set_new_maze=False):
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
            init_pos, goal_pos = self.env_map.sample_global_start_goal_pos(self.fix_start, self.fix_goal, 100)
            self.env_map.update_mapper(init_pos, goal_pos)
            maze_configs['start_pos'] = init_pos + [0]
            maze_configs['goal_pos'] = goal_pos + [0]
            maze_configs['maze_valid_pos'] = self.env_map.valid_pos
            maze_configs['update'] = False

        # obtain the state and goal observation
        state_obs, goal_obs, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs
