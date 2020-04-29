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
import random

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
                 use_goal=False,
                 goal_dist=1,
                 future_k=4):
        # randomness
        self.rnd_seed = random_seed
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
        # set the random seed
        np.random.seed(self.seed_rnd)
        # select a maze
        size = np.random.choice(self.maze_list)
        seed = np.random.choice(self.seed_list)
        # select the map
        env_map = mapper.RoughMap(size, seed, 3)
        # get the start and goal positions
        pos_params = env_map.get_start_goal_pos()
        # get the first middle sub-goal
        goal_step = 1
        pos_params[2:4] = env_map.sample_path_next_goal(goal_step)
        goal_step += 1
        # reset the environment
        state, goal = self.env.reset(size, seed, pos_params)
        self.last_goal = goal
        # plot the goal and the current observations
        fig, arrs = plt.subplots(3, 3)
        img1 = arrs[1, 2].imshow(goal[0])
        img2 = arrs[0, 2].imshow(goal[1])
        img3 = arrs[0, 1].imshow(goal[2])
        img4 = arrs[0, 0].imshow(goal[3])
        img5 = arrs[1, 0].imshow(goal[4])
        top_down_img = arrs[1, 1].imshow(ndimage.rotate(self.env.top_down_obs, -90))
        img6 = arrs[2, 0].imshow(goal[5])
        img7 = arrs[2, 1].imshow(goal[6])
        img8 = arrs[2, 2].imshow(goal[7])

        # running statistics
        rewards = []
        episode_t = 0
        sampled_goal_count = len(env_map.path)
        train_episode_count = self.train_episode_num

        # start training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            """ select an action using epsilon greedy"""
            # compute the epsilon
            eps = self.schedule.get_value(t)
            # get an action from epsilon greedy
            if np.random.sample() < eps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.get_action(self.toTensor(state)) if not self.use_goal else self.agent.get_action(
                    self.toTensor(state), self.toTensor(self.last_goal))

            """ apply the action in the 3D maze"""
            # step in the environment
            next_state, reward, done, dist, _ = self.env.step(action)

            # show the current observation and goal
            img1.set_data(goal[0])
            img2.set_data(goal[1])
            img3.set_data(goal[2])
            img4.set_data(goal[3])
            img5.set_data(goal[4])
            top_down_img.set_data(ndimage.rotate(self.env.top_down_obs, -90))
            img6.set_data(goal[5])
            img7.set_data(goal[6])
            img8.set_data(goal[7])
            fig.canvas.draw()
            plt.pause(0.0001)

            """ add the transition into the replay buffer"""
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
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | Init: {pos_params[0:2]} | Goal: {pos_params[2:4]} | Eps: {eps:.3f}'
                )
                # reset the environments
                rewards = []  # reset the rewards
                episode_t = 0  # reset the time step counter

                # for fixed start and goal positions, train it for #(train_episode_num) episodes
                if train_episode_count > 1:
                    train_episode_count -= 1
                else:
                    # for fixed start, sample the next middle sub-goal
                    if sampled_goal_count > 1:
                        # sample the next sub-goal
                        pos_params[2:4] = env_map.sample_path_next_goal(goal_step)
                        goal_step += 1
                        sampled_goal_count -= 1
                    else:
                        # sample another trajectory or sample a new maze
                        size, seed, pos_params, env_map = self.map_sampling(env_map, self.maze_list, self.seed_list,
                                                                            self.fix_maze)
                        if self.recycle_goal:
                            # sample the first middle sub-goal
                            goal_step = 1
                            pos_params[2:4] = env_map.sample_path_next_goal(goal_step)
                            goal_step += 1
                        sampled_goal_count = len(env_map.path)
                    train_episode_count = self.train_episode_num
                # reset the environment
                self.last_goal = goal
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

    def random_goal_conditioned_run(self):
        """
               Run experiments using goal-conditioned DQN with the nearby goals
               :return: None
               """
        # set the random seed
        np.random.seed(self.seed_rnd)
        # select a maze
        size = np.random.choice(self.maze_list)
        seed = np.random.choice(self.seed_list)
        # select the map
        env_map = mapper.RoughMap(size, seed, 3)
        pos_params = env_map.get_start_goal_pos()
        # get the start and goal positions
        pos_params[0:2], pos_params[2:4] = env_map.sample_global_start_goal_pos(False, False, self.goal_dist)
        # reset the environment
        state, goal = self.env.reset(size, seed, pos_params)
        # plot the goal and the current observations
        fig, arrs = plt.subplots(3, 3)
        img1 = arrs[1, 2].imshow(goal[0])
        img2 = arrs[0, 2].imshow(goal[1])
        img3 = arrs[0, 1].imshow(goal[2])
        img4 = arrs[0, 0].imshow(goal[3])
        img5 = arrs[1, 0].imshow(goal[4])
        top_down_img = arrs[1, 1].imshow(ndimage.rotate(self.env.top_down_obs, 0))
        img6 = arrs[2, 0].imshow(goal[5])
        img7 = arrs[2, 1].imshow(goal[6])
        img8 = arrs[2, 2].imshow(goal[7])

        # running statistics
        rewards = []
        episode_t = 0
        train_episode_count = self.train_episode_num

        # start training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            """ select an action using epsilon greedy"""
            # compute the epsilon
            eps = self.schedule.get_value(t)
            # get an action from epsilon greedy
            if np.random.sample() < eps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.get_action(self.toTensor(state)) if not self.use_goal else self.agent.get_action(
                    self.toTensor(state), self.toTensor(self.last_goal))

            """ apply the action in the 3D maze"""
            # step in the environment
            next_state, reward, done, dist, _ = self.env.step(action)

            # show the current observation and goal
            img1.set_data(goal[0])
            img2.set_data(goal[1])
            img3.set_data(goal[2])
            img4.set_data(goal[3])
            img5.set_data(goal[4])
            top_down_img.set_data(ndimage.rotate(self.env.top_down_obs, -90))
            # map_img.set_data(env_map.map2d_bw)
            img6.set_data(goal[5])
            img7.set_data(goal[6])
            img8.set_data(goal[7])
            fig.canvas.draw()
            plt.pause(0.0001)

            """ add the transition into the replay buffer"""
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
                        f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | Init: {pos_params[0:2]} | Goal: {pos_params[2:4]} | Eps: {eps:.3f}'
                )
                # reset the environments
                rewards = []  # reset the rewards
                episode_t = 0  # reset the time step counter

                # for fixed start and goal positions, train it for #(train_episode_num) episodes
                if train_episode_count > 1:
                    train_episode_count -= 1
                else:
                    size, seed, pos_params, env_map = self.map_sampling(env_map, self.maze_list, self.seed_list,
                                                                        self.fix_maze)
                    train_episode_count = self.train_episode_num
                # reset the environment
                self.last_goal = goal
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

    def random_goal_conditioned_her_run(self):
        """
               Run experiments using goal-conditioned DQN with the nearby goals using HER
               :return: None
        """
        # set randomness
        random.seed(self.rnd_seed)  # !!! once set, the seed is implicitly affect the whole program

        # select a maze
        size = np.random.choice(self.maze_list)
        seed = np.random.choice(self.seed_list)

        # select the map
        env_map = mapper.RoughMap(size, seed, 3)
        pos_params = env_map.get_start_goal_pos()

        # get the start and goal positions
        pos_params[0:2], pos_params[2:4] = env_map.sample_global_start_goal_pos(False, False, self.goal_dist)
        # sample the orientation
        init_ori = random.sample(self.init_orientation_space, 1)[0]
        goal_ori = random.sample(self.goal_orientation_space, 1)[0]
        pos_params[4], pos_params[5] = init_ori, goal_ori

        # reset the environment
        state, goal, trans, _ = self.env.reset(size, seed, pos_params)

        # # plot the goal and the current observations
        # fig, arrs = plt.subplots(3, 3)
        # view_state = state
        # # agent-based egocentric 360 degree observations
        # arrs[0, 1].set_title("front")
        # img1 = arrs[0, 1].imshow(view_state[0])
        # arrs[0, 0].set_title("front-left")
        # img2 = arrs[0, 0].imshow(view_state[1])
        # arrs[1, 0].set_title("left")
        # img3 = arrs[1, 0].imshow(view_state[2])
        # arrs[2, 0].set_title("back-left")
        # img4 = arrs[2, 0].imshow(view_state[3])
        # arrs[2, 1].set_title("back")
        # img5 = arrs[2, 1].imshow(view_state[4])
        # arrs[1, 1].set_title("top-down")
        # top_down_img = arrs[1, 1].imshow(ndimage.rotate(self.env.top_down_obs, 0))
        # arrs[2, 2].set_title("back-right")
        # img6 = arrs[2, 2].imshow(view_state[5])
        # arrs[1, 2].set_title("right")
        # img7 = arrs[1, 2].imshow(view_state[6])
        # arrs[0, 2].set_title("front-right")
        # img8 = arrs[0, 2].imshow(view_state[7])

        # running statistics
        episode_t = 0
        train_episode_count = self.train_episode_num

        # position list
        states_buffer = [state]
        actions_buffer = []
        rewards_buffer = []
        dones_buffer = []
        trans_buffer = []

        # start training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            """ select an action using epsilon greedy"""
            # compute the epsilon
            eps = self.schedule.get_value(t)
            # get an action from epsilon greedy
            if random.uniform(0, 1) < eps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.get_action(self.toTensor(state)) if not self.use_goal else self.agent.get_action(
                    self.toTensor(state), self.toTensor(goal))

            """ apply the action in the 3D maze"""
            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)

            """ update the on-policy buffers """
            states_buffer.append(next_state)
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            dones_buffer.append(done)
            trans_buffer.append(trans)

            # # show the current observation and goal
            # view_state = next_state
            # img1.set_data(view_state[0])
            # img2.set_data(view_state[1])
            # img3.set_data(view_state[2])
            # img4.set_data(view_state[3])
            # img5.set_data(view_state[4])
            # top_down_img.set_data(ndimage.rotate(self.env.top_down_obs, 0))
            # img6.set_data(view_state[5])
            # img7.set_data(view_state[6])
            # img8.set_data(view_state[7])
            # fig.canvas.draw()
            # plt.pause(0.0001)

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
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | Init: {pos_params[0:2]} | Goal: {pos_params[2:4]} | Eps: {eps:.3f} | Buffer: {len(self.replay_buffer)}'
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
                    # sample the next maze, init-goal positions
                    size, seed, pos_params, env_map = self.map_sampling(env_map, self.maze_list, self.seed_list,
                                                                        self.fix_maze)
                    # reset the training episodes
                    train_episode_count = self.train_episode_num
                # reset the environment
                state, goal, trans, _ = self.env.reset(size, seed, pos_params)
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
            next_state = states[t+1]  # s_t+1
            next_state_pos = trans[t]  # position of the current next state
            done = dones[t]  # done_t
            transition = self.toTransition(state, action, next_state, reward, goal, done)  # add the current transition
            self.replay_buffer.add(transition)
            # Hindsight Experience Replay
            future_indices = list(range(t, len(states)-1))
            sampled_goals = random.sample(future_indices, self.her_future_k) if len(future_indices) >= self.her_future_k else future_indices
            for idx in sampled_goals:
                new_goal = states[idx]
                new_goal_pos = trans[idx]
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

    @staticmethod
    def toTensor(obs_list):
        """
        Function is used to convert the data type. In the current settings, the state obtained from the environment is a
        list of 8 RGB observations (numpy arrays). This function will change the list into a tensor with size
        8 x 3 x 64 x 64.
        :param obs_list: List of the 8 observations
        :return: state tensor
        """
        state_obs = torch.tensor(np.array(obs_list).transpose(0, 3, 1, 2), dtype=torch.uint8)
        return state_obs


