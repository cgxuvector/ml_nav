from utils import mapper
import copy
from collections import namedtuple, defaultdict
import tqdm
from model import VAE
from utils import memory
from utils import ml_schedule
import torch
import random
import os
import numpy as np
import pickle
import time
import IPython.terminal.debugger as Debug
import matplotlib.pyplot as plt


# define the global default parameters
DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "next_state", "reward", "goal", "done"])
DEFAULT_ACTION_LIST = ['up', 'down', 'left', 'right']
DEFAULT_ACTION_RAW = ['up', 'down', 'right', 'left']


class Experiment(object):
    def __init__(self,
                 env,  # environment configurations
                 agent,
                 maze_list,
                 seed_list,
                 args,
                 decal_freq=0.1,
                 fix_maze=True,
                 fix_start=True,
                 fix_goal=True,
                 use_goal=False,
                 goal_dist=np.inf,
                 use_true_state=False,  # state configurations
                 train_random_policy=False,
                 sample_start_goal_num=10,
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
                 model_name=None,
                 use_imagine=0,  # imagination flag
                 device='cpu',
                 use_state_est=False
                 ):
        self.args = args
        self.device = torch.device(device)
        # environment
        self.env = env
        self.env_test = copy.copy(env)
        self.env_map = None
        self.env_map_test = None
        self.maze_size = maze_list[0] if len(maze_list) == 1 else None
        self.maze_seed = seed_list[0] if len(seed_list) == 1 else None
        self.maze_size_list = maze_list
        self.maze_seed_list = seed_list
        self.fix_maze = fix_maze
        self.fix_start = fix_start
        self.fix_goal = fix_goal
        self.theme_list = ['MISHMASH']
        self.decal_list = [decal_freq]
        # agent
        self.agent = agent
        # use state estimation
        self.use_state_est = use_state_est
        # state configurations
        self.use_true_state = use_true_state
        # goal-conditioned configurations
        self.use_goal = use_goal
        self.goal_dist = goal_dist
        self.valid_goal_dist = list(range(1, goal_dist+1, 1)) if goal_dist > 0 else list(range(1, 50, 1))
        self.use_imagine = use_imagine
        # generate imagined goals
        self.orientations = [torch.tensor([0, 0, 0, 0, 1, 0, 0, 0]),
                             torch.tensor([0, 0, 1, 0, 0, 0, 0, 0]),
                             torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]),
                             torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]),
                             torch.tensor([0, 0, 0, 1, 0, 0, 0, 0]),
                             torch.tensor([0, 0, 0, 0, 0, 1, 0, 0]),
                             torch.tensor([0, 0, 0, 0, 0, 0, 1, 0]),
                             torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])]
        if self.use_imagine:
            self.thinker = VAE.CVAE(64, use_small_obs=True)
            #self.thinker.load_state_dict(torch.load("/mnt/cheng_results/VAE/models/small_obs_L64_B8.pt",
            #                                         map_location=self.device))
            self.thinker.load_state_dict(torch.load("/mnt/sda/dataset/ml_nav/VAE/model/small_obs_L64_B8.pt"))
            #self.thinker.load_state_dict(torch.load("./results/vae/model/small_obs_L64_B8.pt", map_location=self.device))
            self.thinker.eval()
        # training configurations
        self.train_random_policy = train_random_policy
        self.sample_start_goal_num = sample_start_goal_num
        self.train_episode_num = train_episode_num
        self.start_train_step = start_train_step
        self.max_time_steps = max_time_steps
        self.max_steps_per_episode = episode_time_steps
        self.eval_policy_freq = eval_policy_freq
        # replay buffer configurations
        if buffer_size:
            self.replay_buffer = memory.ReplayMemory(buffer_size, transition)
            # for explore case
            self.replay_buffer_explore = memory.ReplayMemory(buffer_size, transition)
            # for exploit case
            self.replay_buffer_exploit = memory.ReplayMemory(buffer_size, transition)
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
        self.steps = []
        self.lengths = []
        self.policy_returns = []
        self.policy_eval = 0
        # saving settings
        self.model_name = model_name
        self.save_dir = save_dir

    def run_dqn(self):
        """
            Function is used to train the vanilla or double DQN agent.
        """
        print(f"Experiment: Run {self.agent.dqn_mode} DQN.")
        # set the training statistics
        rewards = []  # reward list for one episode
        episode_t = 0  # time step counter

        # initialize the start and goal positions
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # get action
            action = self.agent.get_action(state, eps)

            # step in the environment
            next_state, reward, done, dist, trans, _, _, _, _ = self.env.step(action)
            #print(f'ep id={t}, s = {state}, a = {DEFAULT_ACTION_RAW[action]}, next_s={next_state}, g={goal}, dist={dist}, done={done}, reward={reward}')

            # store the replay buffer and convert the data to tensor
            if self.use_replay_buffer:
                # construct the transition
                trans = self.toTransition(state, action, next_state, reward, _, done)
                # add the transition into the buffer
                self.replay_buffer.add(trans)

            # increment state
            state = next_state
            episode_t += 1
            rewards.append(reward)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                #print('----------------------------------------------------')
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                self.steps.append(t)
                episode_idx = len(self.returns)

                # tdqm bar display function
                pbar.set_description(
                    f'Episode: {episode_idx} | '
                    f'Steps: {episode_t} | '
                    f'Return: {G:2f} | '
                    f'Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos_map} | '
                    f'Goal: {self.env.goal_pos_map} | '
                    f'Eps: {eps:.3f} | '
                    f'Buffer: {len(self.replay_buffer)}'
                )

                # reset the environments
                rewards = []
                episode_t = 0
                state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)

            # start training the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save the results
        self.save_results()

    def run_goal_dqn_map_guide_explore(self):
        """
               Function is used to train the globally goal-conditioned double DQN.
        """
        print("Experiment: Run goal-conditioned DQN")
        # set the training statistics
        rewards = []  # list of rewards for one episode
        episode_t = 0  # time step for one episode

        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # get action
            action = self.explore_using_map_info(state, goal, eps)

            # step in the environment
            next_state, reward, done, dist, trans, _, _, _, _ = self.env.step(action)
            # print(f'ep id={t}, s = {state}, a = {DEFAULT_ACTION_RAW[action]}, next_s={next_state}, g={goal}, dist={dist}, done={done}, reward={reward}')

            # store the replay buffer and convert the data to tensor
            if self.use_replay_buffer:
                # only use to explain the old state is infeasible to learn behavior
                # construct the transition
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                # add the transition into the buffer
                self.replay_buffer.add(trans)

            # increment
            state = next_state
            episode_t += 1
            rewards.append(reward)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                # compute the episode number
                episode_idx = len(self.returns)

                pbar.set_description(
                    f'Episode: {episode_idx} | '
                    f'Steps: {episode_t} | '
                    f'Return: {G:2f} | '
                    f'Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos_map} | '
                    f'Goal: {self.env.goal_pos_map} | '
                    f'Eps: {eps:.3f} | '
                    f'Buffer: {len(self.replay_buffer)}'
                )

                # reset the environments
                rewards = []
                episode_t = 0
                state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                # only used to show the old state is infeasible

            # start training the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save the results
        self.save_results()

    def run_goal_dqn_dual_replay(self):
        """
        Function is used to train the globally goal-conditioned double DQN.
        """
        print("Experiment: Run goal-conditioned DQN")
        # set the training statistics
        states = []
        actions = []
        rewards = []  # list of rewards for one episode
        episode_t = 0  # time step for one episode

        # update the start-goal positions
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)
        states.append(state)
        #goal = goal + [0]
        #state = state[0:2] + [0]
        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # get action
            action = self.agent.get_action(state, goal, eps)

            # step in the environment
            #action = 1
            next_state, reward, done, dist, trans, _, _, _, _ = self.env.step(action)
            # print(f'ep id={t}, s = {state}, a = {DEFAULT_ACTION_RAW[action]}, next_s={next_state}, g={goal}, dist={dist}, done={done}, reward={reward}')

            #Debug.set_trace()
            # store the replay buffer and convert the data to tensor
            if self.use_replay_buffer:
                # only use to explain the old state is infeasible to learn behavior
                #state = state[0:2] + [0]
                #next_state = next_state[0:2] + [0]
                # construct the transition
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                # add the transition into the buffer
                self.replay_buffer_explore.add(trans)

            # increment
            state = next_state
            episode_t += 1
            states.append(state)
            rewards.append(reward)
            actions.append(action)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                if done: 
                    for s_idx in range(len(states)-1):
                        done_flag = 1 if s_idx + 1 == len(states) else 0 
                        # good transitions
                        trans = self.toTransition(states[s_idx], actions[s_idx], states[s_idx + 1], rewards[s_idx], goal, done_flag)
                        # add the transition into the buffer
                        self.replay_buffer_exploit.add(trans)


                #Debug.set_trace()
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                self.steps.append(t)
                # compute the episode number
                episode_idx = len(self.returns)

                if (episode_idx - 1) % self.eval_policy_freq == 0:
                    # evaluate the current policy by interaction
                    #model_save_path = os.path.join(self.save_dir, self.model_name) + f"_{episode_idx}.pt"
                    #torch.save(self.agent.policy_net.state_dict(), model_save_path)
                    self.simple_policy_eval()  
                
                pbar.set_description(
                    f'Episode: {episode_idx} | '
                    f'Steps: {episode_t} | '
                    f'Return: {G:2f} | '
                    f'Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos_map} | '
                    f'Goal: {self.env.goal_pos_map} | '
                    f'Eps: {eps:.3f} | '
                    f'Explore Buffer: {len(self.replay_buffer_explore)} | '
                    f'Exploit Buffer: {len(self.replay_buffer_exploit)} | '
                    f'Eval: {self.policy_eval: .3f}'
                )

                # reset the environments
                rewards = []
                states = []
                actions = []
                episode_t = 0
                state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                # only used to show the old state is infeasible
                #state = state[0:2] + [0]
                #goal = goal + [0]
            # start training the agent
            if t > self.start_train_step:
                if random.uniform(0, 1) < eps:
                    sampled_batch = self.replay_buffer_explore.sample(self.batch_size)
                else:
                    sampled_batch = self.replay_buffer_exploit.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save the results
        self.save_results()



    def run_goal_dqn(self):
        """
        Function is used to train the globally goal-conditioned double DQN.
        """
        print("Experiment: Run goal-conditioned DQN")
        # set the training statistics
        rewards = []  # list of rewards for one episode
        episode_t = 0  # time step for one episode

        # update the start-goal positions
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)
        
        #goal = goal + [0]
        #state = state[0:2] + [0]
        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # get action
            action = self.agent.get_action(state, goal, eps)

            # step in the environment
            #action = 1
            next_state, reward, done, dist, trans, _, _, _, _ = self.env.step(action)
            # print(f'ep id={t}, s = {state}, a = {DEFAULT_ACTION_RAW[action]}, next_s={next_state}, g={goal}, dist={dist}, done={done}, reward={reward}')

            #Debug.set_trace()
            # store the replay buffer and convert the data to tensor
            if self.use_replay_buffer:
                # only use to explain the old state is infeasible to learn behavior
                #state = state[0:2] + [0]
                #next_state = next_state[0:2] + [0]
                # construct the transition
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                # add the transition into the buffer
                self.replay_buffer.add(trans)

            # increment
            state = next_state
            episode_t += 1
            rewards.append(reward)

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                #Debug.set_trace()
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                self.steps.append(t)
                # compute the episode number
                episode_idx = len(self.returns)

                if (episode_idx - 1) % self.eval_policy_freq == 0:
                    # evaluate the current policy by interaction
                    #model_save_path = os.path.join(self.save_dir, self.model_name) + f"_{episode_idx}.pt"
                    #torch.save(self.agent.policy_net.state_dict(), model_save_path)
                    self.simple_policy_eval()  
                
                pbar.set_description(
                    f'Episode: {episode_idx} | '
                    f'Steps: {episode_t} | '
                    f'Return: {G:2f} | '
                    f'Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos_map} | '
                    f'Goal: {self.env.goal_pos_map} | '
                    f'Eps: {eps:.3f} | '
                    f'Buffer: {len(self.replay_buffer)} | '
                    f'Eval: {self.policy_eval: .3f}'
                )

                # reset the environments
                rewards = []
                episode_t = 0
                state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                # only used to show the old state is infeasible
                #state = state[0:2] + [0]
                #goal = goal + [0]
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
        print("Experiment: Run random local goal-conditioned DQN")
        # set the training statistics
        rewards = []
        episode_t = 0  # time step for one episode
        sample_start_goal_num = self.sample_start_goal_num  # number of sampled start-goal pair per maze
        train_episode_num = self.train_episode_num  # training number for each start-goal pair

        # initialize the state and goal
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)
        fake_goal = None 
        real_goal = goal

        # start the training
        pbar = tqdm.trange(self.max_time_steps) 
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)
            
            # relabel the true goal observation with the fake goal with a fixed percentage 
            if self.use_imagine:
                if random.uniform(0, 1) <= self.use_imagine:
                    if fake_goal is None: # if the fake observation is NOT generated
                        loc_goal_map = self.env_map.cropper(self.env_map.map2d_roughPadded, goal_pos)
                        fake_goal = self.imagine_goal_observation(loc_goal_map)
                    # relabel the goal with the fake observation
                    goal = fake_goal
                else:
                    # relabel the goal with true observation
                    goal = real_goal

            # get action
            if not self.args.explore_use_map: 
                action = self.agent.get_action(state, goal, eps) 
            else:
                action = self.explore_using_map_info(state, goal, eps)   
            
            # step in the environment
            next_state, reward, done, dist, state_trans, _, _, _, _ = self.env.step(action)
            #print(f'ep id={t}, s = {state}, a = {DEFAULT_ACTION_RAW[action]}, next_s={next_state}, g={goal}, dist={dist}, done={done}, reward={reward}')  # for debug

            # store the replay buffer and convert the data to tensor
            if self.use_replay_buffer:
                # construct the transition
                trans = self.toTransition(state, action, next_state, reward, goal, done)
                # add the transition into the buffer
                self.replay_buffer.add(trans)

            # update the state
            state = next_state
            rewards.append(reward)
            episode_t += 1

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                self.steps.append(t)
                episode_idx = len(self.returns)
                
                pbar.set_description(
                    f'Episode: {episode_idx}|'
                    f'Steps: {episode_t}|'
                    f'Return: {G:.2f}|'
                    f'Dist: {dist:.2f}|'
                    f'Init: {self.env.start_pos_map[0:2]}|'
                    f'Goal: {self.env.goal_pos_map[0:2]}|'
                    f'Eps: {eps:.3f}|'
                    f'Buffer: {len(self.replay_buffer)}|'
                    f'Eval: {self.policy_eval:.2f}'
                ) 
                # # evaluate the current policy
                if (episode_idx - 1) % self.eval_policy_freq == 0:
                     # evaluate the current policy by interaction
                     model_save_path = os.path.join(self.save_dir, self.model_name) + f"_{episode_idx}.pt"
                     torch.save(self.agent.policy_net.state_dict(), model_save_path)
                     self.eval_policy_novel()

                # reset the environments
                rewards = []
                episode_t = 0
                fake_goal = None
                real_goal = None
                # train a pair of start and goal with fixed number of episodes 
                if sample_start_goal_num > 0:
                    if train_episode_num > 0:
                        # keep the same start and goal
                        self.fix_start = True
                        self.fix_goal = True
                        self.env.sample_repeat_count -= 1 
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        train_episode_num -= 1
                    else:
                        # sample a new pair of start and goal
                        self.fix_start = False
                        self.fix_goal = False
                        # sample a valid distance 
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        train_episode_num = self.train_episode_num
                        sample_start_goal_num -= 1 
                else:
                    # sample a new maze
                    self.fix_start = False
                    self.fix_goal = False
                    # sample a valid distance 
                    state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)
                    # reset the training control
                    train_episode_num = self.train_episode_num
                    sample_start_goal_num = self.sample_start_goal_num
                real_goal = goal
            # train the agent
            if t > self.start_train_step: 
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)
        # save results
        self.save_results()

    def run_random_local_goal_dqn_her_our(self):
        """
        Function is used to train the locally goal-conditioned double DQN.
        """
        # set the training statistics
        print("Variant 1: Run random goal-conditioned DQN with HER")
        states = []
        actions = []
        rewards = []
        trans_poses = []
        dones = []
        episode_t = 0  # time step for one episode
        sample_start_goal_num = self.sample_start_goal_num  # sampled start and goal pair
        train_episode_num = self.train_episode_num  # training number for each start-goal pair
        # initialize the state and goal
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)
        states.append(state)

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # relabel the true goal observation with the fake goal with a fixed precentage
            if self.use_imagine:
                if random.uniform(0, 1) <= self.use_imagine:
                    loc_goal_map = self.env_map.cropper(self.env_map.map2d_roughPadded, goal_pos)
                    goal = self.imagine_goal_observation(loc_goal_map)

            # get action
            action = self.agent.get_action(state, goal, eps)

            # step in the environment
            next_state, reward, done, dist, trans, _, _, _, _ = self.env.step(action)
            reward = -1

            # add terminal estimation 
            if done and self.use_imagine:
                terminal_act = random.sample(range(4), 1)[0]
                terminal_loc_goal_map = self.env_map.cropper(self.env_map.map2d_roughPadded, goal_pos)
                terminal_goal = self.imagine_goal_observation(terminal_loc_goal_map)
                terminal_trans = self.toTransition(next_state, terminal_act, terminal_goal, 0, terminal_goal, terminal_goal, done)
                self.replay_buffer.add(terminal_trans)

            # save the transitions
            episode_t += 1
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            trans_poses.append(trans)
            dones.append(done)

            # update the current state
            state = next_state

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

                # construct the memory buffer using HER
                self.hindsight_experience_replay(states, actions, rewards, trans_poses, goal, dones)

                pbar.set_description(
                    f'Episode: {episode_idx}|Steps: {episode_t}|Return: {G:.2f}|Dist: {dist:.2f}|'
                    f'Init: {self.env.start_pos[0:2]}|Goal: {self.env.goal_pos[0:2]}|'
                    f'Eps: {eps:.3f}|'
                    f'GT: {len(self.env_map.path) - 1}|'
                    f'Buffer: {len(self.replay_buffer)}|'
                    f'Loss: {self.agent.current_total_loss:.4f}|'
                    f'Pred Loss: {self.agent.current_state_loss:.4f}'
                )

                # # evaluate the current policy
                # if (episode_idx - 1) % self.eval_policy_freq == 0:
                #     # evaluate the current policy by interaction
                #     model_save_path = os.path.join(self.save_dir, self.model_name) + f"_{episode_idx}.pt"
                #     torch.save(self.agent.policy_net.state_dict(), model_save_path)
                #     self.eval_policy()

                # reset the environments
                states = []
                actions = []
                trans_poses = []
                rewards = []
                dones = []
                episode_t = 0
                # train a pair of start and goal with fixed number of episodes
                if sample_start_goal_num > 0:
                    if train_episode_num > 0:
                        # keep the same start and goal
                        self.fix_start = True
                        self.fix_goal = True
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        train_episode_num -= 1
                    else:
                        # sample a new pair of start and goal
                        self.fix_start = False
                        self.fix_goal = False
                        self.goal_dist = random.sample(self.valid_goal_dist, 1)[0]
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        #print(start_pos, goal_pos, ' - ', len(self.env_map.path))
                        train_episode_num = self.train_episode_num
                        sample_start_goal_num -= 1
                else:
                    # sample a new maze
                    self.fix_start = False
                    self.fix_goal = False
                    self.goal_dist = random.sample(self.valid_goal_dist, 1)[0]
                    #print(start_pos, goal_pos, ' - ', len(self.env_map.path))
                    state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)
                    # reset
                    train_episode_num = self.train_episode_num
                    sample_start_goal_num = self.sample_start_goal_num
                states.append(state)

            # train the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save results
        self.save_results()

    def run_random_goal_dqn_her(self):
        """
           Function is used to train the locally goal-conditioned double DQN.
        """
        # set the training statistics
        print("Baseline 1: Goal-conditioned DQN with HER")
        # define an episode
        states_ep = []
        actions_ep = []
        rewards_ep = []
        trans_ep = []
        done_ep = []
        episode_t = 0  # time step for one episode
        sample_pair_num_per_maze = self.sample_start_goal_num  # sampled start and goal pair
        train_episode_num_per_pair = self.train_episode_num  # training number for each start-goal pair

        # initialize the state and goal
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)
        states_ep.append(state)

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # get action
            action = self.agent.get_action(state, goal, eps)

            # step in the environment
            next_state, reward, done, dist, trans, _, _, _, _ = self.env.step(action)

            # save the transitions
            states_ep.append(next_state)
            actions_ep.append(action)
            rewards_ep.append(reward)
            trans_ep.append(trans)
            done_ep.append(done)

            # increase
            state = next_state
            episode_t += 1

            # check terminal
            if done or (episode_t == self.max_steps_per_episode):
                # compute the discounted return for each time step
                G = 0
                for r in reversed(rewards_ep):
                    G = r + self.gamma * G

                # store the return, episode length, and final distance for current episode
                self.returns.append(G)
                episode_idx = len(self.returns)

                # construct the memory buffer using HER
                self.hindsight_experience_replay(states_ep, actions_ep, rewards_ep, trans_ep, done_ep, goal)

                # plot information
                pbar.set_description(
                    f'Episode: {episode_idx} | '
                    f'Steps: {episode_t} | '
                    f'Return: {G:2f} | '
                    f'Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos_map[0:2]} | '
                    f'Goal: {self.env.goal_pos_map[0:2]} | '
                    f'Eps: {eps:.3f} | '
                    f'Buffer: {len(self.replay_buffer)}'
                )

                # reset the episode parameters
                states_ep = []
                actions_ep = []
                rewards_ep = []
                trans_ep = []
                done_ep = []
                episode_t = 0

                # reset the environment
                if sample_pair_num_per_maze > 0:
                    if train_episode_num_per_pair > 0:
                        # keep the same start and goal
                        self.fix_start = True
                        self.fix_goal = True
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        train_episode_num_per_pair -= 1
                    else:
                        # sample a new pair of start and goal
                        self.fix_start = False
                        self.fix_goal = False
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        train_episode_num_per_pair = self.train_episode_num
                        sample_pair_num_per_maze -= 1
                else:
                    # sample a new maze
                    self.fix_start = False
                    self.fix_goal = False
                    state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)
                    train_episode_num_per_pair = self.train_episode_num
                    sample_pair_num_per_maze = self.sample_start_goal_num
                states_ep.append(state)

            # train the agent
            if t > self.start_train_step:
                sampled_batch = self.replay_buffer.sample(self.batch_size)
                self.agent.train_one_batch(t, sampled_batch)

        # save results
        self.save_results() 

    # generate the goal imagination
    def imagine_goal_observation(self, pos_loc_map):
        #time_start = time.time()
        with torch.no_grad():
            imagined_obs = []
            loc_map = torch.from_numpy(pos_loc_map).flatten().view(1, -1).float()
            for ori in self.orientations:
                # sample the z
                z = torch.randn(1, 64)
                # concatenate the conditioned z
                tmp_map = torch.cat(2 * [loc_map], dim=1)
                tmp_ori = torch.cat(2 * [ori.view(-1, 1 * 1 * 8).float()], dim=1)
                conditioned_z = torch.cat((z, tmp_map, tmp_ori), dim=1)
                # feed forward
                obs_reconstructed, _ = self.thinker.decoder(conditioned_z)
                # forward
                obs_reconstructed = obs_reconstructed.detach().squeeze(0).numpy().transpose(1, 2, 0) * 255
                imagined_obs.append(obs_reconstructed)
                #print("Reshape obs: ", time.time() - time_start)
            #print("Time get panoramic obs: ", time.time() - time_start)

            # the return might be slow. How to update this function?!!!!
            return np.array(imagined_obs, dtype=np.uint8)

    # HER
    def hindsight_experience_replay(self, s_list, a_list, r_list, t_list, d_list, g):
        # compute the episode length
        episode_len = len(s_list) - 1
        for t in range(episode_len):
            # extract one transition
            state = s_list[t]  # s_t
            action = a_list[t]  # a_t
            reward = r_list[t]  # r_t
            next_state = s_list[t + 1]  # s_t+1
            next_state_pos = t_list[t]  # position of the current next state
            done = d_list[t]  # done_t

            # normal experience replay buffer
            transition_normal = self.toTransition(state, action, next_state, reward, g, done)
            self.replay_buffer.add(transition_normal)

            # Hindsight Experience Replay
            future_indices = list(range(t+1, episode_len, 1))
            # sampled future goals
            sampled_goals = random.sample(future_indices, self.her_future_k) if len(
                future_indices) > self.her_future_k else future_indices
            # relabel the current transition
            for idx in sampled_goals:
                # get the new goal
                new_goal = s_list[idx][0:2]
                new_goal_pos = t_list[idx - 1]
                # compute the new reward
                distance = self.env.compute_distance(next_state_pos, new_goal_pos)
                new_reward = self.env.compute_reward(distance)
                new_done = 0 if new_reward == -1 else 1
                # HER experience replay buffer
                transition_her = self.toTransition(state, action, next_state, new_reward, new_goal, new_done)
                self.replay_buffer.add(transition_her)

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
            if not self.train_random_policy:
                self.env_map.sample_start_goal_pos_with_random_dist(self.fix_start, self.fix_goal)
            else:
                self.env_map.sample_start_goal_pos_with_maximal_dist(self.fix_start, self.fix_goal, self.goal_dist)
            #init_pos = [1, 1]
            #goal_pos = [3, 3]
            #self.env_map.update_mapper(init_pos, goal_pos)
            # initialize the maze 3D
            maze_configs["maze_name"] = f"maze_{self.maze_size}_{self.maze_seed}"  # string type name
            maze_configs["maze_size"] = [self.maze_size, self.maze_size]  # [int, int] list
            maze_configs["maze_seed"] = '1234'  # string type number
            maze_configs["maze_texture"] = random.sample(self.theme_list, 1)[0]  # string type name in theme_list
            maze_configs["maze_decal_freq"] = random.sample(self.decal_list, 1)[0]  # float number in decal_list
            maze_configs["maze_map_txt"] = "".join(self.env_map.map2d_txt)  # string type map
            maze_configs["maze_valid_pos"] = self.env_map.valid_pos  # list of valid positions
            # initialize the maze start and goal positions
            maze_configs["start_pos"] = self.env_map.init_pos + [0]
            maze_configs["goal_pos"] = self.env_map.goal_pos + [0]
            # initialize the update flag
            maze_configs["update"] = True  # update flag
        else:
            if not self.train_random_policy:
                self.env_map.sample_start_goal_pos_with_random_dist(self.fix_start, self.fix_goal)
            else:
                self.env_map.sample_start_goal_pos_with_maximal_dist(self.fix_start, self.fix_goal, self.goal_dist)
            #init_pos = [1, 1]
            #goal_pos = [3, 3]
            #self.env_map.update_mapper(init_pos, goal_pos)
            maze_configs['start_pos'] = self.env_map.init_pos + [0]
            maze_configs['goal_pos'] = self.env_map.goal_pos + [0]
            maze_configs['maze_valid_pos'] = self.env_map.valid_pos
            maze_configs['update'] = False
 
        # obtain the state and goal observation
        state_obs, goal_obs, _, _, _, _ = self.env.reset(maze_configs)

        return state_obs, goal_obs, self.env_map.init_pos, self.env_map.goal_pos

    def eval_policy_novel(self):
        maze_configs = defaultdict(lambda: None)
        # loop all the testing mazes
        for m_size in self.maze_size_list:
            for m_seed in self.maze_seed_list:
                # print the current maze info 
                self.eval_dist_pairs = self.load_pair_data(m_size, m_seed)
                self.maze_size = m_size
                self.maze_seed = m_seed
                self.env_map_test = mapper.RoughMap(m_size, m_seed, 3)
                
                # load the maze parameters
                maze_configs["maze_name"] = f"maze_{m_size}_{m_seed}"  # string type name
                maze_configs["maze_size"] = [m_size, m_size]  # [int, int] list
                maze_configs["maze_seed"] = '1234'  # string type number
                maze_configs["maze_texture"] = random.sample(self.theme_list, 1)[0]  # string type name in theme_list
                maze_configs["maze_decal_freq"] = random.sample(self.decal_list, 1)[0]  # float number in decal_list
                maze_configs["maze_map_txt"] = "".join(self.env_map_test.map2d_txt)  # string type map
                maze_configs['start_pos'] = self.env_map_test.init_pos + [0]
                maze_configs['goal_pos'] = self.env_map_test.goal_pos + [0]
                maze_configs['maze_valid_pos'] = self.env_map_test.valid_pos
                maze_configs['update'] = True
                
                # initialize the test domain
                self.env_test.reset(maze_configs)
                self.env_test.sample_repeat_count = 0
                self.env_test.start_radius = 30

                # loop all the distance
                pairs_dict = {'start': self.eval_dist_pairs['1'][0], 'goal': self.eval_dist_pairs['1'][1]}
                # sample number
                eval_total_num = 20 if len(pairs_dict['start']) > 20 else len(pairs_dict['start'])
                eval_success_num = 0
                # obtain all the pairs
                pairs_idx = random.sample(range(len(pairs_dict['start'])), eval_total_num)
                # loop all the pairs
                for r, idx in enumerate(pairs_idx):
                    # obtain the start-goal pair
                    s_pos, g_pos = pairs_dict['start'][idx], pairs_dict['goal'][idx]
                    
                    # update the start and goal positions
                    self.env_map_test.update_mapper(s_pos, g_pos)
                    maze_configs['start_pos'] = self.env_map_test.init_pos + [0]
                    maze_configs['goal_pos'] = self.env_map_test.goal_pos + [0]
                    maze_configs['maze_valid_pos'] = self.env_map_test.valid_pos
                    maze_configs['update'] = False
                    state, goal, _, _, _, _ = self.env_test.reset(maze_configs)
                    
                    # obtain the fake observation
                    if not self.use_true_state:
                        goal_loc_map = self.env_map_test.cropper(self.env_map_test.map2d_roughPadded, self.env_map_test.path[-1])
                        goal = self.imagine_goal_observation(goal_loc_map)
                    
                    # start navigation
                    max_time_steps = 20
                    act_list = []
                    for t in range(max_time_steps):
                        # get action
                        action = self.agent.get_action(state, goal, 0)
                        act_list.append(DEFAULT_ACTION_LIST[action])
                        # step in the environment
                        next_state, reward, done, dist, next_trans, _, _, _, _ = self.env_test.step(action)
                        if done:
                            eval_success_num += 1
                            break
                        else:
                            state = next_state
                self.policy_eval = eval_success_num / eval_total_num
                self.policy_returns.append(eval_success_num / eval_total_num)
    
    def simple_policy_eval(self):
        self.maze_size = 5
        self.maze_seed = 0
        start_pos = [1, 2]
        goal_pos = [2, 1]
        # create the test environment
        self.env_map_test = mapper.RoughMap(self.maze_size, self.maze_seed, 3)
        self.env_map_test.update_mapper(start_pos, goal_pos)
        # initialize the test environment
        maze_configs = defaultdict(lambda: None)
        # set the maze configurations
        maze_configs["maze_name"] = f"maze_{self.maze_size}_{self.maze_seed}"  # string type name
        maze_configs["maze_size"] = [self.maze_size, self.maze_size]  # [int, int] list
        maze_configs["maze_seed"] = '1234'  # string type number
        maze_configs["maze_texture"] = random.sample(self.theme_list, 1)[0]  # string type name in theme_list
        maze_configs["maze_decal_freq"] = random.sample(self.decal_list, 1)[0]  # float number in decal_list
        maze_configs["maze_map_txt"] = "".join(self.env_map_test.map2d_txt)  # string type map

        maze_configs['start_pos'] = self.env_map_test.init_pos + [0]
        maze_configs['goal_pos'] = self.env_map_test.goal_pos + [0]
        maze_configs['maze_valid_pos'] = self.env_map_test.valid_pos
        maze_configs['update'] = True
        # obtain the state and goal observation
        state, goal, _, _, _, _ = self.env_test.reset(maze_configs) 
        # test
        test_num = 10
        success_num = 0
        for i in range(test_num):
            max_time_steps = 10
            act_list = []
            for t in range(max_time_steps):
                # get action
                action = self.agent.get_action(state, goal, 0)
                act_list.append(DEFAULT_ACTION_RAW[action])
                # step in the environment
                next_state, reward, done, dist, next_trans, _, _, _, _ = self.env_test.step(action)
                if done:
                    act_list.append('1')
                    success_num += 1
                    maze_configs['update'] = False
                    state, goal, _, _, _, _ = self.env_test.reset(maze_configs)
                    break
                else:
                    state = next_state
            print(act_list)
            act_list = []
        self.policy_eval = success_num / test_num
        self.policy_returns.append(self.policy_eval)

    # save the results
    def save_results(self):
        # obtain the saving names
        model_save_path = os.path.join(self.save_dir, self.model_name) + ".pt"
        returns_save_path = os.path.join(self.save_dir, self.model_name + "_return.npy")
        steps_save_path = os.path.join(self.save_dir, self.model_name + "_steps.npy")
        policy_returns_save_path = os.path.join(self.save_dir, self.model_name + "_policy_eval.npy")

        # save the results
        torch.save(self.agent.policy_net.state_dict(), model_save_path)
        np.save(returns_save_path, self.returns)
        np.save(steps_save_path, self.steps)
        np.save(policy_returns_save_path, self.policy_returns)

    # load the pre-extract pairs
    @staticmethod
    def load_pair_data(m_size, m_seed):
        #path = f'/mnt/cheng_results/map/maze_{m_size}_{m_seed}.pkl'
        path = f'/mnt/sda/map/maze_{m_size}_{m_seed}.pkl'
        #path = f'./map/maze_{m_size}_{m_size}.pkl'
        f = open(path, 'rb')
        return pickle.load(f)

    def update_maze_from_pos(self, start_pos, goal_pos):
        maze_configs = defaultdict(lambda: None)
        # print(f"Set start = {start_pos}, goal = {goal_pos}")
        self.env_map.update_mapper(start_pos, goal_pos)
        # set the maze configurations
        maze_configs['start_pos'] = self.env_map.init_pos + [0]
        maze_configs['goal_pos'] = self.env_map.goal_pos + [0]
        maze_configs['maze_valid_pos'] = self.env_map.valid_pos
        maze_configs['update'] = False
        # obtain the state and goal observation
        state_obs, goal_obs, _, _, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs, self.env_map.init_pos, self.env_map.goal_pos

    def explore_using_map_info(self, state, goal, eps):
        if random.uniform(0, 1) < eps:  # with probability epsilon, the agent selects a random action
            action = DEFAULT_ACTION_RAW.index(self.env_map.map_act[0])
        else:  # with probability 1 - epsilon, the agent selects the greedy action
            action = self.agent.get_action(state, goal, 0)
        return action
