from utils import mapper
import numpy as np
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


# define the global default parameters
DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "next_state", "reward", "goal", "done"])
DEFAULT_ACTION_LIST = ['up', 'down', 'left', 'right']
RAW_ACTION_LIST = ['left', 'right', 'forward', 'backward']

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
                 use_cycle_relabel=False,  # whether use cycle relabeling strategy
                 use_state_est=False
                 ):
        self.device = torch.device(device)
        # environment
        self.env = env
        self.env_map = None
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
        # for cycle relabel
        self.reverse_action = [1, 0, 3, 2]
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
            self.thinker.load_state_dict(torch.load("/mnt/sda/dataset/ml_nav/VAE/model/small_obs_L64_B8.pt",
                                                     map_location=self.device))
            self.thinker.eval()
        # training configurations
        self.train_local_policy = train_local_policy
        self.sample_start_goal_num = sample_start_goal_num
        self.train_episode_num = train_episode_num
        self.start_train_step = start_train_step
        self.max_time_steps = max_time_steps
        self.max_steps_per_episode = episode_time_steps
        self.eval_policy_freq = eval_policy_freq
        self.use_cycle_relabel = use_cycle_relabel
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
        self.policy_eval = 0
        self.distance = []
        self.returns = []
        self.lengths = []
        self.policy_returns = []
        # saving settings
        self.model_name = model_name
        self.save_dir = save_dir

    def run_maze_complexity_comparison(self):
        """
            Function is used to run maze complexity analysis
        """
        print("Experiment: Analyzing maze complexity.")
        # testing pairs
        self.maze_size = random.sample(self.maze_size_list, 1)[0]
        test_pairs = [[[1, 1], [self.maze_size - 2, self.maze_size - 2]],  # top left - bottom right
                      [[1, self.maze_size - 2], [self.maze_size - 2, 1]]]  # top right - bottom left
                      #[[1, 1], [1, self.maze_size - 2]],  # top left - top right
                      #[[1, 1], [self.maze_size - 2, 1]],  # top left - bottom left
                      #[[1, self.maze_size - 2], [self.maze_size - 2, self.maze_size - 2]],  # top right - bottom right
                      #[[self.maze_size - 2, 1], [self.maze_size - 2, self.maze_size - 2]]]  # bottom left - bottom right
        # test_pairs = [[[1, 1], [self.maze_size - 2, self.maze_size - 2]]]
        # maze seed
        maze_seed_list = self.maze_seed_list
   
        complex_mean_success = []
        complex_mean_length = []
        for seed in maze_seed_list:
            # initialize the map 2D
            maze_configs = defaultdict(lambda: None)
            self.env_map = mapper.RoughMap(self.maze_size, seed, 3)
            print(f"****** Maze {self.maze_size} - {seed} ******")
            mean_success_rate = 0
            mean_len = 0
            for pair in test_pairs:
                # update the start and goal position
                self.env_map.update_mapper(pair[0], pair[1])
                # initialize the maze 3D
                maze_configs["maze_name"] = f"maze_{self.maze_size}_{self.maze_seed}"  # string type name
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
                # reset the environment
                state, goal, _, _ = self.env.reset(maze_configs)

                # random policy running number
                run_num = 100
                success_count = 0
                length_count = []
                episode_len = 0
                # start estimation
                # print(f"-- Start = {self.env_map.init_pos} - Goal = {self.env_map.goal_pos}")
                last_trans = state
                for r in range(run_num): 
                    # navigation using random policy
                    for t in range(self.max_steps_per_episode):
                        # get action
                        action = random.sample(range(4), 1)[0]
                        # step in the environment
                        next_state, reward, done, dist, trans, _, _ = self.env.step(action)
                        episode_len += 1
                        # print(f"Step={t}: state={last_trans}, action={DEFAULT_ACTION_LIST[action]}, next_state={next_state}, done={done}")
                        if done:
                            success_count += 1
                            break
                        last_trans = next_state
                    
                    # save the episode length
                    length_count.append(episode_len)
                    # reset the episode
                    episode_len = 0
                    # switch the start and goal position after 50 runs
                    if r > (run_num / 2) - 2:
                        self.env_map.update_mapper(pair[1], pair[0])
                    maze_configs["start_pos"] = self.env_map.init_pos + [0]
                    maze_configs["goal_pos"] = self.env_map.goal_pos + [0]
                    # initialize the update flag
                    maze_configs["update"] = False  # update flag
                    # reset the environment
                    self.env.reset(maze_configs)
                mean_success_rate += success_count / run_num
                mean_len += sum(length_count) / len(length_count)
            complex_mean_success.append(mean_success_rate / 6)
            complex_mean_length.append(mean_len / 6)
            print(f"-- Mean traverse success rate = {mean_success_rate / 6}")
            print(f"-- Mean navigation length = {mean_len / 6}")
            print("-----------------------------------------")
       
        save_path = '/'.join([self.save_dir, f"complex_analysis_diagnoal_{self.maze_size}x{self.maze_size}_ep{self.max_steps_per_episode}.txt"])
        counter = 0
        with open(save_path, 'w') as f:
            for s_val, l_val in zip(complex_mean_success, complex_mean_length):
                tmp_str = ','.join([str(counter), str(s_val), str(l_val)]) + '\n'
                counter += 1
                f.write(tmp_str)
            f.close()

    def run_dqn(self):
        """
        Function is used to train the vanilla double DQN agent.
        """
        print("Experiment: Run double DQN.")
        # set the training statistics
        rewards = []  # list of rewards for one episode
        episode_t = 0  # time step counter for one episode

        # initialize the start and goal positions
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)

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

                # reset the environments
                rewards = []
                episode_t = 0
                state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
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
        print("Experiment: Run goal-conditioned DQN")
        # set the training statistics
        rewards = []  # list of rewards for one episode
        episode_t = 0  # time step for one episode

        # update the start-goal positions
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)

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

                # reset the environments
                rewards = []
                episode_t = 0
                state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
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
        print("Experiment: Run random local goal-conditioned DQN")
        # set the training statistics
        rewards = []
        episode_t = 0  # time step for one episode
        sample_start_goal_num = self.sample_start_goal_num
        train_episode_num = self.train_episode_num  # training number for each start-goal pair

        # initialize the state and goal
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=self.fix_maze)
        init_state = state

        # start the training
        pbar = tqdm.trange(self.max_time_steps)
        for t in pbar:
            # compute the epsilon
            eps = self.schedule.get_value(t)

            # relabel the true goal observation with the fake goal with a fixed percentage
            if self.use_imagine:
                if random.uniform(0, 1) <= self.use_imagine:
                    loc_goal_map = self.env_map.cropper(self.env_map.map2d_roughPadded, goal_pos)
                    goal = self.imagine_goal_observation(loc_goal_map)
  
            # get action
            action = self.agent.get_action(state, goal, eps)

            # step in the environment 
            next_state, reward, done, dist, trans, _, t_vel, r_vel, _ = self.env.step(action)
            episode_t += 1
            #print(f'step={t}: state={state}, action={RAW_ACTION_LIST[action]}, next_state={next_state}, reward={reward}, done={done}, goal={goal}, distance={dist}, {np.array(next_state) - np.array(state)}')
            
            #Debug.set_trace()
            # store the replay buffer and convert the data to tensor
            if self.use_replay_buffer:
                # construct the transition
                trans = self.toTransition(state, action, next_state, reward, init_state, goal, done)
                # add the transition into the buffer
                self.replay_buffer.add(trans)

            # update the state
            state = next_state
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
                self.lengths.append(episode_t)
                self.distance.append(dist)
                # compute the episode number
                episode_idx = len(self.returns)
                
                pbar.set_description(
                    f'Episode: {episode_idx}|Steps: {episode_t}|Return: {G:.2f}|Dist: {dist:.2f}|'
                    f'Init: {self.env.start_pos_map[0:2]}|Goal: {self.env.goal_pos_map[0:2]}|'
                    f'Eps: {eps:.3f}|'
                    f'GT: {len(self.env_map.path) - 1}|'
                    f'Buffer: {len(self.replay_buffer)}|'
                    f'Loss: {self.agent.current_total_loss:.4f}|'
                    f'Policy Eval: {self.policy_eval:.2f}'
                ) 
                # evaluate the current policy
                if (episode_idx - 1) % self.eval_policy_freq == 0:
                    # evaluate the current policy by interaction
                    model_save_path = os.path.join(self.save_dir, self.model_name) + f"_{episode_idx}.pt"
                    torch.save(self.agent.policy_net.state_dict(), model_save_path)
                    self.eval_policy_novel() 

                # reset the environments
                rewards = []
                episode_t = 0
                # train a pair of start and goal with fixed number of episodes
                if sample_start_goal_num > 0:
                    if train_episode_num > 0:
                        # keep the same start and goal
                        self.fix_start = True
                        self.fix_goal = True
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        init_state = state
                        train_episode_num -= 1
                    else:
                        # sample a new pair of start and goal
                        self.fix_start = False
                        self.fix_goal = False
                        # sample a valid distance 
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        init_state = state
                        train_episode_num = self.train_episode_num
                        sample_start_goal_num -= 1
                else:
                    # sample a new maze
                    self.fix_start = False
                    self.fix_goal = False
                    # sample a valid distance 
                    state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=True)
                    init_state = state
                    # reset the training control
                    train_episode_num = self.train_episode_num
                    sample_start_goal_num = self.sample_start_goal_num

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
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)
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
                    #f'Pred Loss: {self.agent.current_state_loss:.4f}|'
                    f'Policy Eval: {self.policy_eval:.2f}'
                )

                # evaluate the current policy
                if (episode_idx - 1) % self.eval_policy_freq == 0:
                    # evaluate the current policy by interaction
                    model_save_path = os.path.join(self.save_dir, self.model_name) + f"_{episode_idx}.pt"
                    torch.save(self.agent.policy_net.state_dict(), model_save_path)
                    self.eval_policy()

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

    def run_random_local_goal_dqn_her(self):
        """
        Function is used to train the locally goal-conditioned double DQN.
        """
        # set the training statistics
        print("Baseline 1: Goal-conditioned DQN with HER")
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

            # get action
            action = self.agent.get_action(state, goal, eps)

            # step in the environment

            next_state, reward, done, dist, trans, _, _ = self.env.step(action)            

            # save the transitions
            episode_t += 1
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            trans_poses.append(trans)
            dones.append(done)
            
            # increase
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
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos} | Goal: {self.env.goal_pos} | '
                    f'Eps: {eps:.3f} | Buffer: {len(self.replay_buffer)}'
                )

                # evaluate the current policy
                if (episode_idx - 1) % self.eval_policy_freq == 0:
                    # evaluate the current policy by interaction
                    model_save_path = os.path.join(self.save_dir, self.model_name) + f"_{episode_idx}.pt"
                    torch.save(self.agent.policy_net.state_dict(), model_save_path)
                    self.eval_policy()
   
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
                        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=False)
                        train_episode_num = self.train_episode_num
                        sample_start_goal_num -= 1
                else:
                    # sample a new maze
                    self.fix_start = False
                    self.fix_goal = False
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

    
    def cycle_relabel(self, batch):
        reverse_action = [torch.tensor(self.reverse_action[act.item()], dtype=torch.int8).view(-1, 1) for act in batch.action]
        reverse_reward = []
        reverse_done = []
        for s, i, a in zip(batch.state, batch.init, reverse_action):
            if torch.all(torch.eq(s, i)):
                reverse_reward.append(torch.tensor(0, dtype=torch.int8).view(-1, 1))
                reverse_done.append(torch.tensor(1, dtype=torch.int8).view(-1, 1))
            else:
                reverse_reward.append(torch.tensor(-1, dtype=torch.int8).view(-1, 1))
                reverse_done.append(torch.tensor(0, dtype=torch.int8).view(-1, 1))

        relabeled_batch = self.TRANSITION(state=tuple(list(batch.state) + list(batch.next_state)),
                                          action=tuple(list(batch.action) + reverse_action),
                                          reward=tuple(list(batch.reward) + reverse_reward),
                                          next_state=tuple(list(batch.next_state) + list(batch.state)),
                                          init=tuple(list(batch.init) + list(batch.goal)),
                                          goal=tuple(list(batch.goal) + list(batch.init)),
                                          done=tuple(list(batch.done) + reverse_done)
                                          )
        return relabeled_batch

    # generate the goal imagination
    def imagine_goal_observation(self, pos_loc_map):
        with torch.no_grad():
            imagined_obs = []
            loc_map = torch.from_numpy(pos_loc_map).flatten().view(1, -1).float()
            for ori in self.orientations:
                z = torch.randn(1, 64)
                tmp_map = torch.cat(2 * [loc_map], dim=1)
                tmp_ori = torch.cat(2 * [ori.view(-1, 1 * 1 * 8).float()], dim=1)
                conditioned_z = torch.cat((z, tmp_map, tmp_ori), dim=1)
                obs_reconstructed, _ = self.thinker.decoder(conditioned_z)
                obs_reconstructed = obs_reconstructed.detach().squeeze(0).numpy().transpose(1, 2, 0) * 255
                imagined_obs.append(obs_reconstructed)
            return np.array(imagined_obs, dtype=np.uint8)

    # adding transition to the buffer using HER
    def hindsight_experience_replay(self, states, actions, rewards, trans_poses, goal, dones):
        states_num = len(states) - 1
        for t in range(states_num):
            # extract one transition
            state = states[t]  # s_t
            action = actions[t]  # a_t
            reward = rewards[t]  # r_t
            next_state = states[t + 1]  # s_t+1
            next_state_pos = trans_poses[t]  # position of the current next state
            done = dones[t]  # done_t
            # normal replay buffer
            transition = self.toTransition(state, action, next_state, reward, goal, goal, done)  # add the current transition
            self.replay_buffer.add(transition)
            # Hindsight Experience Replay
            future_indices = list(range(t+1, len(states)))
            # sampled future goals
            sampled_goals = random.sample(future_indices, self.her_future_k) if len(
                future_indices) >= self.her_future_k else future_indices
            # relabel the current transition
            for idx in sampled_goals:
                new_goal = states[idx]
                new_goal_pos = trans_poses[idx - 1]
                distance = self.env.compute_distance(next_state_pos, new_goal_pos)
                new_reward = self.env.compute_reward(distance)
                new_done = 0 if new_reward == -1 else 1
                new_reward = -1
                transition = self.toTransition(state, action, next_state, new_reward, new_goal, new_goal, new_done)
                self.replay_buffer.add(transition)

    def toTransition(self, state, action, next_state, reward, init, goal, done):
        """
        Function is used to construct a transition using state, action, next_state, reward, goal, done.
        """
        if not self.use_goal:  # construct non goal-conditioned transition (Default type is int8 to save memory)
            return self.TRANSITION(state=self.toTensor(state),
                                   action=torch.tensor(action, dtype=torch.int8).view(-1, 1),
                                   reward=torch.tensor(reward, dtype=torch.int8).view(-1, 1),
                                   next_state=self.toTensor(next_state),
                                   done=torch.tensor(done, dtype=torch.int8).view(-1, 1))
        elif not self.use_cycle_relabel:  # construct goal-conditioned transition
            return self.TRANSITION(state=self.toTensor(state),
                                   action=torch.tensor(action, dtype=torch.int8).view(-1, 1),
                                   reward=torch.tensor(reward, dtype=torch.int8).view(-1, 1),
                                   next_state=self.toTensor(next_state),
                                   goal=self.toTensor(goal),
                                   done=torch.tensor(done, dtype=torch.int8).view(-1, 1))
        else:
            return self.TRANSITION(state=self.toTensor(state),
                                   action=torch.tensor(action, dtype=torch.int8).view(-1, 1),
                                   reward=torch.tensor(reward, dtype=torch.int8).view(-1, 1),
                                   next_state=self.toTensor(next_state),
                                   init=self.toTensor(init),
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
            self.env_map.sample_random_start_goal_pos(self.fix_start, self.fix_goal, self.goal_dist)
            init_pos = self.env_map.init_pos
            goal_pos = self.env_map.goal_pos
            self.env_map.update_mapper(init_pos, goal_pos)
            # initialize the maze 3D
            maze_configs["maze_name"] = f"maze_{self.maze_size}_{self.maze_seed}"  # string type name
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
            self.env_map.update_mapper(init_pos, goal_pos)
            maze_configs['start_pos'] = init_pos + [0]
            maze_configs['goal_pos'] = goal_pos + [0]
            maze_configs['maze_valid_pos'] = self.env_map.valid_pos
            maze_configs['update'] = False

        # obtain the state and goal observation
        state_obs, goal_obs, _, _, _, _ = self.env.reset(maze_configs)
        
        # return states and goals
        return state_obs, goal_obs, init_pos, goal_pos

    def eval_policy(self):
        self.eval_dist_pairs = self.load_pair_data(self.maze_size, self.maze_seed)
        # sample a distance
        #tmp_dist = random.sample(list(np.arange(1, 5, 1)), 1)[0]
        tmp_dist = 1
        pairs_dict = {'start': self.eval_dist_pairs[str(tmp_dist)][0], 'goal': self.eval_dist_pairs[str(tmp_dist)][1]}
        # sample number
        eval_total_num = 50 if len(pairs_dict['start']) > 50 else len(pairs_dict['start'])
        eval_success_num = 0
        # obtain all the pairs
        pairs_idx = random.sample(range(len(pairs_dict['start'])), eval_total_num)
        # loop all the pairs
        for r, idx in enumerate(pairs_idx):
            # obtain the start-goal pair
            s_pos, g_pos = pairs_dict['start'][idx], pairs_dict['goal'][idx]
            # update the maze
            state, goal, start_pos, goal_pos = self.update_maze_from_pos(s_pos, g_pos)
            # obtain the fake observation
            #if not self.use_true_state and not self.use_her:
            #    goal_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, self.env_map.path[-1])
            #    goal = self.imagine_goal_observation(goal_loc_map)
            max_time_steps = 20
            act_list = []
            for t in range(max_time_steps):
                # get action
                action = self.agent.get_action(state, goal, 0)
                act_list.append(DEFAULT_ACTION_LIST[action])
                # step in the environment
                next_state, reward, done, dist, next_trans, _, _ = self.env.step(action)
                if done:
                    eval_success_num += 1
                    break
                else:
                    state = next_state
            #print(f"run = {r}: start = {s_pos}, goal = {g_pos}, act = {act_list}, done = {done}")
            #print("-----------------------------------------------")
        self.policy_returns.append(eval_success_num / eval_total_num)
        #print(f"Success rate = {eval_success_num / eval_total_num}")
        #print('********************************************')

    def eval_policy_novel(self):
        # loop all the testing mazes
        for m_size in self.maze_size_list:
            for m_seed in self.maze_seed_list:
                # print the current maze info
                # print(f'Evaluating maze - {m_size} - {m_seed}')
                self.eval_dist_pairs = self.load_pair_data(m_size, m_seed)
                self.maze_size = m_size
                self.maze_seed = m_seed
                self.update_map2d_and_maze3d(set_new_maze=True)
                # load the model
                # loop all the distance
                pairs_dict = {'start': self.eval_dist_pairs['1'][0], 'goal': self.eval_dist_pairs['1'][1]}
                # sample number
                eval_total_num = 10 if len(pairs_dict['start']) > 10 else len(pairs_dict['start'])
                eval_success_num = 0
                # obtain all the pairs
                pairs_idx = random.sample(range(len(pairs_dict['start'])), eval_total_num)
                # loop all the pairs
                for r, idx in enumerate(pairs_idx):
                    # obtain the start-goal pair
                    s_pos, g_pos = pairs_dict['start'][idx], pairs_dict['goal'][idx]
                    # update the maze
                    state, goal, start_pos, goal_pos = self.update_maze_from_pos(s_pos, g_pos)
                    # obtain the fake observation
                    if not self.use_true_state:
                        goal_loc_map = self.env_map.cropper(self.env_map.map2d_roughPadded, self.env_map.path[-1])
                        goal = self.imagine_goal_observation(goal_loc_map)
                    max_time_steps = 100
                    act_list = []
                    for t in range(max_time_steps):
                        # get action
                        action = self.agent.get_action(state, goal, 0)
                        act_list.append(DEFAULT_ACTION_LIST[action])
                        # step in the environment
                        next_state, reward, done, dist, next_trans, _, _, _, _ = self.env.step(action)
                        if done:
                            eval_success_num += 1
                            break
                        else:
                            state = next_state
                    #print(f"run = {r}: start = {s_pos}, goal = {g_pos}, act = {act_list}, done = {done}")
                    #print("-----------------------------------------------")
                self.policy_eval = eval_success_num / eval_total_num
                self.policy_returns.append(eval_success_num / eval_total_num)
                #print(f"Success rate = {eval_success_num / eval_total_num}")
                #print('********************************************')


    # evaluate the policy during training
    def policy_evaluate(self):
        # reset the environment
        self.fix_start, self.fix_goal = True, True
        state, goal, start_pos, goal_pos = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
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
        total_loss_path = os.path.join(self.save_dir, self.model_name + '_total_loss.npy')
        state_pred_loss_path = os.path.join(self.save_dir, self.model_name + '_state_pred_loss.npy')
        
        # save the results
        torch.save(self.agent.policy_net.state_dict(), model_save_path)
        np.save(distance_save_path, self.distance)
        np.save(returns_save_path, self.returns)
        np.save(lengths_save_path, self.lengths)
        np.save(policy_returns_save_path, self.policy_returns)

        # store the loss
        np.save(total_loss_path, self.agent.total_loss)
        np.save(state_pred_loss_path, self.agent.state_esti_loss)

    # load the pre-extract pairs
    @staticmethod
    def load_pair_data(m_size, m_seed):
        path = f'/mnt/sda/map/maze_{m_size}_{m_seed}.pkl'
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
        return state_obs, goal_obs, start_pos, goal_pos
