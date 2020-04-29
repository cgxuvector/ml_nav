from utils import mapper
import numpy as np
from collections import namedtuple, defaultdict
import tqdm
from utils import memory
from utils import ml_schedule
import torch
import os
import random
import IPython.terminal.debugger as Debug
DEFAULT_TRANSITION = namedtuple("transition", ["state", "action", "next_state", "reward", "goal", "done"])
ACTION_LIST = ['up', 'down', 'left', 'right']


class Experiment(object):
    def __init__(self,
                 env,
                 agent,
                 maze_list,
                 seed_list,
                 decal_freq=0.1,
                 fix_maze=True,
                 fix_start=True,
                 fix_goal=True,
                 use_goal=False,
                 sampled_goal=10,
                 train_episode_num=10,
                 start_train_step=1000,
                 max_time_steps=50000,
                 episode_time_steps=2000,
                 use_true_state=False,
                 use_replay=False,
                 use_her=False,
                 buffer_size=None,
                 transition=DEFAULT_TRANSITION,
                 learning_rate=1e-3,
                 batch_size=64,
                 gamma=0.99,
                 eps_start=1,
                 eps_end=0.1,
                 goal_dist=1,
                 future_k=4,
                 save_dir=None,
                 model_name=None,
                 device="cpu"
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
        # training configurations
        self.use_goal = use_goal
        self.use_true_state = use_true_state
        self.last_goal = None
        self.goal_dist = goal_dist
        self.sampled_goal = sampled_goal
        self.train_episode_num = train_episode_num
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
        self.use_replay_buffer = use_replay
        self.use_her = use_her
        # rl related configuration
        self.gamma = gamma
        self.schedule = ml_schedule.LinearSchedule(eps_start, eps_end, max_time_steps/3)
        # results statistics
        self.distance = []
        self.returns = []
        self.lengths = []
        self.policy_returns = []
        # saving settings
        self.model_name = model_name
        self.save_dir = save_dir
        # orientation space
        self.init_orientation_space = np.linspace(0, 360, num=37).tolist()
        self.goal_orientation_space = np.linspace(0, 315, num=8).tolist()
        # future strategy
        self.her_future_k = future_k

    def run_dqn(self):
        """
        Function is used to run the training of the agent
        """
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

                pbar.set_description(
                    f'Episode: {episode_idx} | Steps: {episode_t} | Return: {G:2f} | Dist: {dist:.2f} | '
                    f'Init: {self.env.start_pos} | Goal: {self.env.goal_pos} | '
                    f'Eps: {eps:.3f} | Buffer: {len(self.replay_buffer)}'
                )

                # evaluate the current policy
                if (episode_idx - 1) % 100 == 0:
                    # evaluate the current policy by interaction
                    with torch.no_grad():
                        self.policy_evaluate()
                        # save the model
                        # model_save_path = os.path.join(self.save_dir, self.model_name) + f"_{episode_idx}.pt"
                        # torch.save(self.agent.policy_net.state_dict(), model_save_path)

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

        model_save_path = os.path.join(self.save_dir, self.model_name) + f"_{len(self.returns)}.pt"
        distance_save_path = os.path.join(self.save_dir, self.model_name + "_distance.npy")
        returns_save_path = os.path.join(self.save_dir, self.model_name + "_return.npy")
        policy_returns_save_path = os.path.join(self.save_dir, self.model_name + "_policy_return.npy")
        lengths_save_path = os.path.join(self.save_dir, self.model_name + "_length.npy")
        torch.save(self.agent.policy_net.state_dict(), model_save_path)
        np.save(distance_save_path, self.distance)
        np.save(returns_save_path, self.returns)
        np.save(lengths_save_path, self.lengths)
        np.save(policy_returns_save_path, self.policy_returns)

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

    # evaluate the policy during training
    def policy_evaluate(self):
        # reset the environment
        state, goal = self.update_map2d_and_maze3d(set_new_maze=not self.fix_maze)
        rewards = []
        actions = []
        for i in range(self.max_steps_per_episode):
            # get one action
            action = self.agent.get_action(state, 0)
            actions.append(ACTION_LIST[action])
            # step in the environment
            next_state, reward, done, dist, trans, _, _ = self.env.step(action)

            # check terminal
            if done:
                break
            else:
                state = next_state
                rewards.append(reward)

        # compute the discounted return for each time step
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G

        # store the current policy return
        print("evaluate = ",G, actions)
        self.policy_returns.append(G)
