""" This script is a demo to control agent in a specific static maze environment in DeepMind Lab.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import random
from test.DQNAgent import DQNAgent
from test.GoalDQNAgent import GoalDQNAgent
from utils import mapper
from collections import defaultdict
from envs.LabEnvV2 import RandomMazeTileRaw
import IPython.terminal.debugger as Debug

ACTION_LIST = ['up', 'down', 'left', 'right']


class VisualPolicy(object):
    def __init__(self, env, agent, size, set_goal, fix_start, fix_goal, g_dist):
        self.env = env
        self.env_map = None
        self.agent = agent
        self.maze_size = size
        self.maze_seed = 0
        self.maze_size_list = [size]
        self.maze_seed_list = [0]
        self.fix_start = fix_start
        self.fix_goal = fix_goal
        self.theme_list = ['MISHMASH']
        self.decal_list = [0.1]
        self.use_goal = set_goal
        self.goal_dist = g_dist
        self.gamma = 0.99

    def run_fixed_start_goal_pos(self):
        # init the environment
        self.fix_start = True
        self.fix_goal = True
        state, goal = self.update_map2d_and_maze3d(set_new_maze=True)

        # episodes
        episode_num = 100
        states = [state]
        rewards = []
        actions = []
        for t in range(episode_num):
            # get one action
            if self.use_goal:
                action = self.agent.get_action(state, goal, 0)
            else:
                action = self.agent.get_action(state, 0)

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = my_lab.step(action)

            # save the results
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)

            # print the steps
            print(f"Current state = {state}, Action = {ACTION_LIST[action]}, Next state = {next_state}")

            # check terminal
            if done:
                break
            else:
                state = next_state

        # compute the discounted return
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
        print("Discounted return of the init state = {}".format(G))

    def run_random_start_goal_pos(self):
        # init the environment
        self.fix_start = True
        self.fix_goal = True
        state, goal = self.update_map2d_and_maze3d(set_new_maze=True)

        # episodes
        episode_num = 100
        states = [state]
        rewards = []
        actions = []
        for t in range(episode_num):
            # get one action
            if self.use_goal:
                action = self.agent.get_action(state, goal, 0)
            else:
                action = self.agent.get_action(state, 0)

            # step in the environment
            next_state, reward, done, dist, trans, _, _ = my_lab.step(action)

            # save the results
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)

            # print the steps
            print(f"Current state = {state}, Action = {ACTION_LIST[action]}, Next state = {next_state}")

            # check terminal
            if done:
                break
            else:
                state = next_state

        # compute the discounted return
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
        print("Discounted return of the init state = {}".format(G))

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
            init_pos, goal_pos = self.env_map.sample_random_start_goal_pos(self.fix_start, self.fix_goal, self.goal_dist)
            maze_configs['start_pos'] = init_pos + [0]
            maze_configs['goal_pos'] = goal_pos + [0]
            maze_configs['maze_valid_pos'] = self.env_map.valid_pos
            maze_configs['update'] = False

        # obtain the state and goal observation
        state_obs, goal_obs, _, _ = self.env.reset(maze_configs)
        # return states and goals
        return state_obs, goal_obs


if __name__ == '__main__':
    # set level name
    level_name = 'nav_random_maze'
    # necessary observations (correct: this is the egocentric observations (following the counter clock direction))
    observation_list = [
        'RGB.LOOK_PANORAMA_VIEW',
        'RGB.LOOK_TOP_DOWN_VIEW'
    ]
    observation_width = 32
    observation_height = 32
    observation_fps = 60
    # set configurations
    configurations = {
        'width': str(observation_width),
        'height': str(observation_height),
        'fps': str(observation_fps)
    }
    # create the environment
    my_lab = RandomMazeTileRaw(level_name,
                               observation_list,
                               configurations,
                               use_true_state=True,
                               reward_type="sparse-1",
                               dist_epsilon=1e-3)

    # set parameters
    maze_size = 5
    use_obs = False
    use_goal = True
    goal_dist = 100
    seed = 0

    # load the agent
    if not use_obs:
        if not use_goal:
            my_agent = DQNAgent(use_true_state=True, use_small_obs=True).policy_net
            my_agent.load_state_dict(
                torch.load(f"./results/5-4/ddqn_{maze_size}x{maze_size}_true_state_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
        else:
            my_agent = GoalDQNAgent(use_true_state=True, use_small_obs=True).policy_net
            my_agent.load_state_dict(
                torch.load(f"./results/5-4/random_goal_ddqn_{maze_size}x{maze_size}_true_state_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
    else:
        if not use_goal:
            my_agent = DQNAgent(use_true_state=False, use_small_obs=True).policy_net
            my_agent.load_state_dict(
                torch.load(f"./results/5-4/ddqn_{maze_size}x{maze_size}_obs_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
        else:
            my_agent = GoalDQNAgent(use_true_state=False, use_small_obs=True).policy_net
            my_agent.load_state_dict(
                torch.load(f"./results/5-4/random_goal_ddqn_{maze_size}x{maze_size}_obs_double_seed_{seed}.pt",
                           map_location=torch.device('cpu'))
            )
    my_agent.eval()

    # run the agent
    myVis = VisualPolicy(env=my_lab,
                         agent=my_agent,
                         size=maze_size,
                         set_goal=use_goal,
                         fix_start=True,
                         fix_goal=True,
                         g_dist=goal_dist)
    myVis.run_fixed_start_goal_pos()