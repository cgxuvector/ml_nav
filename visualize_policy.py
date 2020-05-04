""" This script is a demo to control agent in a specific static maze environment in DeepMind Lab.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from test.DQNAgent import DQNAgent
from test.GoalDQNAgent import GoalDQNAgent
import random

from utils import mapper
from collections import defaultdict
import IPython.terminal.debugger as Debug

from envs.LabEnv import RandomMaze
from envs.LabEnvV2 import RandomMazeTileRaw


ACTION_LIST = ['up', 'down', 'left', 'right']


class VisualPolicy(object):
    def __init__(self, agent, maze_size, random_start, random_goal):
        self.agent = agent
        self.maze_size = maze_size
        self.random_start = random_start
        self.random_goal = random_goal
        # orientation space
        self.init_orientation_space = np.linspace(0, 360, num=37).tolist()
        self.goal_orientation_space = np.linspace(0, 315, num=8).tolist()

    def get_action(self, state, goal):
        state = torch.tensor(np.array(state).transpose(0, 3, 1, 2)).float()
        goal = torch.tensor(np.array(goal).transpose(0, 3, 1, 2)).float()
        action = self.agent(state, goal).max(dim=1)[1].item()
        return action

    def run_true_state(self):
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

        # load the state
        maze_configs = defaultdict(lambda: None)
        # maze size and seeds
        maze_size = self.maze_size
        maze_seed = 0
        # initialize the map 2D
        env_map = mapper.RoughMap(maze_size, maze_seed, 3)
        print("Planned policy and states:")
        print([pos.tolist() for pos in env_map.path])
        print(env_map.map_act)
        print("********************************************")
        # initialize the maze 3D
        maze_configs["maze_name"] = f"maze_{maze_size}x{maze_size}"  # string type name
        maze_configs["maze_size"] = [maze_size, maze_size]  # [int, int] list
        maze_configs["maze_seed"] = '1234'  # string type number
        maze_configs["maze_map_txt"] = "".join(env_map.map2d_txt)  # string type map
        maze_configs["maze_valid_pos"] = env_map.valid_pos  # list of valid positions
        # initialize the maze start and goal positions
        maze_configs["start_pos"] = env_map.init_pos + [0]  # start position on the txt map [rows, cols, orientation]
        maze_configs["goal_pos"] = env_map.goal_pos + [0]  # goal position on the txt map [rows, cols, orientation]
        # initialize the update flag
        maze_configs["update"] = True  # update flag
        state, _, _, _ = my_lab.reset(configs=maze_configs)
        # episodes
        episode_num = 100
        print("Learned policy and states:")
        for t in range(episode_num):
            action = self.agent(torch.tensor(state).float()).max(dim=0)[1].item()
            next_state, reward, done, dist, trans, _, _ = my_lab.step(action)

            print(f"Current state = {state}, Action = {ACTION_LIST[action]}, Next state = {next_state}")

            # check terminal
            if done:
                break
            else:
                state = next_state

    def run_true_obs(self):
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
                                   use_true_state=False,
                                   reward_type="sparse-1",
                                   dist_epsilon=1e-3)

        # load the state
        maze_configs = defaultdict(lambda: None)
        # maze size and seeds
        maze_size = self.maze_size
        maze_seed = 0
        # initialize the map 2D
        env_map = mapper.RoughMap(maze_size, maze_seed, 3)
        print("Planned policy and states:")
        print([pos.tolist() for pos in env_map.path])
        print(env_map.map_act)
        print("********************************************")
        # initialize the maze 3D
        maze_configs["maze_name"] = f"maze_{maze_size}x{maze_size}"  # string type name
        maze_configs["maze_size"] = [maze_size, maze_size]  # [int, int] list
        maze_configs["maze_seed"] = '1234'  # string type number
        maze_configs["maze_map_txt"] = "".join(env_map.map2d_txt)  # string type map
        maze_configs["maze_valid_pos"] = env_map.valid_pos  # list of valid positions
        # initialize the maze start and goal positions
        maze_configs["start_pos"] = env_map.init_pos + [0]  # start position on the txt map [rows, cols, orientation]
        maze_configs["goal_pos"] = env_map.goal_pos + [0]  # goal position on the txt map [rows, cols, orientation]
        # initialize the update flag
        maze_configs["update"] = True  # update flag
        state, _, state_pos, _ = my_lab.reset(configs=maze_configs)
        # episodes
        episode_num = 100
        print("Learned policy and states:")
        with torch.no_grad():
            for t in range(episode_num):
                action = self.agent(torch.tensor(state.transpose(0, 3, 1, 2)).float()).max(dim=1)[1].item()
                next_state, reward, done, dist, trans, _, _ = my_lab.step(action)

                print(f"Current state = {state_pos}, Action = {ACTION_LIST[action]}, Next state = {trans}")
                # check terminal
                if done:
                    break
                else:
                    state = next_state
                    state_pos = trans


if __name__ == '__main__':
    # load the agent
    my_agent = DQNAgent(0, 0, use_true_state=False, use_small_obs=True).policy_net
    my_agent.load_state_dict(torch.load("./results/5-1/ddqn_13x13_obs_decal_1_m20000_double_13828.pt", map_location=torch.device('cpu')))
    my_agent.eval()

    # run the agent
    myVis = VisualPolicy(my_agent, 13, True, True)
    myVis.run_true_obs()