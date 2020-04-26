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

    def run(self):
        """ Set up the Deepmind environment"""
        # necessary observations (correct: this is the egocentric observations (following the counter clock direction))
        observation_list = ['RGB.LOOK_EAST',
                            'RGB.LOOK_NORTH_EAST',
                            'RGB.LOOK_NORTH',
                            'RGB.LOOK_NORTH_WEST',
                            'RGB.LOOK_WEST',
                            'RGB.LOOK_SOUTH_WEST',
                            'RGB.LOOK_SOUTH',
                            'RGB.LOOK_SOUTH_EAST',
                            'RGB.LOOK_RANDOM',
                            'DEBUG.POS.TRANS',
                            'DEBUG.POS.ROT',
                            'RGB.LOOK_TOP_DOWN']
        observation_width = 32
        observation_height = 32
        observation_fps = 60

        # create the environment
        my_lab = RandomMaze(observation_list, observation_width, observation_height, observation_fps)

        # episodes
        episode_num = 100

        # maze size and seeds
        maze_size = [self.maze_size]
        maze_seed = [0]
        fig, arr = plt.subplots(1, 2)

        for ep in tqdm(range(episode_num), desc='Episode loop'):
            size = random.sample(maze_size, 1)[0]
            seed = random.sample(maze_seed, 1)[0]

            # load map
            env_map = mapper.RoughMap(size, seed, 3)

            # reset the environment using the size and seed
            pos_params = [env_map.init_pos[0],
                          env_map.init_pos[1],
                          env_map.goal_pos[0],
                          env_map.goal_pos[1],
                          0,
                          0]  # [init_pos, goal_pos, init_orientation]

            # update the postions
            pos_params[0:2], pos_params[2:4] = env_map.sample_global_start_goal_pos(self.random_start, self.random_goal, 2)
            state, goal = my_lab.reset(size, seed, pos_params)
            # show the observation in rgb and depth
            fig.canvas.set_window_title("Episode {} - {} x {} Maze - {} seed".format(ep, size, size, seed))
            arr[0].set_title('Map View')
            image_artist_rgb = arr[0].imshow(env_map.map2d_bw)
            arr[1].set_title('Top Down View')
            image_artist_topdown = arr[1].imshow(my_lab.top_down_obs)

            # show the policy
            max_steps = 20
            for t in tqdm(range(200), desc="Step loop"):
                # for i in range(1000):
                act = self.get_action(state, goal)
                next_state, reward, done, dist, _ = my_lab.step(act)
                max_steps -= 1
                # show the current positions
                image_artist_rgb.set_data(env_map.map2d_bw)
                image_artist_topdown.set_data(ndimage.rotate(my_lab.top_down_obs, -90))
                fig.canvas.draw()
                plt.pause(0.001)
                # check terminal
                if done or max_steps < 0:
                    max_steps = 20
                    env_map.map2d_bw[pos_params[0], pos_params[1]] = 1.0
                    env_map.map2d_bw[pos_params[2], pos_params[3]] = 1.0
                    start_pos, end_pos = env_map.sample_global_start_goal_pos(not self.random_start, not self.random_goal, 2)
                    pos_params = [start_pos[0],
                                  start_pos[1],
                                  end_pos[0],
                                  end_pos[1],
                                  0]
                    env_map.map2d_bw[end_pos[0], end_pos[1]] = 0.2
                    env_map.map2d_bw[start_pos[0], start_pos[1]] = 0.8
                    print("  Current pos : {} - {}".format(pos_params[0:2], pos_params[2:-1]))
                    state, goal = my_lab.reset(size, seed, pos_params)
                else:
                    state = next_state
            plt.cla()

    def init_run(self):
        """ Set up the Deepmind environment"""
        # necessary observations (correct: this is the egocentric observations (following the counter clock direction))
        observation_list = ['RGB.LOOK_EAST',
                            'RGB.LOOK_NORTH_EAST',
                            'RGB.LOOK_NORTH',
                            'RGB.LOOK_NORTH_WEST',
                            'RGB.LOOK_WEST',
                            'RGB.LOOK_SOUTH_WEST',
                            'RGB.LOOK_SOUTH',
                            'RGB.LOOK_SOUTH_EAST',
                            'RGB.LOOK_RANDOM',
                            'DEBUG.POS.TRANS',
                            'DEBUG.POS.ROT',
                            'RGB.LOOK_TOP_DOWN']
        observation_width = 32
        observation_height = 32
        observation_fps = 60

        # create the environment
        my_lab = RandomMaze(observation_list, observation_width, observation_height, observation_fps)

        # episodes
        episode_num = 100

        # maze size and seeds
        maze_size = [self.maze_size]
        maze_seed = [0]
        fig, arr = plt.subplots(1, 2)

        for ep in tqdm(range(episode_num), desc='Episode loop'):
            size = random.sample(maze_size, 1)[0]
            seed = random.sample(maze_seed, 1)[0]

            # load map
            env_map = mapper.RoughMap(size, seed, 3)
            # define the init position and goal position
            init_pos = [1, 1]
            goal_pos = [3, 3]
            env_map.update_mapper(init_pos, goal_pos)
            subgoal_count = 1
            subgoal_pos = env_map.path[subgoal_count].tolist()
            subgoal_pos.append(0)
            print("Next goal = ", subgoal_pos)
            # reset the environment using the size and seed
            pos_params = [env_map.init_pos[0],
                          env_map.init_pos[1],
                          env_map.goal_pos[0],
                          env_map.goal_pos[1],
                          0]  # [init_pos, goal_pos, init_orientation]
            state, goal = my_lab.reset(size, seed, pos_params)
            my_lab.goal_pos = subgoal_pos
            goal = my_lab.get_random_observations(subgoal_pos)
            # show the observation in rgb and depth
            fig.canvas.set_window_title("Episode {} - {} x {} Maze - {} seed".format(ep, size, size, seed))
            arr[0].set_title('Map View')
            image_artist_rgb = arr[0].imshow(env_map.map2d_bw)
            arr[1].set_title('Top Down View')
            image_artist_topdown = arr[1].imshow(my_lab.top_down_obs)

            # show the policy
            max_steps = 40
            # for t in tqdm(range(200), desc="Step loop"):
            for t in range(1000):
                act = self.get_action(state, goal)
                next_state, reward, done, dist, _ = my_lab.step(act)
                max_steps -= 1
                # show the current positions
                image_artist_rgb.set_data(env_map.map2d_bw)
                image_artist_topdown.set_data(ndimage.rotate(my_lab.top_down_obs, -90))
                fig.canvas.draw()
                plt.pause(0.001)
                # check terminal
                print("dist", dist)
                # if done or max_steps < 0:
                if dist < 40:
                    if my_lab.goal_pos == goal_pos + [0]:
                        break
                    subgoal_count += 1
                    subgoal_pos = env_map.path[subgoal_count].tolist()
                    subgoal_pos.append(0)
                    goal = my_lab.get_random_observations(subgoal_pos)
                    my_lab.goal_pos = subgoal_pos
                    # env_map.map2d_bw[pos_params[0], pos_params[1]] = 1.0
                    # env_map.map2d_bw[pos_params[2], pos_params[3]] = 1.0
                    # start_pos, end_pos = env_map.sample_global_start_goal_pos(not self.random_start,
                    #                                                           not self.random_goal, 2)
                    # pos_params = [start_pos[0],
                    #               start_pos[1],
                    #               end_pos[0],
                    #               end_pos[1],
                    #               0]
                    # env_map.map2d_bw[end_pos[0], end_pos[1]] = 0.2
                    # env_map.map2d_bw[start_pos[0], start_pos[1]] = 0.8
                    # print("  Current pos : {} - {}".format(pos_params[0:2], pos_params[2:-1]))
                    print("next goal = ", subgoal_pos)
                    # state, goal = my_lab.reset(size, seed, pos_params)
                else:
                    state = next_state
            plt.cla()

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


if __name__ == '__main__':
    # load the agent
    my_agent = DQNAgent(0, 0, use_true_state=True).policy_net
    # my_agent = GoalDQNAgent(0, 0, use_small_obs=True).policy_net
    my_agent.load_state_dict(torch.load("./results/4-23/double_dqn_11x11_ep_1000_b64_m50000.pt", map_location=torch.device('cpu')))
    my_agent.eval()

    # run the agent
    myVis = VisualPolicy(my_agent, 11, True, True)
    myVis.run_true_state()