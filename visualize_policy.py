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
import random

from utils import mapper
import IPython.terminal.debugger as Debug

from envs.LabEnv import RandomMaze


class VisualPolicy(object):
    def __init__(self, agent, maze_size, random_start, random_goal):
        self.agent = agent
        self.maze_size = maze_size
        self.random_start = random_start
        self.random_goal = random_goal

    def get_action(self, state):
        state = torch.tensor(np.array(state).transpose(0, 3, 1, 2)).float()
        action = self.agent(state).max(dim=1)[1].item()
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
        observation_width = 64
        observation_height = 64
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
            # # reset the environment using the size and seed
            pos_params = [env_map.init_pos[0],
                          env_map.init_pos[1],
                          env_map.goal_pos[0],
                          env_map.goal_pos[1],
                          0]  # [init_pos, goal_pos, init_orientation]

            state, goal = my_lab.reset(size, seed, pos_params)

            # show the observation in rgb and depth
            fig.canvas.set_window_title("Episode {} - {} x {} Maze - {} seed".format(ep, size, size, seed))
            # image_artist_rgb = arr[0].imshow(state[0])
            image_artist_rgb = arr[0].imshow(env_map.map2d_bw)
            arr[0].set_title('Front View')
            image_artist_topdown = arr[1].imshow(my_lab.top_down_obs)
            arr[1].set_title('Top Down View')

            # one episode starts
            total_reward = 0
            # valid positions
            num_valid_pos = len(env_map.valid_pos)
            max_steps = 50
            for t in tqdm(range(500), desc="Step loop"):
                # for i in range(1000):
                act = self.get_action(state)
                next_state, reward, done, dist, _ = my_lab.step(act)
                print("  Count = ", 50 - max_steps, dist)
                max_steps -= 1
                # image_artist_rgb.set_data(next_state[0])
                image_artist_rgb.set_data(env_map.map2d_bw)
                image_artist_topdown.set_data(ndimage.rotate(my_lab.top_down_obs, -90))

                fig.canvas.draw()
                plt.pause(0.001)
                if done or max_steps < 0:
                    max_steps = 50
                    env_map.map2d_bw[pos_params[0], pos_params[1]] = 1.0
                    env_map.map2d_bw[pos_params[2], pos_params[3]] = 1.0
                    start_pos, end_pos = env_map.sample_start_goal_pos((not self.random_start), (not self.random_goal))
                    pos_params = [start_pos[0],
                                  start_pos[1],
                                  end_pos[0],
                                  end_pos[1],
                                  0]
                    env_map.map2d_bw[end_pos[0], end_pos[1]] = 0.2
                    env_map.map2d_bw[start_pos[0], start_pos[1]] = 0.8
                    print("  Current pos : {} - {}".format(pos_params[0:2], pos_params[2:-1]))
                    my_lab.reset(size, seed, pos_params)
                else:
                    state = next_state
            plt.cla()


if __name__ == '__main__':
    # load the agent
    my_agent = DQNAgent(0, 0).policy_net
    my_agent.load_state_dict(torch.load("./results/3-28/double_dqn_fixed_goal.pt", map_location=torch.device('cpu')))
    my_agent.eval()

    # run the agent
    myVis = VisualPolicy(my_agent, 5, False, False)
    myVis.run()