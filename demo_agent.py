""" This script is a demo to control agent in a specific static maze environment in DeepMind Lab.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pprint

# from envs import LabEnv
from model import NavAgent
from utils import mapper


def parse_input():
    """
        Function defines the input and parse the input
        Input args:
            None

        Output args:
            Input arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maze_type', type=str, default='static', help='Type of the generated maze')
    parser.add_argument('--max_frame', type=int, default=100, help='Maximal number of frames.')
    parser.add_argument('--width', type=int, default=320, help='Horizontal size of the observation')
    parser.add_argument('--height', type=int, default=240, help='Vertical size of the observation')
    parser.add_argument('--fps', type=int, default=60, help='Number of frames per second')
    parser.add_argument('--level_script', type=str, default='nav_maze_static_01', help='The environment to load')

    return parser.parse_args()


def run_agent(maze_type, max_frame, win_width, win_height, frame_fps, level_name):
    """
        Function is used to run the agent
    :param maze_type: type of the generated maze
    :param max_frame: maximal number of frames. i.e. total time steps
    :param win_width: width of the display window
    :param win_height: height of the display window
    :param frame_fps: frame per second
    :param level_name: name of the level script
    :return: None
    """
    # create the environment
    myEnv = LabEnv.LabEnv(maze_type, win_width, win_height, frame_fps, level_name)
    myAgent = NavAgent.NavAgent(myEnv.action_spec())
    # myAgent.print_info()
    episode_num = 100

    # # iterate through all the time steps
    fig, arr = plt.subplots(1, 2)
    for ep in tqdm(range(episode_num)):
        # initialization
        current_obs, _ = myEnv.reset()
        total_reward = 0
        img_artist_RGB = arr[0].imshow(current_obs[:, :, 0:3])
        img_artist_D = arr[1].imshow(current_obs[:, :, 3], cmap="Greys")
        total_reward = 0
        for t in tqdm(range(max_frame)):
            act = myAgent.step(current_obs)  # obtain the next action
            next_obs, reward, done = myEnv.step(act)

            img_artist_RGB.set_data(next_obs[:, :, 0:3])
            img_artist_D.set_data(next_obs[:, :, 3])
            plt.draw()
            plt.pause(0.000001)

            current_obs = next_obs
            total_reward += reward
        print('Ep {} - Reward - {}'.format(t+1, total_reward))


if __name__ == '__main__':
    # parse the input
    input_args = parse_input()

    # load a map and show
    env_map = mapper.RoughMap(7, 15)
    env_map.show_map('all')

    # env_map.path2egoaction(env_map.path, 'left')

    # print(env_map.path)
    # obtain the egocentric
    # # run the agent
    # run_agent(input_args.maze_type, input_args.max_frame, input_args.width, input_args.height, input_args.fps, \
    #           input_args.level_script)