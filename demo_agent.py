""" This script is a demo to control agent in a specific static maze environment in DeepMind Lab.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pprint
import numpy as np

from utils import mapper

import gym
import gym_deepmindlab

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
    parser.add_argument('--width', type=int, default=120, help='Horizontal size of the observation')
    parser.add_argument('--height', type=int, default=120, help='Vertical size of the observation')
    parser.add_argument('--fps', type=int, default=60, help='Number of frames per second')
    parser.add_argument('--level_script', type=str, default='load_random_maze', help='The environment to load')

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
    # myEnv = gym.make('DeepmindLabLoadRandomMaze-v0', width=win_width, height=win_height, colors="DEBUG.CAMERA.TOP_DOWN")
    myEnv = gym.make('DeepmindLabLoadRandomMaze-v0', width=win_width, height=win_height, colors="RGBD_INTERLEAVED")

    # myAgent.print_info()
    episode_num = 100

    maze_size_list = [5, 7, 9, 11, 13]
    maze_seed_list = list(range(19))
    fig, arr = plt.subplots(1, 3)
    for ep in tqdm(range(episode_num)):
        # randomly select a maze size and a seed
        maze_size = np.random.choice(maze_size_list)
        maze_seed = np.random.choice(maze_seed_list)
        # reset the environment using the size and seed
        # maze_seed = 5
        # maze_size = 5
        init_obs = myEnv.reset(maze_size, maze_seed)
        # plt.imshow(init_obs.transpose(1, 2, 0))
        # plt.show()
        # # show the observation in rgb and depth
        image_artist_rgb = arr[0].imshow(init_obs[:, :, 0:3])
        arr[0].set_title('RGB')
        image_artist_depth = arr[1].imshow(init_obs[:, :, 3])
        arr[1].set_title('Depth')
        fig.canvas.set_window_title("{} x {} Maze - {} seed".format(maze_size, maze_size, maze_seed))

        # load a map and show
        env_map = mapper.RoughMap(maze_size, maze_seed, 3)
        arr[2].set_title('Map')
        image_artist_map = arr[2].imshow(env_map.map2d_rough)

        # one episode starts
        total_reward = 0
        for t in tqdm(range(max_frame)):
            act = np.random.randint(0, 5)
            current_obs, reward, done, _ = myEnv.step(act)
            image_artist_rgb.set_data(current_obs[:, :, 0:3])
            image_artist_depth.set_data(current_obs[:, :, 3])
            image_artist_map.set_data(env_map.map2d_rough)
            fig.canvas.draw()
            plt.pause(0.0001)
            total_reward += reward
        plt.cla()
        print('Ep {} - Reward - {}'.format(t+1, total_reward))


if __name__ == '__main__':
    # parse the input
    input_args = parse_input()

    # # load a map and show
    # env_map = mapper.RoughMap(9, 15, 3)
    # env_map.show_map('all')
    # env_map.crop_local_maps()

    # environment

    # run the agent
    run_agent(input_args.maze_type, input_args.max_frame, input_args.width, input_args.height, input_args.fps, \
              input_args.level_script)