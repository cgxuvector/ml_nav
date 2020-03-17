""" This script is a demo to control agent in a specific static maze environment in DeepMind Lab.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from utils import mapper
import IPython.terminal.debugger as Debug

import gym
from envs.LabEnv import RandomMaze


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


def run_agent(max_frame, win_width, win_height, frame_fps, level_name):
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
    # create the environment)
    observations = ["RGBD_INTERLEAVED", "RGB.LOK_TOP_DOWN", "RGB.LOOK_RANDOM"]
    # myEnv = gym.make("LabRandomMaze-v0",
    #                  observations=observations,
    #                  width=win_width,
    #                  height=win_height,
    #                  fps=frame_fps,
    #                  set_goal=True,
    #                  set_texture=True)
    myEnv = RandomMaze(observations=observations,
                       width=win_width,
                       height=win_height,
                       fps=frame_fps,
                       set_goal=True,
                       set_texture=True)

    # episodes
    episode_num = 100

    # maze size and seeds
    maze_size_list = [5]
    maze_seed_list = list(range(19))
    fig, arr = plt.subplots(1, 3, figsize=(12, 4))

    for ep in tqdm(range(episode_num), desc='Episode loop'):
        # randomly select a maze size and a seed
        maze_size = np.random.choice(maze_size_list)
        maze_seed = np.random.choice(maze_seed_list)

        # load map
        env_map = mapper.RoughMap(maze_size, maze_seed, 3)
        # # reset the environment using the size and seed
        pos_params = [env_map.raw_pos['init'][0],
                      env_map.raw_pos['init'][1],
                      env_map.raw_pos['goal'][0],
                      env_map.raw_pos['goal'][1],
                      0]  # [init_pos, goal_pos, init_orientation]

        print("Init : ({}, {})".format((pos_params[1] - 1) * 100 + 50, (maze_size - pos_params[0]) * 100 + 50))
        print("Goal : ({}, {})".format((pos_params[3] - 1) * 100 + 50, (maze_size - pos_params[2]) * 100 + 50))

        init_obs = myEnv.reset(maze_size, maze_seed, pos_params)

        # show the observation in rgb and depth
        image_artist_rgb = arr[0].imshow(init_obs['RGBD_INTERLEAVED'][:, :, 0:3])
        arr[0].set_title('Front View')
        image_artist_topdown = arr[1].imshow(init_obs['RGB.LOOK_TOP_DOWN'])
        arr[1].set_title('Top Down View')
        fig.canvas.set_window_title("Episode {} - {} x {} Maze - {} seed".format(ep, maze_size, maze_size, maze_seed))
        arr[2].set_title('2D Map')
        # image_artist_map = arr[2].imshow(env_map.map2d_bw)
        image_artist_random = plt.imshow(init_obs['RGB.LOOK_RANDOM'])

        # one episode starts
        total_reward = 0
        # valid positions
        num_valid_pos = len(env_map.valid_pos)
        for t in tqdm(range(max_frame), desc="Step loop"):
            act = np.random.randint(0, 5)
            current_obs, reward, done, _ = myEnv.step(act)
            image_artist_rgb.set_data(current_obs['RGBD_INTERLEAVED'][:, :, 0:3])
            image_artist_topdown.set_data(ndimage.rotate(current_obs['RGB.LOOK_TOP_DOWN'], -90))
            # image_artist_map.set_data(env_map.map2d_bw)
            myEnv.goal_observation = myEnv.get_random_observations(myEnv.goal_pos)
            image_artist_random.set_data(myEnv.goal_observation[np.random.choice(8, 1).item()])
            # obs = myEnv.get_random_observations(env_map.valid_pos[np.random.choice(num_valid_pos, 1).item()])
            # image_artist_random.set_data(obs[np.random.choice(8, 1).item()])

            fig.canvas.draw()
            plt.pause(0.001)
            if reward == 10:
                break
        plt.cla()
        print('Ep {} - Reward - {}'.format(t+1, total_reward))


if __name__ == '__main__':
    # parse the input
    input_args = parse_input()

    # # run the agent
    run_agent(input_args.max_frame, input_args.width, input_args.height, input_args.fps, \
              input_args.level_script)