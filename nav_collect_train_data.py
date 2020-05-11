""" This script is used to collect data to train the conditioned-VAE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from envs.LabEnvV2 import RandomMazeTileRaw
import argparse
import random
import matplotlib.pyplot as plt
from utils import mapper
from utils import save_data
from collections import defaultdict
import IPython.terminal.debugger as Debug

plt.rcParams.update({'font.size': 8})


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=32, help='Horizontal size of the observation')
    parser.add_argument('--height', type=int, default=32, help='Vertical size of the observation')
    parser.add_argument('--fps', type=int, default=60, help='Number of frames per second')
    parser.add_argument('--level', type=str, default='nav_random_maze', help='The environment to load')
    return parser.parse_args()


def run_agent(win_width, win_height, frame_fps, level):
    # desired observations
    observation_list = ['RGBD_INTERLEAVED',
                        'RGB.LOOK_PANORAMA_VIEW',
                        'RGB.LOOK_TOP_DOWN_VIEW'
                        ]

    # configurations
    configurations = {
        'width': str(win_width),
        'height': str(win_height),
        "fps": str(frame_fps)
    }

    # create the map environment
    myEnv = RandomMazeTileRaw(level,
                              observation_list,
                              configurations,
                              use_true_state=False,
                              reward_type="sparse-1",
                              dist_epsilon=1e-3)

    # maze sizes and seeds
    maze_size_list = [5, 7, 9, 11, 13]
    maze_seed_list = list(range(13))

    # maze
    theme_list = ['MISHMASH']
    decal_list = [0.001]

    pano_view_name = ["RGB.LOOK_EAST", "RGB.LOOK_NORTH_EAST",  "RGB.LOOK_NORTH", "RGB.LOOK_NORTH_WEST",
                      "RGB.LOOK_WEST", "RGB.LOOK_SOUTH_WEST", "RGB.LOOK_SOUTH", "RGB.LOOK_SOUTH_EAST"]

    for maze_size in maze_size_list:
        for maze_seed in maze_seed_list:
            # load the 2D and obtain valid positions
            env_map = mapper.RoughMap(maze_size, maze_seed, 3)
            print('Maze size : {} - {}'.format(maze_size, maze_seed))
            # initialize the maze environment
            maze_configs = defaultdict(lambda: None)
            maze_configs["maze_name"] = f"maze_{maze_size}x{maze_size}"  # string type name
            maze_configs["maze_size"] = [maze_size, maze_size]  # [int, int] list
            maze_configs["maze_seed"] = '1234'  # string type number
            maze_configs["maze_texture"] = random.sample(theme_list, 1)[0]  # string type name in theme_list
            maze_configs["maze_decal_freq"] = random.sample(decal_list, 1)[0]  # float number in decal_list
            maze_configs["maze_map_txt"] = "".join(env_map.map2d_txt)  # string type map
            maze_configs["maze_valid_pos"] = env_map.valid_pos  # list of valid positions
            # initialize the maze start and goal positions
            maze_configs["start_pos"] = env_map.init_pos + [0]  # start position on the txt map [rows, cols, orientation]
            maze_configs["goal_pos"] = env_map.goal_pos + [0]  # goal position on the txt map [rows, cols, orientation]
            maze_configs["update"] = True  # update flag
            # set the maze
            state, _, _, _ = myEnv.reset(maze_configs)
            for pos in env_map.valid_pos:
                # get the local map at the position
                pos_loc_map = env_map.cropper(env_map.map2d_roughPadded, pos)
                # get the observation at the position
                pos_obs = myEnv.get_random_observations(pos + [0])
                obs_list = [pos_obs[i, :, :, :] for i in range(8)]
                # save the images
                save_data.save_loc_maps_and_observations(maze_size,
                                                         maze_seed,
                                                         pos,
                                                         pos_loc_map,
                                                         obs_list,
                                                         pano_view_name,
                                                         "uniform-small")


if __name__ == '__main__':
    # parse the input
    input_args = parse_input()

    # # run the agent
    run_agent(input_args.width, input_args.height, input_args.fps, input_args.level)