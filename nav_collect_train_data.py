# """ This script is a demo to control agent in a specific static maze environment in DeepMind Lab.
# """
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import argparse
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 8})
# import numpy as np
# from utils import mapper
# import gym
# import gym_deepmindlab
#
# from utils import save_data
#
#
# def parse_input():
#     """
#         Function defines the input and parse the input
#         Input args:
#             None
#
#         Output args:
#             Input arguments
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--maze_type', type=str, default='static', help='Type of the generated maze')
#     parser.add_argument('--max_frame', type=int, default=10, help='Maximal number of frames.')
#     parser.add_argument('--width', type=int, default=120, help='Horizontal size of the observation')
#     parser.add_argument('--height', type=int, default=120, help='Vertical size of the observation')
#     parser.add_argument('--fps', type=int, default=60, help='Number of frames per second')
#     parser.add_argument('--level_script', type=str, default='load_random_maze', help='The environment to load')
#
#     return parser.parse_args()
#
#
# def run_agent(max_frame, win_width, win_height, frame_fps, level_name):
#     """
#         Function is used to run the agent
#     :param maze_type: type of the generated maze
#     :param max_frame: maximal number of frames. i.e. total time steps
#     :param win_width: width of the display window
#     :param win_height: height of the display window
#     :param frame_fps: frame per second
#     :param level_name: name of the level script
#     :return: None
#     """
#     # create the environment)
#     myEnv = gym.make('DeepmindLabRandomMazeCustomViewNoGoalVariousTexture-v0', width=win_width, height=win_height, \
#                      observations=["RGBD_INTERLEAVED", \
#                                    "RGB.LOOK_EAST", \
#                                    "RGB.LOOK_NORTH_EAST", \
#                                    "RGB.LOOK_NORTH", \
#                                    "RGB.LOOK_NORTH_WEST", \
#                                    "RGB.LOOK_WEST", \
#                                    "RGB.LOOK_SOUTH_WEST", \
#                                    "RGB.LOOK_SOUTH", \
#                                    "RGB.LOOK_SOUTH_EAST"])
#
#     # set the plotting
#     pano_fig, pano_arr = plt.subplots(3, 3, figsize=(9, 9))
#     pano_fig.canvas.set_window_title("360 view")
#     pano_view_title = ["NORTH_WEST", "NORTH", "NORTH_EAST", \
#                       "WEST", "Local 2D Map", "EAST", \
#                       "SOUTH_WEST", "SOUTH", "SOUTH_EAST"]
#     pano_view_name =["RGB.LOOK_NORTH_WEST", "RGB.LOOK_NORTH", "RGB.LOOK_NORTH_EAST", \
#                       "RGB.LOOK_WEST", "", "RGB.LOOK_EAST", \
#                       "RGB.LOOK_SOUTH_WEST", "RGB.LOOK_SOUTH", "RGB.LOOK_SOUTH_EAST"]
#     sample_num = 100
#     # maze_size_list = [5, 7, 9, 11, 13]
#     # maze_seed_list  = list(range(19))
#     maze_size_list = [5, 7, 9]  # currently, I only test 7x7 size maze
#     maze_seed_list = list(range(15))  # 0 - 14 for training and 15 - 19 for testing
#
#     np.random.seed(1234)
#     for maze_size in maze_size_list:
#         for maze_seed in maze_seed_list:
#             # load the 2D and obtain valid positions
#             env_map = mapper.RoughMap(maze_size, maze_seed, 3)
#             print('Maze size : {} - {}'.format(maze_size, maze_seed))
#             for pos in env_map.valid_pos:
#                 env_map.raw_pos['init'] = pos
#             # for s in range(len(env_map.valid_pos)):
#             #     # load map
#             #     env_map.raw_pos['init'] = env_map.valid_pos[np.random.randint(len(env_map.valid_pos))]
#                 print(env_map.raw_pos['init'])
#
#                 # env_map.show_map("all")
#
#                 # reset the environment using the size and seed
#                 pos_params = [env_map.raw_pos['init'][0] + 1,
#                               env_map.raw_pos['init'][1] + 1,
#                               env_map.raw_pos['goal'][0] + 1,
#                               env_map.raw_pos['goal'][1] + 1,
#                               0]  # [init_pos, goal_pos, init_orientation]
#
#                 init_obs = myEnv.reset(maze_size, maze_seed, pos_params)
#                 # construct artists for plotting
#                 tmp_loc_map = env_map.cropper(env_map.map2d_roughPadded, np.array(env_map.raw_pos['init']))
#
#                 pano_artists = []
#                 for i in range(3):
#                     for j in range(3):
#                         pano_arr[i, j].set_title(pano_view_title[i * 3 + j])
#                         if i == 1 and j == 1:
#                             pano_artists.append(pano_arr[i, j].imshow(tmp_loc_map))
#                         else:
#                             pano_artists.append(pano_arr[i, j].imshow(init_obs[pano_view_name[i * 3 + j]]))
#                 # # plt.show()
#                 # plt.pause(0.0001)
#
#                 # one episode starts
#                 total_reward = 0
#                 for t in range(10):
#                     # act = np.random.randint(0, 5)
#                     act = -1
#                     current_obs, reward, done, _ = myEnv.step(act)
#                     for m in range(3):
#                         for n in range(3):
#                             if m == 1 and n == 1:
#                                 pano_artists[m * 3 + n].set_data(tmp_loc_map)
#                             else:
#                                 pano_artists[m * 3 + n].set_data(current_obs[pano_view_name[m * 3 + n]])
#                     # pano_fig.canvas.draw()
#                     # plt.pause(0.0001)
#                     if reward == 10:
#                         break
#                 # plt.savefig("./figures/local_map_obs/maze_{}_{}_{}.png".format(maze_size, maze_seed, env_map.raw_pos['init']), dpi=100)
#                 # save_data.save_loc_maps_and_observations(maze_size, maze_seed, env_map.raw_pos['init'], tmp_loc_map, current_obs, pano_view_name, "random")
#                 # save the images
#
#
# if __name__ == '__main__':
#     # parse the input
#     input_args = parse_input()
#
#     # # run the agent
#     run_agent(input_args.max_frame, input_args.width, input_args.height, input_args.fps, \
#               input_args.level_script)