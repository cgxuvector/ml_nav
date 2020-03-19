"""
    Implementation of collecting data from the environment
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
from utils import mapper
import gym
import gym_deepmindlab
import matplotlib.pyplot as plt
from skimage.transform import resize

from utils import save_data

plt.rcParams.update({'font.size': 8})

DEFAULT_LEVELS = ['DeepmindLabRandomMazeCustomViewFixedTexture-v0',
                  'DeepmindLabRandomMazeCustomViewNoGoalFixedTexture-v0',
                  'DeepmindLabRandomMazeCustomViewVariousTexture-v0',
                  'DeepmindLabRandomMazeCustomViewNoGoalVariousTexture-v0']

DEFAULT_OBSERVATIONS = ["RGBD_INTERLEAVED", "RGB_INTERLEAVED",
                        "RGB.LOOK_EAST", "RGB.LOOK_NORTH_EAST",
                        "RGB.LOOK_NORTH", "RGB.LOOK_NORTH_WEST",
                        "RGB.LOOK_WEST", "RGB.LOOK_SOUTH_WEST",
                        "RGB.LOOK_SOUTH", "RGB.LOOK_SOUTH_EAST",
                        "RGB.TOP_DOWN"]

DEFAULT_MAZE_SIZE = [5, 7, 9, 11, 13]


# reshape the map
def rescale_map(raw_map, size=21):
    # map size
    map_size = raw_map.shape[0]
    assert map_size > 0, f"Invalid map size. Expect map size > 0, but get {map_size}"
    # scale and padding
    scale_size = size // map_size
    size_to_pad = size - scale_size * map_size
    size_to_cut = (scale_size + 1) * map_size - size

    # select the smaller operation
    scale_size = scale_size if size_to_pad <= size_to_cut else scale_size + 1

    # scale the map
    scaled_map = np.zeros((scale_size * map_size, scale_size * map_size))
    row = scaled_map.shape[0]
    col = scaled_map.shape[1]
    for i in range(row):
        for j in range(col):
            raw_i = i // scale_size
            raw_j = j // scale_size
            scaled_map[i, j] = raw_map[raw_i, raw_j]

    row = scaled_map.shape[0]
    col = scaled_map.shape[1]
    if size_to_pad <= size_to_cut:  # padding the map
        padded_scaled_map = np.zeros((size, size))
        padding_step = size_to_pad // 2
        if padding_step:
            padded_scaled_map[padding_step:(padding_step+row), padding_step:(padding_step+col)] = scaled_map
        else:
            padded_scaled_map[0:row, 0:col] = scaled_map
        final_map = padded_scaled_map
    else:  # cut the image
        cut_scaled_map = scaled_map[1:row+1, 1:col+1]
        final_map = cut_scaled_map

    return final_map


def collect_map_and_obs(env,
                        env_map,
                        size,
                        seed,
                        obs_titles,
                        obs_names,
                        fig_arr,
                        fig,
                        mode):
    count = 0
    # loop all the valid positions in the map
    for pos in env_map.valid_pos:
        # if count > 0:
        #     break
        # count += 1
        env_map.raw_pos['init'] = pos
        # reset the environment using the size and seed
        pos_params = [env_map.raw_pos['init'][0] + 1,
                      env_map.raw_pos['init'][1] + 1,
                      env_map.raw_pos['goal'][0] + 1,
                      env_map.raw_pos['goal'][1] + 1,
                      0]  # [init_pos, goal_pos, init_orientation]
        init_obs = env.reset(size, seed, pos_params)
        # crop the local map
        if mode == "local":
            tmp_loc_map = env_map.cropper(env_map.map2d_roughPadded, np.array(env_map.raw_pos['init']))
        else:
            tmp_loc_map = copy.deepcopy(env_map.map2d_rough)
            tmp_loc_map[pos_params[0]-1, pos_params[1]-1] = 0.5
            tmp_loc_map = rescale_map(tmp_loc_map)

        # construct artists for plotting
        pano_artists = []
        for i in range(3):
            for j in range(3):
                fig_arr[i, j].set_title(obs_titles[i * 3 + j])
                if i == 1 and j == 1:
                    pano_artists.append(fig_arr[i, j].imshow(tmp_loc_map))
                else:
                    pano_artists.append(fig_arr[i, j].imshow(init_obs[obs_names[i * 3 + j]]))
        # # plt.show()
        # plt.pause(0.0001)

        # capture the observation after 10 time steps: avoid disturb
        for t in range(10):
            act = -1
            current_obs, reward, done, _ = env.step(act)
            for m in range(3):
                for n in range(3):
                    if m == 1 and n == 1:
                        pano_artists[m * 3 + n].set_data(tmp_loc_map)
                    else:
                        pano_artists[m * 3 + n].set_data(current_obs[obs_names[m * 3 + n]])
            # fig.canvas.draw()
            # plt.pause(0.0001)
            if reward == 10:
                break
            # plt.cla
        # plt.savefig("./figures/local_map_obs/maze_{}_{}_{}.png".format(maze_size, maze_seed, env_map.raw_pos['init']), dpi=100)
        save_data.save_loc_maps_and_observations(size, seed, env_map.raw_pos['init'], tmp_loc_map, current_obs, obs_names, "uniform")
        # save the images


# collect data for conditional VAE
def run_collect_vae_data(level_name, image_size, maze_size_list, observations, collect_mode="local"):
    # check the input variables
    assert level_name in DEFAULT_LEVELS, f"Invalid level name. Expect {DEFAULT_LEVELS}, but get {level_name}"
    assert set(maze_size_list) <= set(DEFAULT_MAZE_SIZE), f"Invalid maze sizes: Expect sizes in {DEFAULT_MAZE_SIZE}, " \
                                                          f"but get {maze_size_list}"

    assert set(observations) <= set(DEFAULT_OBSERVATIONS), f"Invalid observations. Expect observations in" \
                                                           f"{DEFAULT_OBSERVATIONS}, but get {observations}"

    # create the environment)
    myEnv = gym.make(level_name, width=image_size[0], height=image_size[1], observations=observations)

    # set the plotting
    pano_fig, pano_arr = plt.subplots(3, 3, figsize=(9, 9))
    pano_fig.canvas.set_window_title("360 view")
    if collect_mode == "local":
        pano_view_title = ["NORTH_WEST", "NORTH", "NORTH_EAST",
                           "WEST", "local 2D Map", "EAST",
                           "SOUTH_WEST", "SOUTH", "SOUTH_EAST"]
    else:
        pano_view_title = ["NORTH_WEST", "NORTH", "NORTH_EAST",
                           "WEST", "global 2D Map", "EAST",
                           "SOUTH_WEST", "SOUTH", "SOUTH_EAST"]

    pano_view_name = ["RGB.LOOK_NORTH_WEST", "RGB.LOOK_NORTH", "RGB.LOOK_NORTH_EAST",
                      "RGB.LOOK_WEST", "", "RGB.LOOK_EAST",
                      "RGB.LOOK_SOUTH_WEST", "RGB.LOOK_SOUTH", "RGB.LOOK_SOUTH_EAST"]

    np.random.seed(1234)
    maze_seed_list = np.arange(15)  # 0 - 14 for training and 15 - 19 for testing
    for maze_size in maze_size_list:
        for maze_seed in maze_seed_list:
            # print info
            print('Maze size : {} - {}'.format(maze_size, maze_seed))
            # load the 2D and obtain valid positions
            env_map = mapper.RoughMap(maze_size, maze_seed, 3)
            collect_map_and_obs(myEnv,
                                env_map,
                                maze_size,
                                maze_seed,
                                pano_view_title,
                                pano_view_name,
                                pano_arr,
                                pano_fig,
                                collect_mode)


if __name__ == "__main__":
    run_collect_vae_data(
        level_name='DeepmindLabRandomMazeCustomViewNoGoalFixedTexture-v0',
        image_size=(64, 64),
        maze_size_list=[5, 7, 9, 11, 13],
        observations=["RGB.LOOK_EAST",
                      "RGB.LOOK_NORTH_EAST",
                      "RGB.LOOK_NORTH",
                      "RGB.LOOK_NORTH_WEST",
                      "RGB.LOOK_WEST",
                      "RGB.LOOK_SOUTH_WEST",
                      "RGB.LOOK_SOUTH",
                      "RGB.LOOK_SOUTH_EAST"],
        collect_mode='global'
    )