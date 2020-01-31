"""
    Test agent in
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
from envs import LabEnv
from utils import mapper
from utils import searchAlg

import cv2
import numpy as np


def run(max_frame, width, height, fps, level, actions, oriens):
    """
        Input args
            max_frame:
            width: horizontal resolution of the observation frames (D: 320)
            height: vertical resolution of the observation frames (D: 240)
            fps:
            level: specified environment
        Output args
            None
    """
    # create the environment
    myEnv = LabEnv.NavEnv(width, height, fps, level)

    # create an agent
    myAgent = LabEnv.NavAgent(myEnv.action_spec)

    # number of runs
    run_num = 1
    episode_num = 1

    # for each run

    myEnv.reset()  # init the environment
    myAgent.reset()  # init the agent
    action = myAgent.step(actions[0])  # take action

    obs, r, done = myEnv.step(action)
    img_artist = plt.imshow(obs['RGB'].transpose(1, 2, 0))

    # adjust orientation
    init_orien = obs['DEBUG.POS.ROT'][1]
    print(init_orien)
    while abs(init_orien) > 1:
        observation, r, done = myEnv.step(np.array([10, 0, 0, 0, 0, 0, 0], dtype=np.intc))
        init_orien = observation['DEBUG.POS.ROT'][1]

    # for each episode
    count = 1

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('/home/chengguang/Planning.avi', fourcc, 20, (320, 240))

    print(oriens)
    while max_frame > 0:
        # episode termination check
        if done or count == len(actions):
            print("Environment is shutdown.")
            break

        # obtain an action
        print(actions[count])
        if count < len(oriens):
            current_ori = oriens[count]
        action = myAgent.step(actions[count])

        # step ahead
        exe_count = 19
        if current_ori == 'left-up' or current_ori == 'right-up':
            exe_count = 22
        if actions[count] == 'left' or actions[count] == 'right':
            exe_count = 44

        print("*********** Executing: ", actions[count])
        while exe_count > 0:
            observation, r, done = myEnv.step(action)

            obs_img = observation['RGB'].transpose(1, 2, 0)

            # show observations
            img_artist.set_data(obs_img)
            plt.draw()
            plt.pause(0.00001)

            # add reward
            myAgent.add_reward(r)
            max_frame -= 1
            exe_count -= 1

            # convert from RGB to BGR
            r_channel = obs_img[:, :, 0]
            g_channel = obs_img[:, :, 1]
            b_channel = obs_img[:, :, 2]
            new_obs = np.dstack((b_channel, g_channel, r_channel))
            writer.write(new_obs)

        count += 1
    writer.release()

    print('Finished after {} steps, Total reward is {}'.format(max_frame,
                                                               myAgent.get_reward()))




if __name__ == '__main__':
    # obtain the input arguments
    input_args = parse_input()

    # currently fix the maze
    maze_name = 'map_13_10.txt'
    map_data = mapper.load_map(maze_name, 'bw')
    mapper.show_map(map_data, 'bw')
    pos, grid_map = mapper.generate_grid_map(map_data)
    grid_map = grid_map.tolist()

    # # search the path
    path = searchAlg.A_star(grid_map, pos[0], pos[1])

    mapper.show_plan_path(path, grid_map)
    # initial_orientation = 'right-up'
    actions, oriens = mapper.path_to_actions(path, initial_orientation)
    # print(actions)

    # run(input_args.max_frame, input_args.width, input_args.height, input_args.fps, input_args.level_script, actions, oriens)



