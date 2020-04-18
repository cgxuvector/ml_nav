import deepmind_lab
import numpy as np
import random
import matplotlib.pyplot as plt
import IPython.terminal.debugger as Debug
from utils import mapper
from envs.LabEnvV1 import RandomMazeV1
from collections import defaultdict
from scipy import ndimage


def run_test():
    # level name
    level = "nav_random_maze"

    # desired observations
    observation_list = ['RGBD_INTERLEAVED',
                        'RGB.LOOK_PANORAMA',
                        'RGB.LOOK_RANDOM',
                        'RGB.LOOK_TOP_DOWN',
                        'DEBUG.POS.TRANS',
                        'DEBUG.POS.ROT',
                        ]

    # configurations
    configurations = {
        'width': str(160),
        'height': str(160),
        "fps": str(60)
    }

    # maze sizes and seeds
    maze_size_list = [5]
    maze_seed_list = [0]

    # mapper
    size = random.sample(maze_size_list, 1)[0]
    seed = random.sample(maze_seed_list, 1)[0]
    env_map = mapper.RoughMap(size, seed, 3, False)

    # create the map environment
    myEnv = RandomMazeV1(level,
                         observation_list,
                         configurations)
    # initialize the maze environment
    maze_configs = defaultdict(lambda: None)
    maze_configs["maze_name"] = f"maze_{size}x{size}"
    maze_configs["maze_size"] = [size, size]
    maze_configs["maze_seed"] = '1234'
    maze_configs["maze_map_txt"] = "".join(env_map.map2d_txt)
    maze_configs["start_pos"] = env_map.init_pos + [0]
    maze_configs["goal_pos"] = env_map.goal_pos + [0]
    maze_configs["update"] = True
    state_obs, goal_obs, state_trans, state_rots = myEnv.reset(maze_configs)

    # create observation windows
    fig, arrs = plt.subplots(3, 3)
    state_obs = myEnv._goal_observation
    front_view = arrs[0, 1].imshow(state_obs[0])
    front_left_view = arrs[0, 0].imshow(state_obs[1])
    left_view = arrs[1, 0].imshow(state_obs[2])
    top_down_view = arrs[1, 1].imshow(ndimage.rotate(myEnv._top_down_obs, -90))
    back_left_view = arrs[2, 0].imshow(state_obs[3])
    back_view = arrs[2, 1].imshow(state_obs[4])
    back_right_view = arrs[2, 2].imshow(state_obs[5])
    right_view = arrs[1, 2].imshow(state_obs[6])
    front_right_view = arrs[0, 2].imshow(state_obs[7])

    # start test
    time_steps_num = 10000
    random.seed(maze_configs["maze_seed"])
    ep = 0

    # maze
    theme_list = ["TRON", "MINESWEEPER", "TETRIS", "GO", "PACMAN", "INVISIBLE_WALLS"]
    decal_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for t in range(time_steps_num):
        act = random.sample(range(4), 1)[0]
        next_obs, reward, done, dist, trans, rots, _ = myEnv.step(act)

        next_obs = myEnv._goal_observation
        front_view.set_data(next_obs[0])
        front_left_view.set_data(next_obs[1])
        left_view.set_data(next_obs[2])
        top_down_view.set_data(ndimage.rotate(myEnv._top_down_obs, -90))
        back_left_view.set_data(next_obs[3])
        back_view.set_data(next_obs[4])
        back_right_view.set_data(next_obs[5])
        right_view.set_data(next_obs[6])
        front_right_view.set_data(next_obs[7])
        fig.canvas.draw()
        plt.pause(0.0001)

        if done or t % 20 == 0:
            ep += 1
            # print("Ep = ", ep)
            if ep % 4 == 0:
                size = random.sample(maze_size_list, 1)[0]
                seed = random.sample(maze_seed_list, 1)[0]
                env_map = mapper.RoughMap(size, seed, 3, False)
                maze_configs["maze_name"] = f"maze_{size}x{size}"
                maze_configs["maze_size"] = [size, size]
                maze_configs["maze_seed"] = '1234'
                maze_configs["maze_map_txt"] = "".join(env_map.map2d_txt)
                maze_configs["start_pos"] = env_map.init_pos + [0]
                maze_configs["goal_pos"] = env_map.goal_pos + [0]
                maze_configs["maze_decal_freq"] = random.sample(decal_list, 1)[0]
                maze_configs["maze_texture"] = "MISHMASH"
                maze_configs["update"] = True
                myEnv.reset(maze_configs)
            else:
                maze_configs["update"] = False
                myEnv.reset(maze_configs)


if __name__ == '__main__':
    run_test()


