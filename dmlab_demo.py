import random
import numpy as np
from utils import mapper
from envs.LabEnvV2 import RandomMazeTileRaw
from collections import defaultdict
import IPython.terminal.debugger as Debug
import matplotlib.pyplot as plt


def run_demo():
    # level name
    level = "nav_random_maze"

    # desired observations
    observation_list = ['RGBD_INTERLEAVED',
                        'RGB.LOOK_PANORAMA_VIEW',
                        'RGB.LOOK_TOP_DOWN_VIEW'
                        ]

    # configurations
    configurations = {
        'width': str(84),
        'height': str(84),
        "fps": str(60)
    }

    # maze sizes and seeds
    maze_size_list = [5]
    maze_seed_list = [0]

    # maze
    theme_list = ['MISHMASH']
    decal_list = [0.001]

    # mapper
    size = random.sample(maze_size_list, 1)[0]
    seed = random.sample(maze_seed_list, 1)[0]
    env_map = mapper.RoughMap(size, seed, 3, False)

    # create the map environment
    myEnv = RandomMazeTileRaw(level,
                              observation_list,
                              configurations)

    # initialize the maze environment
    maze_configs = defaultdict(lambda: None)
    maze_configs["maze_name"] = f"maze_{size}x{size}"  # string type name
    maze_configs["maze_size"] = [size, size]  # [int, int] list
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

    # create observation windows
    myEnv.show_panorama_view_test(None, state)

    for r in range(5):
        for idx, pos in enumerate(env_map.path):
            print(r, ' - ', idx, ' - ', pos)
            state = myEnv.get_random_observations(pos.tolist() + [0])
            Debug.set_trace()
            myEnv.show_panorama_view_test(1, state)
        init_pos, goal_pos = env_map.sample_random_start_goal_pos(False, False, 5)
        print(init_pos, goal_pos, env_map.path)
        maze_configs["start_pos"] = init_pos + [0]
        maze_configs["goal_pos"] = goal_pos + [0]
        maze_configs["update"] = False
        myEnv.reset(maze_configs)


if __name__ == '__main__':
    run_demo()


