import random
import numpy as np
from utils import mapper
from envs.LabEnvV2 import RandomMazeTileRaw
from envs.LabEnvV1 import RandomMazeV1
from collections import defaultdict
import IPython.terminal.debugger as Debug
import time
import matplotlib.pyplot as plt


def run_demo():
    # level name
    level = "nav_random_maze_tile_bsp"

    # desired observations
    observation_list = ['RGB.LOOK_RANDOM_PANORAMA_VIEW',
                        'RGB.LOOK_TOP_DOWN_VIEW'
                        ]
    # observation_list = ['RGB.LOOK_PANORAMA_VIEW',
    #                     'RGB.LOOK_TOP_DOWN_VIEW'
    #                     ]

    # configurations
    configurations = {
        'width': str(32),
        'height': str(32),
        "fps": str(60)
    }

    # maze sizes and seeds
    maze_size_list = [5, 7, 9, 11, 13, 15, 17, 19, 21]
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
    # myEnv = RandomMazeV1(level,
    #                      observation_list,
    #                      configurations)

    # initialize the maze environment
    maze_configs = defaultdict(lambda: None)
    maze_configs["maze_name"] = f"maze_{size}_{seed}"  # string type name
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
    start = time.time()
    state, _, _, _ = myEnv.reset(maze_configs)
    print("New maze reset = {}".format(time.time() - start))

    # maximal time steps
    max_time_steps = 1000
    step_time = []
    reset_time = []
    success_count = 0
    total_count = 0
    for t in range(max_time_steps):
        action = random.sample(range(4), 1)[0]
        start = time.time()
        next_state, r, done, dist, _, _, _ = myEnv.step(action)
        step_time.append(time.time() - start)
        if done:
            success_count += 1
        if done or t % 10 == 0:
            total_count += 1
            start = time.time()
            maze_configs = defaultdict(lambda: None)
            init_pos, goal_pos = env_map.sample_random_start_goal_pos(True, True, 2)
            # self.env_map.update_mapper(init_pos, goal_pos)
            maze_configs['start_pos'] = init_pos + [0]
            maze_configs['goal_pos'] = goal_pos + [0]
            maze_configs['maze_valid_pos'] = env_map.valid_pos
            maze_configs['update'] = True
            myEnv.reset(maze_configs)
            reset_time.append(time.time() - start)
            print("Same maze reset = {}".format(time.time() - start))

    print("Mean step time = {}".format(sum(step_time) / len(step_time)))
    print("Mean reset time = {}".format(sum(reset_time) / len(reset_time)))
    print("Success rate = {}".format(success_count/total_count))


if __name__ == '__main__':
    run_demo()


