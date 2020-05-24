import random
from utils import mapper
from envs.LabEnvV2 import RandomMazeTileRaw
from collections import defaultdict
import os
import sys
import shutil
import fnmatch
import IPython.terminal.debugger as Debug
import time
import matplotlib.pyplot as plt


def run_demo():
    # level name
    level = "nav_random_maze_tile"

    # desired observations
    observation_list = ['RGB.LOOK_RANDOM_PANORAMA_VIEW',
                        'RGB.LOOK_TOP_DOWN_VIEW'
                        ]
    # configurations
    configurations = {
        'width': str(32),
        'height': str(32),
        "fps": str(60)
    }

    # maze sizes and seeds
    maze_size_list = [5, 7, 9, 11, 13, 15, 17, 19, 21]
    maze_seed_list = list(range(20))

    # maze
    theme_list = ['MISHMASH']
    decal_list = [0.001]

    # create the map environment
    myEnv = RandomMazeTileRaw(level,
                              observation_list,
                              configurations)

    # check the directories
    if os.path.exists("./precompiled_map"):
        shutil.rmtree("./precompiled_map")
    else:
        os.mkdir("./precompiled_map")

    # precompile map
    for size in maze_size_list:
        for seed in maze_seed_list:
            env_map = mapper.RoughMap(size, seed, 3, False)
            maze_name = f"maze_{size}_{seed}"
            # initialize the maze environment
            maze_configs = defaultdict(lambda: None)
            maze_configs["maze_name"] = maze_name  # string type name
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

            # save the .pk3 file at /tmp/dmlab_temp_folder_*/baselab/*
            dir_names = os.listdir("/tmp/")
            lab_dir_name = fnmatch.filter(dir_names, "dmlab_temp_folder_*")[0]
            source_dir = f'/tmp/{lab_dir_name}/baselab/{maze_name}.pk3'
            target_dir = f'./precompiled_map/{maze_name}.pk3'
            print(source_dir)
            print(target_dir)
            shutil.move(source_dir, target_dir)
            print("--------- File moved ---------")
            # extract the pk3 file
            os.system(f"unzip -d ./precompiled_map/ ./precompiled_map/{maze_name}.pk3")
            print("Finish extraction")


if __name__ == '__main__':
    run_demo()


