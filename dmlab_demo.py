import random
from utils import mapper
from envs.LabEnvV4 import RandomMaze
from collections import defaultdict
import IPython.terminal.debugger as Debug
import time
import matplotlib.pyplot as plt


ACTION_LIST_TILE = ['up', 'down', 'left', 'right']
ACTION_LIST_RAW = ['look left', 'look right', 'forward', 'backward']


def run_demo():
    # level name
    level = "nav_random_maze_tile_bsp"

    # desired observations
    observation_list = ['RGB.LOOK_RANDOM_PANORAMA_VIEW',
                        'RGB.LOOK_PANORAMA_VIEW',
                        'RGB.LOOK_TOP_DOWN_VIEW',
                        'DEBUG.POS.TRANS',
                        'DEBUG.POS.ROT'
                        ]

    # configurations
    configurations = {
        'width': str(64),
        'height': str(64),
        "fps": str(60)
    }

    # maze sizes and seeds
    use_true_state = False

    # maze
    theme_list = ['MISHMASH']
    decal_list = [0.001]

    # mapper
    size = 7
    seed = 0
    env_map = mapper.RoughMap(size, seed, 3, False)
    # create the map environment
    myEnv = RandomMaze(level,
                       observation_list,
                       configurations,
                       use_true_state=use_true_state,
                       reward_type='sparse-1')

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
    maze_configs["start_pos"] = [1, 1] + [0]  # start position on the txt map [rows, cols, orientation]
    maze_configs["goal_pos"] = [2, 1] + [0]  # goal position on the txt map [rows, cols, orientation]
    maze_configs["update"] = True  # update flag
    # set the maze
    state, goal, _, _ = myEnv.reset(maze_configs)

    if not use_true_state:
        myEnv.show_panorama_view(None, 'agent')

    steps = 1000
    for t in range(steps):
        action = random.sample(range(4), 1)[0]
        next_state, reward, done, dist, trans, rots, _ = myEnv.step(action)
        if use_true_state:
            print(f"Step = {t}, current_pos={state}, action={ACTION_LIST_RAW[action]}, current_pos={next_state}, reward={reward}, done={done}, goal={goal}")
        state = next_state
        if not use_true_state:
            myEnv.show_panorama_view(state, 'agent')
        if done:
            break

    # maximal time steps
    # success_count = 0
    # action_list = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    # for t in range(len(action_list)):
    #     action = action_list[t]
    #     next_state, r, done, dist, trans, _, _ = myEnv.step(action)
    #     if not use_true_state:
    #         print(f"Step = {t}, current_pos={last_trans}, action={ACTION_LIST_TILE[action]}, current_pos={trans}, reward={r}, done={done}")
    #     else:
    #         print(f"Step = {t}, current_pos={state}, action={ACTION_LIST_TILE[action]}, next_pos={next_state}, reward={r}, done={done}")
    #
    #     if not use_true_state:
    #         myEnv.show_panorama_view_test(1, next_state)
    #         last_trans = trans
    #     state = next_state
    #     if done:
    #         success_count += 1


if __name__ == '__main__':
    run_demo()


