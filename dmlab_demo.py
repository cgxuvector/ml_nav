import random
from utils import mapper
from envs.LabEnvV3 import RandomMazeTileRaw
from collections import defaultdict
import IPython.terminal.debugger as Debug
import time
import matplotlib.pyplot as plt


ACTION_LIST_TILE = ['up', 'down', 'left', 'right']


def run_demo():
    # level name
    level = "nav_random_maze_tile_bsp"

    # desired observations
    observation_list = ['RGB.LOOK_RANDOM_PANORAMA_VIEW',
                        'RGB.LOOK_TOP_DOWN_VIEW'
                        ]

    # configurations
    configurations = {
        'width': str(128),
        'height': str(128),
        "fps": str(60)
    }

    # maze sizes and seeds
    maze_size_list = [7]
    maze_seed_list = [0]
    use_true_state = True

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
                              configurations,
                              use_true_state=use_true_state)

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
    maze_configs["goal_pos"] = [5, 5] + [0]  # goal position on the txt map [rows, cols, orientation]
    maze_configs["update"] = True  # update flag
    # set the maze
    state, goal, _, _ = myEnv.reset(maze_configs)
    print(f"Start = {myEnv.start_pos}, Goal = {myEnv.goal_pos}")

    if not use_true_state:
        myEnv.show_panorama_view_test(None, state)

    if not use_true_state:
        last_trans = myEnv._trans

    # maximal time steps
    success_count = 0
    action_list = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    for t in range(len(action_list)):
        action = action_list[t]
        next_state, r, done, dist, trans, _, _ = myEnv.step(action)
        if not use_true_state:
            print(f"Step = {t}, current_pos={last_trans}, action={ACTION_LIST_TILE[action]}, current_pos={trans}")
        else:
            print(f"Step = {t}, current_pos={state}, action={ACTION_LIST_TILE[action]}, next_pos={next_state}")

        if not use_true_state:
            myEnv.show_panorama_view_test(1, next_state)
            last_trans = trans
        state = next_state
        if done:
            success_count += 1
        # if done or t % 10 == 0:
        #     total_count += 1
        #     start = time.time()
        #     maze_configs = defaultdict(lambda: None)
        #     init_pos, goal_pos = env_map.sample_random_start_goal_pos(True, True, 2)
        #     # self.env_map.update_mapper(init_pos, goal_pos)
        #     maze_configs['start_pos'] = init_pos + [0]
        #     maze_configs['goal_pos'] = goal_pos + [0]
        #     maze_configs['maze_valid_pos'] = env_map.valid_pos
        #     maze_configs['update'] = True
        #     myEnv.reset(maze_configs)
        #     reset_time.append(time.time() - start)
        #     print("Same maze reset = {}".format(time.time() - start))
    #
    # print("Mean step time = {}".format(sum(step_time) / len(step_time)))
    # print("Mean reset time = {}".format(sum(reset_time) / len(reset_time)))
    # print("Success rate = {}".format(success_count/total_count))


if __name__ == '__main__':
    run_demo()


