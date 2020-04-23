import random
import numpy as np
from utils import mapper
from envs.LabEnvV1 import RandomMazeV1
from envs.LabEnvV2 import RandomMazeTileRaw
from collections import defaultdict
from agent import RandomAgent
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
        'width': str(160),
        'height': str(160),
        "fps": str(60)
    }

    # maze sizes and seeds
    maze_size_list = [5, 7, 9, 11, 13]
    maze_seed_list = [1, 2, 3, 4, 5, 6, 7]

    # maze
    # theme_list = ["TRON", "MINESWEEPER", "TETRIS", "GO", "PACMAN", "INVISIBLE_WALLS"]
    theme_list = ['INVISIBLE_WALLS']
    decal_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
    myEnv.reset(maze_configs)

    # create observation windows
    # fig = myEnv.show_panorama_view(obs_type='agent')
    myEnv.show_front_view()

    # start test
    time_steps_num = 10000
    random.seed(maze_configs["maze_seed"])
    actions = ['up', 'down', 'left', 'right']

    for i in range(20):
        myEnv._lab.step(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.intc))

    act_idx = 0
    for t in range(time_steps_num):

        # select one action
        # action = np.random.choice(actions, 1).item()
        action = env_map.map_act[act_idx]
        act_idx += 1

        # take one step
        _, reward, done, dist, _, _, _ = myEnv.step(action)

        print("Step = {}, Action = {}, Reward = {}, dist = {}, done = {}".format(t+1, action, reward, dist, done))

        # for the panorama view
        # fig = myEnv.show_panorama_view(time_step=t, obs_type='agent')
        # for the front view
        myEnv.show_front_view(time_step=t)

        if done or t + 1 % 100 == 0:
            # randomly sample a maze
            size = random.sample(maze_size_list, 1)[0]
            seed = random.sample(maze_seed_list, 1)[0]
            env_map = mapper.RoughMap(size, seed, 3, False)
            # set the new maze params
            maze_configs["maze_name"] = f"maze_{size}x{size}"
            maze_configs["maze_size"] = [size, size]
            maze_configs["maze_seed"] = '1234'
            maze_configs["maze_map_txt"] = "".join(env_map.map2d_txt)
            maze_configs["maze_decal_freq"] = random.sample(decal_list, 1)[0]
            maze_configs["maze_texture"] = random.sample(theme_list, 1)[0]
            maze_configs["maze_valid_pos"] = env_map.valid_pos
            # set the start and goal positions
            maze_configs["start_pos"] = env_map.init_pos + [0]
            maze_configs["goal_pos"] = env_map.goal_pos + [0]
            maze_configs["update"] = True
            # reset the maze
            myEnv.reset(maze_configs)
            act_idx = 0
            for i in range(100):
                myEnv._lab.step(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.intc))


        # # reset the whole maze after 10 episodes
        # if pos_len == len(env_map.path) - 1:
        #     # randomly sample a maze
        #     size = random.sample(maze_size_list, 1)[0]
        #     seed = random.sample(maze_seed_list, 1)[0]
        #     env_map = mapper.RoughMap(size, seed, 3, False)
        #     # set the new maze params
        #     maze_configs["maze_name"] = f"maze_{size}x{size}"
        #     maze_configs["maze_size"] = [size, size]
        #     maze_configs["maze_seed"] = '1234'
        #     maze_configs["maze_map_txt"] = "".join(env_map.map2d_txt)
        #     maze_configs["maze_decal_freq"] = random.sample(decal_list, 1)[0]
        #     maze_configs["maze_texture"] = random.sample(theme_list, 1)[0]
        #     maze_configs["maze_valid_pos"] = env_map.valid_pos
        #     # set the start and goal positions
        #     maze_configs["start_pos"] = env_map.init_pos + [0]
        #     maze_configs["goal_pos"] = env_map.goal_pos + [0]
        #     maze_configs["update"] = True
        #     # reset the maze
        #     myEnv.reset(maze_configs)
        #     pos_len = 1
        #     break
        # else:  # for a fixed maze, sequentially set the start along the valid positions on the map
        #     # e.g. 5x5 maze txt
        #     #    ---------> cols
        #     #    | * * * * *
        #     #    | * P     *
        #     #    | *   * * *
        #     #    | *     G *
        #     #    | * * * * *
        #     #   rows
        #     #     P = (1, 1)
        #     #     G = (3, 3)
        #     #    type = [rows, cols, orientation]
        #     #    start_pos = (1, 1, 0)
        #     #    goal_pos = (3, 3, 0)
        #     #   where 0 is the orientation in [0, 360]
        #     maze_configs["start_pos"] = list(env_map.path[pos_len]) + [0]
        #     pos_len = pos_len + 1 if pos_len + 1 < len(env_map.path) else len(env_map.path) - 1
        #     # maze_configs["goal_pos"] = env_map.goal_pos + [0]
        #     myEnv._goal_observation = myEnv.get_random_observations(maze_configs['start_pos'])
        #     maze_configs["update"] = False
        #     # myEnv.step(4)
        #     print(maze_configs['start_pos'], maze_configs['goal_pos'])
        #     # myEnv.reset(maze_configs)


if __name__ == '__main__':
    run_demo()


