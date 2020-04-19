import random
from utils import mapper
from envs.LabEnvV1 import RandomMazeV1
from collections import defaultdict
import IPython.terminal.debugger as Debug


def run_demo():
    # level name
    level = "nav_random_maze"

    # desired observations
    observation_list = [
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
    maze_size_list = [5, 7, 9]
    maze_seed_list = [1, 2, 3]

    # maze
    theme_list = ["TRON", "MINESWEEPER", "TETRIS", "GO", "PACMAN", "INVISIBLE_WALLS"]
    decal_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
    maze_configs["maze_name"] = f"maze_{size}x{size}"  # string type name
    maze_configs["maze_size"] = [size, size]  # [int, int] list
    maze_configs["maze_seed"] = '1234'  # string type number
    maze_configs["maze_texture"] = random.sample(theme_list, 1)[0]  # string type name in theme_list
    maze_configs["maze_decal_freq"] = random.sample(decal_list, 1)[0]  # float number in decal_list
    maze_configs["maze_map_txt"] = "".join(env_map.map2d_txt)  # string type map
    # initialize the maze start and goal positions
    maze_configs["start_pos"] = env_map.init_pos + [0]  # start position on the txt map [rows, cols, orientation]
    maze_configs["goal_pos"] = env_map.goal_pos + [0]  # goal position on the txt map [rows, cols, orientation]
    maze_configs["update"] = True  # update flag
    # set the maze
    myEnv.reset(maze_configs)

    # # create observation windows
    # myEnv._last_observation = myEnv.get_random_observations(myEnv.position_map2maze([1, 3, 0], myEnv.maze_size))
    myEnv.show_panorama_view()
    # myEnv.show_front_view()

    # start test
    time_steps_num = 10000
    random.seed(maze_configs["maze_seed"])
    ep = 0
    pos_len = 1
    for t in range(time_steps_num):
        # sample an action
        act = random.sample(range(4), 1)[0]
        # agent takes the action
        myEnv.step(act)

        # for a random maze view
        # myEnv._last_observation = myEnv.get_random_observations(myEnv.position_map2maze([1, 3, 0], myEnv.maze_size))
        # for the panorama view
        myEnv.show_panorama_view(t)
        # for the front view
        # myEnv.show_front_view(t)

        # test episode length is 20
        if t % 20 == 0:
            ep += 1
            # reset the whole maze after 10 episodes
            if ep % 10 == 0:
                # randomly sample a maze
                size = random.sample(maze_size_list, 1)[0]
                seed = random.sample(maze_seed_list, 1)[0]
                env_map = mapper.RoughMap(size, seed, 3, False)
                # set the new maze params
                maze_configs["maze_name"] = f"maze_{size}x{size}"
                maze_configs["maze_size"] = [size, size]
                maze_configs["maze_seed"] = '1234'
                maze_configs["maze_map_txt"] = "".join(env_map.map2d_txt)
                maze_configs["start_pos"] = env_map.init_pos + [0]
                maze_configs["goal_pos"] = env_map.goal_pos + [0]
                maze_configs["maze_decal_freq"] = random.sample(decal_list, 1)[0]
                maze_configs["maze_texture"] = random.sample(theme_list, 1)[0]
                maze_configs["update"] = True
                # set the maze
                myEnv.reset(maze_configs)
                pos_len = 1
            else:  # for a fixed maze, sequentially set the start along the valid positions on the map
                # e.g. 5x5 maze txt
                #    ---------> cols
                #    | * * * * *
                #    | * P     *
                #    | *   * * *
                #    | *     G *
                #    | * * * * *
                #   rows
                #     P = (1, 1)
                #     G = (3, 3)
                #    type = [rows, cols, orientation]
                #    start_pos = (1, 1, 0)
                #    goal_pos = (3, 3, 0)
                #   where 0 is the orientation in [0, 360]
                maze_configs["start_pos"] = env_map.valid_pos[pos_len] + [0]
                pos_len = pos_len + 1 if pos_len + 1 < len(env_map.valid_pos) else len(env_map.valid_pos) - 1
                maze_configs["goal_pos"] = env_map.goal_pos + [0]
                maze_configs["update"] = False
                myEnv.reset(maze_configs)


if __name__ == '__main__':
    run_demo()


