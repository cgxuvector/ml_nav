import random
from utils import mapper
from envs.LabEnvV1 import RandomMazeV1
from collections import defaultdict
import IPython.terminal.debugger as Debug


def run_test():
    # level name
    level = "nav_random_maze"

    # desired observations
    observation_list = ["RGBD_INTERLEAVED",
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
    maze_configs["maze_name"] = f"maze_{size}x{size}"
    maze_configs["maze_size"] = [size, size]
    maze_configs["maze_seed"] = '1234'
    maze_configs["maze_map_txt"] = "".join(env_map.map2d_txt)
    maze_configs["start_pos"] = env_map.init_pos + [0]
    maze_configs["goal_pos"] = env_map.goal_pos + [0]
    maze_configs["maze_texture"] = "INVISIBLE_WALLS"
    maze_configs["update"] = True
    myEnv.reset(maze_configs)

    # # create observation windows
    # myEnv._last_observation = myEnv.get_random_observations(myEnv.position_map2maze([1, 3, 0], myEnv.maze_size))
    # myEnv.show_panorama_view()
    myEnv.show_front_view()

    # start test
    time_steps_num = 10000
    random.seed(maze_configs["maze_seed"])
    ep = 0

    pos_len = 1
    for t in range(time_steps_num):
        act = random.sample(range(4), 1)[0]
        myEnv.step(act)

        # myEnv._last_observation = myEnv.get_random_observations(myEnv.position_map2maze([1, 3, 0], myEnv.maze_size))
        # myEnv.show_panorama_view(t)
        myEnv.show_front_view(t)

        if t % 20 == 0:
            ep += 1
            # print("Ep = ", ep)
            if ep % 10 == 0:
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
                maze_configs["maze_texture"] = random.sample(theme_list, 1)[0]
                maze_configs["update"] = True
                pos_len = 1
                myEnv.reset(maze_configs)
            else:
                maze_configs["start_pos"] = env_map.valid_pos[pos_len] + [0]
                pos_len = pos_len + 1 if pos_len + 1 < len(env_map.valid_pos) else len(env_map.valid_pos) - 1
                maze_configs["goal_pos"] = env_map.goal_pos + [0]
                maze_configs["update"] = False
                myEnv.reset(maze_configs)


if __name__ == '__main__':
    run_test()


