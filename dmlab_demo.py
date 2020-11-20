import random
from utils import mapper
from envs.LabEnvV2 import RandomMazeTileRaw
from envs.LabEnvV3 import RandomMazeTileRatio
from envs.LabEnvV4 import RandomMaze
from collections import defaultdict
import IPython.terminal.debugger as Debug
import time
import matplotlib.pyplot as plt
import argparse
import tqdm

ACTION_LIST_TILE = ['up', 'down', 'left', 'right']
ACTION_LIST_RAW = ['look left', 'look right', 'forward', 'backward']


def parse_input():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_type", type=str, default="discrete-raw", help="Type of the environment:"
                                                                             " Discrete (raw tile / ratio tile) "
                                                                             "or continuous")
    parser.add_argument("--env_run_mode", type=str, default='eval', help="Running mode: train or eval")
    parser.add_argument("--env_render", type=str, default='software', help="Type of the render. Hardware for GPU "
                                                                           "machine while Software for CPU machine")
    parser.add_argument("--test_maze_size", type=int, default=5, help="Size of the test maze.")
    parser.add_argument("--use_true_state", action="store_true", default=False, help="Whether use the true state or not")
    parser.add_argument("--goal_dist", type=int, default=-1, help="Map distance between the start and goal positions")
    parser.add_argument("--tile_ratio", type=float, default=0.4, help="Ratio of the tile after "
                                                                      "take one action [0.2, 0.4]")
    parser.add_argument("--start_radius", type=float, default=0, help="Sampling radius")
    parser.add_argument("--sample_repeat_count", type=int, default=0, help="Number of repeat for one sampled pair.")
    parser.add_argument("--terminal_dist", type=float, default=1e-3, help="Distance threshold for termination.")
    parser.add_argument("--total_time_steps", type=int, default=10000, help="Total time steps.")
    parser.add_argument("--episode_time_steps", type=int, default=100, help="Time steps for one episode.")

    return parser.parse_args()


def update_map2d_and_maze3d(args, env_3d, map_2d):
    """
    Function is used to update the 2D map and the 3D maze.
    """
    # set maze configurations
    maze_configs = defaultdict(lambda: None)
    map_2d.sample_start_goal_pos_with_random_dist(True, True)
    # initialize the maze 3D
    maze_configs["maze_name"] = f"maze_{args.test_maze_size}_0"  # string type name
    maze_configs["maze_size"] = [args.test_maze_size, args.test_maze_size]  # [int, int] list
    maze_configs["maze_seed"] = '1234'  # string type number
    maze_configs["maze_texture"] = "MISHMASH"  # string type name in theme_list
    maze_configs["maze_decal_freq"] = 0.001  # float number in decal_list
    maze_configs["maze_map_txt"] = "".join(map_2d.map2d_txt)  # string type map
    maze_configs["maze_valid_pos"] = map_2d.valid_pos  # list of valid positions
    # initialize the maze start and goal positions
    maze_configs["start_pos"] = map_2d.init_pos + [0]
    maze_configs["goal_pos"] = map_2d.goal_pos + [0]
    # initialize the update flag
    maze_configs["update"] = True  # update flag

    # obtain the state and goal observation
    if args.env_type == "discrete-raw":
        state_obs, goal_obs, _, _ = env_3d.reset(maze_configs)
    elif args.env_type == "discrete-ratio":
        state_obs, goal_obs, _, _ = env_3d.reset(maze_configs)
    else:
        state_obs, goal_obs, _, _, _, _ = env_3d.reset(maze_configs)

    return state_obs, goal_obs, map_2d.init_pos, map_2d.goal_pos


def run_env_continuous(args, env, map):
    """
    Demo function for raw DeepMind Lab:
    :param args: input args
    :param env: 3-D maze object
    :param map: 2-D map object
    :return: None

    Observation space: Panoramic observation containing 8 RGB images
    State space: [x, y, rot_z, velocity_x, velocity_y, velocity_rot_z]
    Action space: forward, backward, turn left, turn right

    This is the original DeepMind Lab. The only difference is that I do not use strafe actions.
    """
    # set the maze
    state, goal, _, _ = update_map2d_and_maze3d(args, env, map)

    # show the observation
    if not args.use_true_state:
        env.show_panorama_view(None, 'agent')

    episode_t = 0
    max_time_steps = 100
    last_trans = [9999, 9999, 9999]
    last_rots = [9999, 9999, 9999]
    for t in range(max_time_steps):
        # random sample an action
        action = random.sample(range(4), 1)[0]
        # step in the env
        next_state, reward, done, dist, trans, rots, _, _, _ = env.step(action)

        if not args.use_true_state:
            env.show_panorama_view(t, 'agent')

        # print the steps info
        if args.use_true_state:
            print(f"Step {episode_t}:"
                  f" state = {state[0:3]},"
                  f" act = {ACTION_LIST_RAW[action]},"
                  f" next_state = {next_state[0:3]},"
                  f" goal = {goal},"
                  f" done = {done}")
        else:
            print(f"Step {episode_t}:"
                  f" state = {last_trans[0:2] + last_rots[1:2]},"
                  f" act = {ACTION_LIST_RAW[action]},"
                  f" next_state = {trans[0:2] + rots[1:2]},"
                  f" goal = {env._goal_state},"
                  f" done = {done}")

        # if done
        if done:
            # reset the environment
            state, goal, _, _ = update_map2d_and_maze3d(args, env, map)
        else:
            episode_t += 1
            state = next_state
            last_trans = trans
            last_rots = rots


def run_env_discrete_ratio(args, env, map):
    """
    Demo function for discrete ratio DeepMind Lab:
    :param args: input args
    :param env: 3-D maze object
    :param map: 2-D map object
    :return: None

    Observation space: Panoramic observation containing 8 RGB images
    State space: [x, y]
    Action space: up, down, left, right

    This is the discrete ratio version of DeepMind Lab. One action will move tile_ratio * 100 of a tile.
    This version is a little bit difficult than the discrete raw version.
    """
    # set the maze
    state, goal, _, _ = update_map2d_and_maze3d(args, env, map)

    # show the observation
    if not args.use_true_state:
        env.show_panorama_view(None, 'agent')

    episode_t = 0
    max_time_steps = 100
    last_trans = None
    for t in range(max_time_steps):
        # random sample an action
        action = random.sample(range(4), 1)[0]
        # step in the env
        next_state, reward, done, dist, trans, rots, _ = env.step(action)

        if not args.use_true_state:
            env.show_panorama_view(t, 'agent')

        # print the steps info
        if args.use_true_state:
            print(f"Step {episode_t}:"
                  f" state = {state},"
                  f" act = {ACTION_LIST_TILE[action]},"
                  f" next_state = {next_state},"
                  f" goal = {goal},"
                  f" done = {done}")
        else:
            print(f"Step {episode_t}:"
                  f" state = {last_trans},"
                  f" act = {ACTION_LIST_TILE[action]},"
                  f" next_state = {trans},"
                  f" goal = {env.goal_trans},"
                  f" done = {done}")

        # if done
        if done:
            # reset the environment
            state, goal, _, _ = update_map2d_and_maze3d(args, env, map)
        else:
            episode_t += 1
            state = next_state
            last_trans = trans


def run_env_discrete_raw(args, env, map):
    """
    Demo function for discrete ratio DeepMind Lab:
    :param args: input args
    :param env: 3-D maze object
    :param map: 2-D map object
    :return: None

    Observation space: Panoramic observation containing 8 RGB images
    State space: [x, y]
    Action space: up, down, left, right

    This is the discrete version. The 3-D map is same as the 2-D map. The only difficult comes from the observation
    encoding.
    """
    # set the maze
    state, goal, _, _ = update_map2d_and_maze3d(args, env, map)

    # show the observation
    if not args.use_true_state:
        env.show_panorama_view(None, 'agent')

    episode_t = 0
    max_time_steps = 200
    last_trans = None
    for t in range(max_time_steps):
        # random sample an action
        action = random.sample(range(4), 1)[0]
        # step in the env
        next_state, reward, done, dist, trans, rots, _ = env.step(action)

        if not args.use_true_state:
            env.show_panorama_view(t, 'agent')

        # print the steps info
        if args.use_true_state:
            print(f"Step {episode_t}:"
                  f" state = {state},"
                  f" act = {ACTION_LIST_TILE[action]},"
                  f" next_state = {next_state},"
                  f" goal = {goal},"
                  f" done = {done}")
        else:
            print(f"Step {episode_t}:"
                  f" state = {last_trans},"
                  f" act = {ACTION_LIST_TILE[action]},"
                  f" next_state = {trans},"
                  f" goal = {env.goal_pos},"
                  f" done = {done}")

        # if done
        if done:
            # reset the environment
            state, goal, _, _ = update_map2d_and_maze3d(args, env, map)
        else:
            episode_t += 1
            state = next_state
            last_trans = trans


def run_demo():
    # Parse the input
    inputs = parse_input()

    # Set level name
    # This level uses pre-built bsp files to generate the 3-D mazes.
    # The bsp files are pre-built, I can't change the attributes of
    # the 3-D mazes, such as texture or decal frequency.
    level = "nav_random_maze_tile_bsp"

    # set desired observations
    if inputs.env_type == "continuous":
        observation_list = ['RGB.LOOK_RANDOM_PANORAMA_VIEW',
                            'RGB.LOOK_PANORAMA_VIEW',
                            'RGB.LOOK_TOP_DOWN_VIEW',
                            'DEBUG.POS.TRANS',
                            'DEBUG.POS.ROT',
                            'VEL.TRANS',
                            'VEL.ROT'
                            ]
    elif inputs.env_type == "discrete-ratio":
        observation_list = ['RGB.LOOK_RANDOM_PANORAMA_VIEW',
                            'RGB.LOOK_PANORAMA_VIEW',
                            'RGB.LOOK_TOP_DOWN_VIEW'
                            ]
    elif inputs.env_type == "discrete-raw":
        observation_list = ['RGB.LOOK_RANDOM_PANORAMA_VIEW',
                            'RGB.LOOK_PANORAMA_VIEW',
                            'RGB.LOOK_TOP_DOWN_VIEW']
    else:
        assert f"Wrong environment type, expect discrete-raw, discrete-ratio or continuous but get {inputs.env_type}"
        return False

    # set observation configurations
    configurations = {
        'width': str(64),
        'height': str(64),
        "fps": str(60)
    }

    # initialize the 3-D maze
    # default maze seed is 0
    size = inputs.test_maze_size
    seed = 0

    # initialize the 2-D map
    env_map = mapper.RoughMap(size, seed, 3, False)

    # create the map environment and run a demo
    # when use_true_state is disabled, the panoramic
    # observations will be shown
    if inputs.env_type == "discrete-ratio":  # discrete DeepMind Lab with [0.2, 0.4] ratio, set with --tile_ratio
        my_env = RandomMazeTileRatio(level,
                                     observation_list,
                                     configurations,
                                     args=inputs,
                                     reward_type='sparse-1',
                                     dist_epsilon=inputs.terminal_dist)
        run_env_discrete_ratio(inputs, my_env, env_map)
    elif inputs.env_type == "discrete-raw":  # discrete DeepMind Lab with tile version. One action moving one tile
        my_env = RandomMazeTileRaw(level,
                                   observation_list,
                                   configurations,
                                   args=inputs,
                                   reward_type='sparse-1',
                                   dist_epsilon=inputs.terminal_dist)
        run_env_discrete_raw(inputs, my_env, env_map)
    elif inputs.env_type == "continuous":  # continuous DeepMInd Lab.
        my_env = RandomMaze(level,
                            observation_list,
                            configurations,
                            args=inputs,
                            reward_type='sparse-1',
                            dist_epsilon=inputs.terminal_dist)
        run_env_continuous(inputs, my_env, env_map)
    else:
        return False


if __name__ == '__main__':
    run_demo()


