import random
from utils import mapper
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

    parser.add_argument("--env_run_mode", type=str, default='eval', help="Mode of running the envrionment")
    parser.add_argument("--env_render", type=str, default='software', help="Type of the render. Hardware for GPU machine while Software for CPU machine")
    parser.add_argument("--start_radius", type=float, default=0, help="Sampling radius")
    parser.add_argument("--sample_repeat_count", type=int, default=0, help="Number of repeat for one sampled pair.")
    parser.add_argument("--terminal_dist", type=float, default=1e-3, help="Distance threshold for termination.")
    parser.add_argument("--total_time_steps", type=int, default=10000, help="Total time steps.")
    parser.add_argument("--episode_time_steps", type=int, default=100, help="Time steps for one episode.")

    return parser.parse_args()

def run_demo():
    inputs = parse_input()

    # level name
    level = "nav_random_maze_tile_bsp"

    # desired observations
    observation_list = ['RGB.LOOK_RANDOM_PANORAMA_VIEW',
                        'RGB.LOOK_PANORAMA_VIEW',
                        'RGB.LOOK_TOP_DOWN_VIEW',
                        'DEBUG.POS.TRANS',
                        'DEBUG.POS.ROT',
                        'VEL.TRANS',
                        'VEL.ROT'
                        ]

    # configurations
    configurations = {
        'width': str(32),
        'height': str(32),
        "fps": str(60)
    }

    # maze sizes and seeds
    use_true_state = False

    # maze
    theme_list = ['MISHMASH']
    decal_list = [0.001]

    # mapper
    size = 21
    seed = 0
    env_map = mapper.RoughMap(size, seed, 3, False)
    # create the map environment
    myEnv = RandomMaze(level,
                       observation_list,
                       configurations,
                       args=inputs,
                       use_true_state=use_true_state,
                       reward_type='sparse-1',
                       dist_epsilon=inputs.terminal_dist)

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
    maze_configs["goal_pos"] = [19, 19] + [0]  # goal position on the txt map [rows, cols, orientation]
    maze_configs["update"] = True  # update flag
    # set the maze
    state, goal, _, _ , _, _= myEnv.reset(maze_configs)

    #if not use_true_state:
    #    myEnv.show_panorama_view(None, 'agent')

    episode_t = 0
    episode_c = 0
    pbar = tqdm.trange(inputs.total_time_steps)
    for t in pbar:
        action = random.sample(range(4), 1)[0]
        next_state, reward, done, dist, trans, rots, _, _, _ = myEnv.step(action)
        #if use_true_state:
        #    print(f"Step = {t}, current_pos={state}, action={ACTION_LIST_RAW[action]}, current_pos={next_state}, reward={reward}, done={done}, goal={goal}")
        state = next_state
        episode_t += 1
        #if not use_true_state:
        #    myEnv.show_panorama_view(state, 'agent')
        if done or episode_t == inputs.episode_time_steps:
            # count one episode
            episode_c += 1
            # show the information
            pbar.set_description(
                    f'Episode: {episode_c}'
                    f'Steps: {episode_t}'
                    f'Done: {done}'
            )
            
            # reset the environment
            maze_configs["update"] = False
            maze_configs["start_pos"] = [1, 1, 0]
            maze_configs["goal_pos"] = [19, 19, 0]
            episode_t = 0
            state, goal, _, _, _, _ = myEnv.reset(maze_configs)

if __name__ == '__main__':
    run_demo()


