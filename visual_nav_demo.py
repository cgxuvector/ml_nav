import random
from utils import mapper
from envs.LabEnvV2 import RandomMazeTileRaw
from collections import defaultdict
import numpy as np
import IPython.terminal.debugger as Debug
import time
import matplotlib as plt
import cv2

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
    maze_size_list = [21]
    maze_seed_list = [0]
    use_true_state = False

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
    maze_configs["start_pos"] = [19, 1] + [0]  # start position on the txt map [rows, cols, orientation]
    maze_configs["goal_pos"] = [7, 15] + [0]  # goal position on the txt map [rows, cols, orientation]
    maze_configs["update"] = True  # update flag
    # set the maze
    state, goal, _, _ = myEnv.reset(maze_configs)

    fig = myEnv.show_panorama_view(None)

    out = cv2.VideoWriter('maze_21x21.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (900, 900))

    action_list = [0, 0, 3, 3, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0, 3, 0, 0, 0, 3, 3, 0, 0, 2, 2, 2, 2, 0, 0, 3, 3, 3, 3, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 0, 0, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1]
    for t in range(len(action_list)):
        action = action_list[t]
        next_state, r, done, dist, trans, _, _ = myEnv.step(action)

        map_start = myEnv.position_maze2map(trans, [21, 21])

        maze_configs["start_pos"] = map_start  # start position on the txt map [rows, cols, orientation]
        maze_configs["update"] = False  # update flag
        # set the maze
        state, goal, _, _ = myEnv.reset(maze_configs)

        fig = myEnv.show_panorama_view(t)
        print(ACTION_LIST_TILE[action_list[t]], print(trans))

        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with opencv or any operation you like
        # cv2.imshow("plot", img)
        print(img.shape)
        # cv2.waitKey(1)
        out.write(img)
    out.release()


if __name__ == '__main__':
    run_demo()

#
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.animation as manimation
#
# FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Movie Test', artist='Matplotlib',
#                 comment='Movie support!')
# writer = FFMpegWriter(fps=15, metadata=metadata)
#
# fig = plt.figure()
# l, = plt.plot([], [], 'k-o')
#
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
#
# x0, y0 = 0, 0
#
# with writer.saving(fig, "writer_test.mp4", 100):
#     for i in range(100):
#         x0 += 0.1 * np.random.randn()
#         y0 += 0.1 * np.random.randn()
#         l.set_data(x0, y0)
#         writer.grab_frame()