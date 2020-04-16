import deepmind_lab
import numpy as np
import random
import matplotlib.pyplot as plt


def _action(*entries):
    return np.array(entries, dtype=np.intc)


# actions in Deepmind
ACTION_LIST = [
        _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
        _action(20, 0, 0, 0, 0, 0, 0),  # look_right
        _action(0, 0, 0, 1, 0, 0, 0),  # forward
        _action(0, 0, 0, -1, 0, 0, 0),  # backward
        _action(0, 0, 0, 0, 0, 0, 0)  # NOOP
]


if __name__ == "__main__":
    # set environment configurations
    level_name = "random_customized_maze"
    observations = ['RGBD_INTERLEAVED', "RGB.LOOK_TOP_DOWN"]
    configurations = {'fps': str(60),
                      "width": str(320),
                      "height": str(320)
                      }

    # create the environment
    myEnv = deepmind_lab.Lab(level_name,
                             observations,
                             configurations
                             )

    # create episode
    ep = 0
    myEnv.reset(episode=ep)
    # state_obs = myEnv.observations()["RGBD_INTERLEAVED"]
    state_obs = myEnv.observations()["RGB.LOOK_TOP_DOWN"]
    fig, arr = plt.subplots(1, 1)
    img = arr.imshow(state_obs)

    theme_list = ["TRON", "MINESWEEPER", "TETRIS", "GO", "PACMAN", "INVISIBLE_WALLS"]

    # set testing params
    total_time_steps = 10000
    env_theme = 0
    for i in range(total_time_steps):
        # act = random.sample(ACTION_LIST, 1)[0]
        act = ACTION_LIST[-1]
        myEnv.step(act, num_steps=4)
        # state_obs = myEnv.observations()["RGBD_INTERLEAVED"]
        state_obs = myEnv.observations()["RGB.LOOK_TOP_DOWN"]
        img.set_data(state_obs)
        fig.canvas.draw()
        plt.pause(0.0001)

        if i % 10 == 0:
            ep += 1
            if ep % 10 == 0:
                myEnv.write_property("params._map_texture", theme_list[env_theme])
                myEnv.write_property("params.maze_set.size", str(7))
                myEnv.write_property("params.goal_pos.x", str(4))
                myEnv.write_property("params.goal_pos.y", str(4))
                myEnv.write_property("params._map_txt", "*******\n* P   *\n*   ***\n*    G*\n** ** *\n*     *\n******")
                myEnv.write_property("params._map_decal_freq", str(0.5))
                myEnv.reset(episode=0)
                env_theme += 1
            else:
                myEnv.reset()

        if not myEnv.is_running():
            break