import deepmind_lab
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import IPython.terminal.debugger as Debug


def _action(*entries):
    return np.array(entries, dtype=np.intc)


# actions in Deepmind
ACTION_LIST = [
        _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
        _action(20, 0, 0, 0, 0, 0, 0),  # look_right
        _action(0, 0, 0, 1, 0, 0, 0),  # forward
        _action(0, 0, 0, -1, 0, 0, 0),  # backward
]


# Valid observations
VALID_OBS = ['RGBD_INTERLEAVED',
             'RGB.LOOK_TOP_DOWN',
             'RGB.LOOK_EAST',
             'RGB.LOOK_NORTH_EAST',
             'RGB.LOOK_NORTH',
             'RGB.LOOK_NORTH_WEST',
             'RGB.LOOK_WEST',
             'RGB.LOOK_SOUTH_WEST',
             'RGB.LOOK_SOUTH',
             'RGB.LOOK_SOUTH_EAST',
             'RGB.LOOK_RANDOM',
             'DEBUG.POS.TRANS',
             'DEBUG.POS.ROT']


# customize Deepmind Lab environment
class RandomMazeV1(object):
    def __init__(self, level, observations, configs):
        """

        :param level:
        :param observations:
        :param configs:
        """
        """ set up the 3D maze using default settings"""
        # set the level name
        self._level_name = level
        # set the level configuration
        self._level_configs = configs
        # check the validation of the observations
        assert set(observations) <= set(VALID_OBS), f"Observations contain invalid observations. Please check the " \
                                                    f"valid list here {VALID_OBS}."
        self.observation_names = observations
        # create the lab maze environment with default settings
        self._lab = deepmind_lab.Lab(self._level_name,
                                     self.observation_names,
                                     self._level_configs)

        """ observations from the maze """
        # current observations: contains all the required observations, since we only use a subset of it
        # (e.g. 8 observations). I call it "state"
        self._current_state = None
        # last observation
        self._last_observation = None
        # goal observation
        self._goal_observation = None
        # top down view for debugging or policy visualization
        self._top_down_obs = None
        # last distance
        self._last_distance = None
        # position info
        self._trans = None
        # orientation info
        self._rots = None

        """ configurable parameters for the maze"""
        # map name
        self.maze_name = "default_maze"
        # map size and seed (default: 5x5 maze with random seed set to be 1234)
        self.maze_size = 5
        self.maze_seed = 1234
        # maze map txt (default: 5x5 empty room)
        # note: "P" and "G" are necessary to generate the start agent and the goal.
        # Omitting the "G" will lead to no visual goal generated in the maze.
        self.maze_map_txt = "*****\n*P  *\n*   *\n*  G*\n*****"
        # maze texture name (default)
        self.maze_texture = "MISHMASH"
        # maze wall posters proportion (default)
        self.maze_decal_freq = "0.1"

        """ configurable parameters for the navigation"""
        # start and goal position w.r.t. rough map
        self.start_pos = [1, 1, 0]
        self.goal_pos = [3, 3, 0]
        # global orientations: [0, 45, 90, 135, 180, 225, 270, 315]
        self.orientations = np.arange(0, 360, 45)
        # reward configurations
        self.reward_type = "sparse-0"
        # terminal conditions
        self.dist_epsilon = 35

    # reset function
    def reset(self, configs):
        # check the input type
        assert type(configs) == defaultdict, f"Invalid configuration type. It should be collections.defaultdict, " \
                                             f"but get {type(configs)}"
        # update the maze using the input configurations
        if configs['update']:
            """ 3D maze configurations """
            # initialize the name
            self.maze_name = configs['maze_name'] if configs['maze_name'] else self.maze_name
            # initialize the size
            self.maze_size = configs['maze_size'] if configs['maze_size'] else self.maze_size
            # initialize the seed
            self.maze_seed = configs['maze_seed'] if configs['maze_seed'] else self.maze_seed
            # initialize the map
            self.maze_map_txt = configs['maze_map_txt'] if configs['maze_map_txt'] else self.maze_map_txt
            # initialize the texture
            self.maze_texture = configs['maze_texture'] if configs['maze_texture'] else self.maze_texture
            # initialize the wall posters
            self.maze_decal_freq = configs['maze_decal_freq'] if configs['maze_decal_freq'] else self.maze_decal_freq

            """ navigation configurations"""
            # initialize the start and goal positions
            self.start_pos = configs['start_pos'] if configs['start_pos'] else self.start_pos
            self.goal_pos = configs['goal_pos'] if configs['goal_pos'] else self.goal_pos

            """ send the parameters to lua """
            self._lab.write_property("params._map_name", self.maze_name)
            self._lab.write_property("params.maze_set.size", str(self.maze_size))
            self._lab.write_property("params._map_seed", str(self.maze_seed))
            self._lab.write_property("params._map_texture", self.maze_texture)
            self._lab.write_property("params._map_decal_freq", str(self.maze_decal_freq))
            self._lab.write_property("params._map_txt", self.maze_map_txt)
            # send initial position
            self._lab.write_property("params.agent_pos.x", str(self.start_pos[0] + 1))
            self._lab.write_property("params.agent_pos.y", str(self.start_pos[1] + 1))
            self._lab.write_property("params.agent_pos.theta", str(self.start_pos[2]))
            # send target position
            self._lab.write_property("params.goal_pos.x", str(self.goal_pos[0] + 1))
            self._lab.write_property("params.goal_pos.y", str(self.goal_pos[1] + 1))
            # send the view position
            self._lab.write_property("params.view_pos.x", str(self.goal_pos[0] + 1))
            self._lab.write_property("params.view_pos.y", str(self.goal_pos[1] + 1))
            self._lab.write_property("params.view_pos.theta", str(self.goal_pos[2]))
            self._lab.reset(episode=0)
        else:
            self._lab.reset()

        """ initialize the 3D maze"""
        # initialize the current state
        self._current_state = self._lab.observations()
        # initialize the current observations
        obs_num = len(self.observation_names) - 4
        self._last_observation = [self._current_state[key] for key in self._current_state.keys()][0:obs_num]
        # initialize the goal observations
        self._goal_observation = self.get_random_observations(self.goal_pos)
        # initialize the top down view
        self._top_down_obs = self._current_state['RGB.LOOK_TOP_DOWN']
        # initialize the positions and orientations
        self._trans = self._current_state['DEBUG.POS.TRANS']
        self._rots = self._current_state['DEBUG.POS.ROT']
        # initialize the distance
        self._last_distance = np.inf

        return self._last_observation, self._goal_observation, self._trans, self._rots

    # step function
    def step(self, action):
        """ step #(num_steps) in Deepmind Lab"""
        # take one step in the environment
        self._lab.step(ACTION_LIST[action], num_steps=4)

        """ check the terminal and return observations"""
        if self._lab.is_running():  # If the maze is still running
            # get the next state
            self._current_state = self._lab.observations()
            # get the next observations
            obs_num = len(self._current_state) - 4
            self._last_observation = [self._current_state[key] for key in self._current_state.keys()][0:obs_num]
            # get the next top down observations
            self._top_down_obs = self._current_state['RGB.LOOK_TOP_DOWN']
            # get the next position
            pos_x, pos_y, pos_z = self._current_state['DEBUG.POS.TRANS'].tolist()
            # get the next orientations
            pos_pitch, pos_yaw, pos_roll = self._current_state['DEBUG.POS.ROT'].tolist()
            # check if the agent reaches the goal given the current position and orientation
            terminal, dist = self.reach_goal([pos_x, pos_y, pos_yaw])
            # update the current distance between the agent and the goal
            self._last_distance = dist
            # update the rewards
            reward = self.compute_reward(dist)
            # update the positions and rotations
            self._trans = [pos_x, pos_y, pos_z]
            self._rots = [pos_pitch, pos_yaw, pos_roll]
        else:
            # set the terminal reward
            reward = 0.0
            # set the terminal flag
            terminal = True
            # set the current observation
            self._last_observation = np.copy(self._last_observation)
            # set the top down observation
            self._top_down_obs = np.copy(self._top_down_obs)
            # set the position and orientations
            self._trans = np.copy(self._trans)
            self._rots = np.copy(self._rots)
            # set the distance
            self._last_distance = self._last_distance

        return self._last_observation, reward, terminal, self._last_distance, self._trans, self._rots, dict()

    # render function
    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            return self._lab.observations()
        else:
            super(RandomMazeV1, self).render(mode=mode)  # just raise an exception

    # close the deepmind
    def close(self):
        self._lab.close()

    # seed function
    def seed(self, seed=None):
        self._lab.reset(seed=seed)

    # obtain goal image
    def get_random_observations(self, pos):
        """
        Function is used to get the observations at any position with any orientation.
        :param pos: List contains [x, y, ori]
        :return: List of egocentric observations at position (x, y)
        """
        # store observations
        ego_observations = []
        # re-arrange the observations of the agent
        ori_idx = list(self.orientations).index(pos[2])
        re_arranged_ori_indices = list(range(ori_idx, 8, 1)) + list(range(0, ori_idx, 1))
        angles = self.orientations[re_arranged_ori_indices]
        # obtain the egocentric observations
        self._lab.write_property("params.view_pos.x", str(pos[0] + 1))
        self._lab.write_property("params.view_pos.y", str(pos[1] + 1))
        for a in angles:
            self._lab.write_property("params.view_pos.theta", str(a))
            ego_observations.append(self._lab.observations()['RGB.LOOK_RANDOM'])
        return ego_observations

    def reach_goal(self, current_pos):
        # convert the position from map to 3D maze
        goal_pos_3d = self.position_map2maze(self.goal_pos, self.maze_size)
        # compute the distance and angle error
        dist = self.compute_distance(current_pos, goal_pos_3d)
        if dist < self.dist_epsilon:
            return 1, dist
        else:
            return 0, dist

    def compute_reward(self, dist):
        if self.reward_type == 'sparse-0':
            reward = 1 if dist < self.dist_epsilon else 0
        elif self.reward_type == 'sparse-1':
            reward = 0 if dist < self.dist_epsilon else -1
        elif self.reward_type == 'dense-euclidean':
            reward = dist
        else:
            raise Exception("Invalid reward type, Desired one in sparse-0, sparse-1, dense-euclidean but get {}"
                            .format(self.reward_type))
        return reward

    @staticmethod
    def compute_distance(pos_1, pos_2):
        return np.sqrt((pos_1[0] - pos_2[0])**2 + (pos_1[1] - pos_2[1])**2)

    @staticmethod
    def position_map2maze(pos, size):
        # convert the positions on the map to the positions in the 3D maze.
        # Note: the relation between the positions on the 2D map and the 3D maze is as follows:
        #      2D map: x, y
        #      3D maze: (y + 1 - 1) * 100 + 50, (maze_size - x) * 100 + 50
        return [(pos[1] + 1 - 1) * 100 + 50, (size - pos[0] - 1) * 100 + 50]


#########################
# Test code
#########################
def run_test():
    # level name
    level = "random_customized_maze"
    # desired observations
    observation_list = ['RGBD_INTERLEAVED',
                        'RGB.LOOK_RANDOM',
                        'DEBUG.POS.TRANS',
                        'DEBUG.POS.ROT',
                        'RGB.LOOK_TOP_DOWN']
    # configurations
    configurations = {
        'width': str(160),
        'height': str(160),
        "fps": str(60)
    }
    # create the map environment
    myEnv = RandomMazeV1(level, observation_list, configurations)
    # initialize the maze environment
    maze_configs = defaultdict(lambda: None)
    maze_configs["update"] = True
    state_obs, goal_obs, state_trans, state_rots = myEnv.reset(maze_configs)

    # create observation windows
    fig, arrs = plt.subplots(1, 2)
    front_view = arrs[0].imshow(state_obs[0])
    top_down_view = arrs[1].imshow(myEnv._top_down_obs)

    # start test
    time_steps_num = 10000
    random.seed(maze_configs["maze_seed"])
    ep = 0

    # maze
    theme_list = ["TRON", "MINESWEEPER", "TETRIS", "GO", "PACMAN", "INVISIBLE_WALLS"]
    decal_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for t in range(time_steps_num):
        act = random.sample(range(4), 1)[0]
        next_obs, reward, done, dist, trans, rots, _ = myEnv.step(act)

        front_view.set_data(next_obs[0])
        top_down_view.set_data(myEnv._top_down_obs)
        fig.canvas.draw()
        plt.pause(0.0001)

        if done or t % 20 == 0:
            print("Ep = ", ep)
            ep += 1
            if ep % 10 == 0:
                maze_configs["start_pos"] = [1, 2, 0]
                maze_configs["goal_pos"] = [2, 2, 0]
                maze_configs["update"] = True
                maze_configs["maze_decal_freq"] = random.sample(decal_list, 1)[0]
                maze_configs["maze_texture"] = random.sample(theme_list, 1)[0]
                myEnv.reset(maze_configs)
            else:
                maze_configs["update"] = False
                myEnv.reset(maze_configs)


if __name__ == '__main__':
    run_test()


