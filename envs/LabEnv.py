"""
    Definition of the customize Deepmind Lab environment
        - import gym
        - import deepmind_lab

        Class RandomMaze(gym.Env):
            def __init__(self, self-defined params):
                - env
                - action_space (discrete): gym.spaces.Discrete
                - observations_space : gym.spaces.Box
            def reset
            def step
            def close
            def seed
        Note: the extra params in __init__() are keyword arguments
"""
import gym
import numpy as np
import deepmind_lab


def _action(*entries):
    return np.array(entries, dtype=np.intc)


# actions in Deepmind
ACTION_LIST = [
        _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
        _action(20, 0, 0, 0, 0, 0, 0),  # look_right
        _action(0, 0, 0, 1, 0, 0, 0),  # forward
        _action(0, 0, 0, -1, 0, 0, 0),  # backward
        _action(0, 0, 0, 0, 0, 0, 0),  # NOOP
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
class RandomMaze(gym.Env):
    def __init__(self, observations, width, height, fps, set_goal=True, set_texture=False):
        """
        Initialization
        :param observations: list of valid observations
        :param width: width of the observation
        :param height: height of the observation
        :param fps: frame number per second
        :param set_goal: flag, if true, a orange goal figure will be shown in the environment
        :param set_texture: flag, if true, generate mazes with random textures
            - Note:  set_goal  set_texture   level name
                       true       true       random_maze_custom_view_various_texture
                       true       false      random_maze_custom_view_fixed_texture
                       false      true       random_maze_custom_view_no_goal_various_texture
                       false      false      random_maze_custom_view_no_goal_fixed_texture

        """
        # set the level name
        if not set_texture:
            if set_goal:  # random_maze_custom_view_fixed_texture
                self._level_name = 'random_maze_custom_view_fixed_texture'
            else:  # random_maze_custom_view_no_goal_fixed_texture
                self._level_name = 'random_maze_custom_view_no_goal_fixed_texture'
        else:
            if set_goal:  # random_maze_custom_view_various_texture
                self._level_name = 'random_maze_custom_view_various_texture'
            else:  # random_maze_custom_view_no_goal_various_texture
                self._level_name = 'random_maze_custom_view_no_goal_various_texture'
        # set the level configuration
        self._level_configs = {
            "fps": str(fps),
            "width": str(width),
            "height": str(height)
        }
        # create the environment using the level name and configurations
        # check the validation of the observations
        assert set(observations) <= set(VALID_OBS), f"Observations contain invalid observations. Please check the " \
                                                    f"valid list here {VALID_OBS}."
        self.observation_names = observations
        self._lab = deepmind_lab.Lab(self._level_name,
                                     [obs for obs in observations],
                                     self._level_configs)
        # construct the action space
        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        # construct the observation space
        self.observation_space = gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)
        # last observations
        self._current_observation = None
        self._last_observation = None
        # record the start position and end position
        self.start_pos = []
        self.goal_pos = []
        # current state
        self._current_state = None
        # goal observation
        self.goal_observation = []
        # global orientations
        self.orientations = np.arange(0, 360, 45)
        # map info
        self.maze_size = 0
        # debug
        self.top_down_obs = None

    # reset function
    def reset(self, maze_size=5, maze_seed=0, params=None):
        assert (maze_size > 0 and maze_seed >= 0)
        self.maze_size = maze_size
        # record the start and end position and the orientation is set default as EAST
        self.start_pos = [params[0], params[1]]
        tmp_ori = np.random.choice(self.orientations, 1).item()
        self.goal_pos = [params[2], params[3], tmp_ori]
        # send maze
        self._lab.write_property("params.maze_set.size", str(maze_size))
        self._lab.write_property("params.maze_set.seed", str(maze_seed))
        # send initial position
        self._lab.write_property("params.agent_pos.x", str(self.start_pos[0] + 1))
        self._lab.write_property("params.agent_pos.y", str(self.start_pos[1] + 1))
        self._lab.write_property("params.agent_pos.theta", str(params[-1] * 90))
        # send target position
        self._lab.write_property("params.goal_pos.x", str(self.goal_pos[0] + 1))
        self._lab.write_property("params.goal_pos.y", str(self.goal_pos[1] + 1))
        # set the view position
        self._lab.write_property("params.view_pos.x", str(self.goal_pos[0] + 1))
        self._lab.write_property("params.view_pos.y", str(self.goal_pos[1] + 1))
        # reset the agent
        self._lab.reset()
        # obtain the goal observations based on the position and orientation
        self.goal_observation = self.get_random_observations(self.goal_pos)
        # record the current observation
        self._current_state = self._lab.observations()
        self._last_observation = [self._current_state[key] for key in self._current_state.keys()][0:8]
        self.top_down_obs = self._current_state['RGB.LOOK_TOP_DOWN']
        return self._last_observation, self.goal_observation

    # step function
    def step(self, action):
        # take one step in the environment
        reward = self._lab.step(ACTION_LIST[action], num_steps=4)
        # compute terminal flag
        if self._lab.is_running():
            # get the current observations
            self._current_state = self._lab.observations()
            # # get the current position and orientation
            # pos_x, pos_y, pos_z = self._current_state['DEBUG.POS.TRANS'].tolist()
            # pos_pitch, pos_yaw, pos_roll = self._current_state['DEBUG.POS.ROT'].tolist()
            # # get the terminal flag
            # terminal = self.reach_goal([pos_x, pos_y, pos_yaw])
            if reward == 10:
                terminal = True
            else:
                terminal = False
            # get the reward
            reward = 0.0 if terminal else self.compute_reward()
            # get the next
            next_obs = None if terminal else [self._current_state[key] for key in self._current_state.keys()][0:8]
            self._last_observation = next_obs if next_obs is not None else np.copy(self._last_observation)
            self.top_down_obs = self._current_state['RGB.LOOK_TOP_DOWN'] if not terminal else np.copy(self.top_down_obs)
        else:
            # set the terminal observations
            next_obs = None
            # set the terminal reward
            reward = 0.0
            # set the terminal flag
            terminal = True
            self._last_observation = np.copy(self._last_observation)
            self.top_down_obs = np.copy(self.top_down_obs)

        return self._last_observation, reward, terminal, dict()

    # render function
    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            return self._lab.observations()
        else:
            super(RandomMaze, self).render(mode=mode)  # just raise an exception

    # close the deepmind
    def close(self):
        self._lab.close()

    # seed function
    def seed(self, seed=None):
        self._lab.reset(seed=seed)

    # obtain goal image
    def get_random_observations(self, pos):
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
        dist = np.sqrt((current_pos[0] - goal_pos_3d[0])**2 + (current_pos[1] - goal_pos_3d[1])**2)
        angle_error = np.abs(current_pos[2] - self.goal_pos[2])

        if dist < 1 and angle_error < 1:
            return True
        else:
            return False

    @staticmethod
    def compute_reward():
        return -1.0

    @staticmethod
    def position_map2maze(pos, size):
        # convert the positions on the map to the positions in the 3D maze.
        return [(pos[1] - 1) * 100 + 50, (size - pos[0]) * 100 + 50]


# reference
""" Note: deepmind_lab.Lab(level, observations, config={}, renderer='software', level_cache=None)
                      Input args:
                            level: game script file containing the general design of the maze
                            observations: list object that contains the supported observations.
                                        # Supported type:
                                        ['RGB_INTERLEAVED', 'RGBD_INTERLEAVED', 'RGB', 'RGBD', 'BGR_INTERLEAVED'
                                          'BGRD_INTERLEAVED', ...]
                                        # Output obs:
                                        [{'dtype': <type 'numpy.uint8'>, 'name': 'RGB_INTERLEAVED', 'shape': (180, 320, 3)},
                                         {'dtype': <type 'numpy.uint8'>, 'name': 'RGBD_INTERLEAVED', 'shape': (180, 320, 4)},
                                         {'dtype': <type 'numpy.uint8'>, 'name': 'RGB', 'shape': (3, 180, 320)},
                                         {'dtype': <type 'numpy.uint8'>, 'name': 'RGBD', 'shape': (4, 180, 320)},
                                         {'dtype': <type 'numpy.uint8'>, 'name': 'BGR_INTERLEAVED', 'shape': (180, 320, 3)},
                                         {'dtype': <type 'numpy.uint8'>, 'name': 'BGRD_INTERLEAVED', 'shape': (180, 320, 4)},
                                         {'dtype': <type 'numpy.float64'>, 'name': 'MAP_FRAME_NUMBER', 'shape': (1,)},
                                         {'dtype': <type 'numpy.float64'>, 'name': 'VEL.TRANS', 'shape': (3,)},
                                         {'dtype': <type 'numpy.float64'>, 'name': 'VEL.ROT', 'shape': (3,)},
                                         {'dtype': <type 'str'>, 'name': 'INSTR', 'shape': ()},
                                         {'dtype': <type 'numpy.float64'>, 'name': 'DEBUG.POS.TRANS', 'shape': (3,)},
                                         {'dtype': <type 'numpy.float64'>, 'name': 'DEBUG.POS.ROT', 'shape': (3,)},
                                         {'dtype': <type 'numpy.float64'>, 'name': 'DEBUG.PLAYER_ID', 'shape': (1,)},...]
                            config: dictionary object to set the additional settings.
                                        # Supported configurations Note the value is string type
                                        {'width': '320', 'height': '240', 'fps':'60', ...}
                            rendered: Building with --define graphics=<option> sets which graphics implementation is used.
                                    1. --define graphics=osmesa_or_egl.
                                    If no define is set then the build uses this config_setting at the default.
                                    If renderer is set to 'software' then osmesa is used for rendering.
                                    If renderer is set to 'hardware' then EGL is used for rendering.

                                    2. --define graphics=osmesa_or_glx.
                                    If renderer is set to 'software' then osmesa is used for rendering.
                                    If renderer is set to 'hardware' then GLX is used for rendering.

                                    3. --define graphics=sdl.
                                    This will render the game to the native window. One of the observation starting with 
                                    'RGB' must be in the observations for the game to render correctly.

                            level_cache: This is important for changing map for each episode.
            """
