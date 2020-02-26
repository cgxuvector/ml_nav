import gym
import gym_deepmindlab


class LabEnv(object):
    """
        This class is object is created to generate desired maze environment in DeepMind Lab
        A good wrap of an environment should contain:
            - step(current action) -> next state, reward, done flag
            - reset() -> restart an episode
    """
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
    def __init__(self, observations, width, height, fps, goal=True, uniform_texture=True):
        """
        Function that create a customized Deepmind Environment using gym wrapper
        :param observations: name of the observations
        :param width: width of the figure
        :param height: height of the figure
        :param fps: fps
        :param goal: If it is set to be True, the goal is set in the environment.
        :param uniform_texture: If it is set to be True, all the wall, floor, and sky in
        different mazes is uniform.
        """
        # create the environment
        if uniform_texture:
            if goal:
                self._env = gym.make('DeepmindLabRandomMazeCustomViewFixedTexture-v0',
                                     width=width,
                                     height=height,
                                     fps=fps,
                                     observations=observations)
            else:
                self._env = gym.make('DeepmindLabRandomMazeCustomViewNoGoalFixedTexture-v0',
                                     width=width,
                                     height=height,
                                     fps=fps,
                                     observations=observations)
        else:
            if goal:
                self._env = gym.make('DeepmindLabRandomMazeCustomViewVariousTexture-v0',
                                     width=width,
                                     height=height,
                                     fps=fps,
                                     observations=observations)
            else:
                self._env = gym.make('DeepmindLabRandomMazeCustomViewNoGoalVariousTexture-v0',
                                     width=width,
                                     height=height,
                                     fps=fps,
                                     observations=observations)
        # actions: forward, back, left, right, NOOP
        self.action_space = self._env.action_space
        # name of the observations: panoramic and top-down observations
        self.observation_name = observations
        self.observation_space = self._env.observation_space
        # last observations
        self._last_observations = None

    # reset function
    def reset(self, maze_size, maze_seed, params):
        """
        Function is used to set up the 3d maze.
        :param maze_size: size of the maze
        :param maze_seed: seed of the maze
        :param params: agent and goal positions [init_x, init_y, goal_x, goal_y, init_orientation]
        :return: observations
        """
        self._last_observations = self._env.reset(maze_size, maze_seed, params)
        return self._last_observations

    # step function
    def step(self, action):
        """
        Step function
        :param action: index of the five actions. (0:left, 1:right, 2:forward, 3:backward, 4:NOOP)
        :return: observations, reward, done, args
        """
        observation, reward, done, args = self._env.step(action)
        return observation, reward, done, args

    # render function
    def render(self):
        return self._env.render()

    # close the deepmind
    def close(self):
        self._env.close()
