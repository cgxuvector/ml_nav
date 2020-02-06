import deepmind_lab


class LabEnv(object):
    """
        This class is object is created to generate desired maze environment in DeepMind Lab
        A good wrap of an environment should contain:
            - step(current action) -> next state, reward, done flag
            - reset() -> restart an episode
    """
    def __init__(self, maze_type, width, height, fps, level):
        # environment configuration
        self.env_config = {
            'fps': str(fps),
            'width': str(width),
            'height': str(height)
        }
        self.mt = maze_type
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
        self.env = deepmind_lab.Lab(level, ['RGBD_INTERLEAVED'], config=self.env_config)
        self.done = False
        self.reward = 0
        self.observation = None

    # Optional
    def action_spec(self):
        return self.env.action_spec()

    def reset(self, episode):
        """
        Reset function, return the environment reset flag and the observations for the new episode
        """
        """ reset(episode=-1, seed=None): This function resets the environment and start a new episode when the last one
                                                  ends. Additionally, it automatically set is_running()=False

                                                  episode: load the level in a specific episode.
                                                  seed: random seed for each episode.
        """
        self.done = self.env.reset(episode=episode)
        self.observation = self.env.observations()['RGBD_INTERLEAVED']

        return self.observation, self.done

    def step(self, action):
        """
        Step function
            Input: action
            Output: observation of next state, reward, and done flag
        """
        assert(self.env.is_running())  # make sure the environment is running

        """ step(action, num_steps=1): built-in step function to execute the action in the following "num_steps" frames.
        """
        self.reward = self.env.step(action, num_steps=1)  # obtain the reward
        self.observation = self.env.observations()['RGBD_INTERLEAVED']  # return a list of observations
        self.done = not self.env.is_running()  # obtain the terminal flag

        return self.observation, self.reward, self.done
