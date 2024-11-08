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

    Notes:
        - State observation: egocentric counter-clocked 360 degree observations (8 frames)
        - Goal observation: counter-clocked 360 degree observations (8 frames) but the front orientation is fixed to be
                            [0, 45, 90, 135, 180, 225, 270, 315]
"""
import gym
import numpy as np
import deepmind_lab
import matplotlib.pyplot as plt
from scipy import ndimage
import IPython.terminal.debugger as Debug
plt.rcParams.update({'font.size': 8})


def _action(*entries):
    return np.array(entries, dtype=np.intc)


# actions in Deepmind
ACTION_LIST = [
        _action(-20, 0, 0, 0, 0, 0, 0),  # look_left
        _action(20, 0, 0, 0, 0, 0, 0),  # look_right
        _action(0, 0, 0, 1, 0, 0, 0),  # forward
        _action(0, 0, 0, -1, 0, 0, 0),  # backward
        # _action(0, 0, 0, 0, 0, 0, 0)  # NOOP
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
    def __init__(self, observations, width, height, fps, set_goal=True, set_texture=False, reward_type='sparse-0'):
        """
        Initialization
        :param observations: list of valid observations
        :param width: width of the observation
        :param height: height of the observation
        :param fps: frame number per second
        :param set_goal: flag, if true, a orange goal figure will be shown in the environment
        :param set_texture: flag, if true, generate mazes with random textures
        :param reward_type: flag, if none, use default sparse-0, or (sparse-1, dense-euclidean)
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
        # check the validation of the observations
        assert set(observations) <= set(VALID_OBS), f"Observations contain invalid observations. Please check the " \
                                                    f"valid list here {VALID_OBS}."
        self.observation_names = observations
        # create the lab maze environment
        self._lab = deepmind_lab.Lab(self._level_name,
                                     [obs for obs in observations],
                                     self._level_configs)
        # set the action space
        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        # set the observation space
        self.observation_space = gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)
        # record the start position and end position
        self.start_pos = []
        self.goal_pos = []
        # agent 8 observations
        self._last_observation = None
        self._last_distance = None
        # current state
        self._current_state = None
        # goal observation
        self._goal_observation = []
        # global orientations
        self.orientations = np.arange(0, 360, 45)
        # map info
        self.maze_size = 0
        # top down view for debugging or policy visualization
        self.top_down_obs = None
        # reward configurations
        self.reward_type = reward_type
        self.dist_epsilon = 35

        # position info
        self._trans = None
        self._rots = None

    # reset function
    def reset(self, maze_size=5, maze_seed=0, params=None):
        """
        Reset function of current lab environment
        :param maze_size: maze size
        :param maze_seed: maze seed
        :param params: init_x, init_y, goal_x, goal_y, init_ori, goal_ori
                       Note: init_ori in [0 - 360] is continuous
                             goal_ori in [0, 45, 90, 135, 180, 270, 315] is discrete
        :return: init_state_obs, goal_state_obs
                 Note: for init_state_obs = [init_ori + 0, init_ori + 45, ..., init_ori + 315]
                       for goal_state_obs = [goal_ori, ] rearrange [0, 45, .., 315] from goal_ori
        """
        """ customized reset """

        """ maze customized configurations """
        # check the validation of the size and seed
        assert (maze_size > 0 and maze_seed >= 0)
        # store the size
        self.maze_size = maze_size
        # store the start and goal positions in 2D map
        self.start_pos = [params[0], params[1], params[4]]
        self.goal_pos = [params[2], params[3], params[5]]  # default goal orientation is 0.

        """ send customized configurations to Deepmind """
        # send maze
        self._lab.write_property("params.maze_set.size", str(maze_size))
        self._lab.write_property("params.maze_set.seed", str(maze_seed))
        # send initial position
        self._lab.write_property("params.agent_pos.x", str(self.start_pos[0] + 1))
        self._lab.write_property("params.agent_pos.y", str(self.start_pos[1] + 1))
        self._lab.write_property("params.agent_pos.theta", str(self.start_pos[2]))
        # send target position (uncomment to use the customized terminal)
        # self._lab.write_property("params.goal_pos.x", str(self.goal_pos[0] + 1))
        # self._lab.write_property("params.goal_pos.y", str(self.goal_pos[1] + 1))
        # send the view position
        self._lab.write_property("params.view_pos.x", str(self.goal_pos[0] + 1))
        self._lab.write_property("params.view_pos.y", str(self.goal_pos[1] + 1))
        self._lab.write_property("params.view_pos.theta", str(self.goal_pos[2]))

        """ initialize the Deepmind maze """
        # reset the agent
        self._lab.reset()

        """ obtain the desired observations and return """
        # record the all current observations
        self._current_state = self._lab.observations()
        # obtain the goal observations based on the position and orientation
        self._goal_observation = self.get_random_observations(self.goal_pos)
        # extract the agent current 8 observations
        self._last_observation = [self._current_state[key] for key in self._current_state.keys()][0:8]
        # extract the agent current position
        self._trans = self._current_state['DEBUG.POS.TRANS']
        self._rots = self._current_state['DEBUG.POS.ROT']
        # initialize the distance to be infinite
        self._last_distance = np.inf
        # extract the current top down observations
        self.top_down_obs = self._current_state['RGB.LOOK_TOP_DOWN']
        return self._last_observation, self._goal_observation, self._trans, self._rots

    # step function
    def step(self, action):
        """ step function """

        """ step #(num_steps) in Deepmind Lab"""
        # take one step in the environment
        self._lab.step(ACTION_LIST[action], num_steps=8)

        """ check the terminal and return observations"""
        if self._lab.is_running():  # If the maze is still running
            # get the observations after taking the action
            self._current_state = self._lab.observations()
            # get the current position
            trans = self._current_state['DEBUG.POS.TRANS']
            pos_x, pos_y, pos_z = trans.tolist()
            # get the current orientations
            rots = self._current_state['DEBUG.POS.ROT']
            pos_pitch, pos_yaw, pos_roll = rots.tolist()
            # check if the agent reaches the goal given the current position and orientation
            terminal, dist = self.reach_goal([pos_x, pos_y, pos_yaw])
            # update the current distance between the agent and the goal
            self._last_distance = dist
            # update the rewards
            reward = self.compute_reward(dist)
            # update the observations
            next_obs = None if terminal else [self._current_state[key] for key in self._current_state.keys()][0:8]
            self._last_observation = next_obs if not terminal else np.copy(self._last_observation)
            self.top_down_obs = self._current_state['RGB.LOOK_TOP_DOWN'] if not terminal else np.copy(self.top_down_obs)
            # update the positions and rotations
            self._trans = np.copy(self._trans) if terminal else trans
            self._rots = np.copy(self._rots) if terminal else rots
        else:
            # set the terminal reward
            reward = 0.0
            # set the terminal flag
            terminal = True
            self._last_observation = np.copy(self._last_observation)
            self._trans = np.copy(self._trans)
            self._rots = np.copy(self._rots)
            self._last_distance = self._last_distance
            self.top_down_obs = np.copy(self.top_down_obs)

        # Note, we also return the distance
        return self._last_observation, reward, terminal, self._last_distance, self._trans, self._rots, dict()

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


# """  Test code
# """
#
#
# def env_test(view_name):
#     # necessary observations (correct: this is the egocentric observations (following the counter clock direction))
#     observation_list = ['RGB.LOOK_EAST',
#                         'RGB.LOOK_NORTH_EAST',
#                         'RGB.LOOK_NORTH',
#                         'RGB.LOOK_NORTH_WEST',
#                         'RGB.LOOK_WEST',
#                         'RGB.LOOK_SOUTH_WEST',
#                         'RGB.LOOK_SOUTH',
#                         'RGB.LOOK_SOUTH_EAST',
#                         'RGB.LOOK_RANDOM',
#                         'DEBUG.POS.TRANS',
#                         'DEBUG.POS.ROT',
#                         'RGB.LOOK_TOP_DOWN']
#     observation_width = 32
#     observation_height = 32
#     observation_fps = 60
#
#     # create the environment
#     my_lab = RandomMaze(observation_list, observation_width, observation_height, observation_fps, reward_type='dense-euclidean')
#
#     # episodes
#     episode_num = 100
#
#     # maze size and seeds
#     size = 5
#     seed = 0
#     init_pos = [1, 3, 0]
#     goal_pos = [1, 2, 0]
#
#     # load map
#     # env_map = mapper.RoughMap(size, seed, 3)
#     # reset the environment using the size and seed
#     pos_params = [init_pos[0],
#                   init_pos[1],
#                   goal_pos[0],
#                   goal_pos[1],
#                   init_pos[2],
#                   goal_pos[2]]  # [init_pos, goal_pos, init_orientation]
#     state, goal = my_lab.reset(size, seed, pos_params)
#
#     # plot the goal and the current observations
#     fig, arrs = plt.subplots(3, 3)
#     fig.canvas.set_window_title("Egocentric 360 Degree view")
#     if view_name == 'goal':
#         view_state = goal
#     else:
#         view_state = state
#     # agent-based egocentric 360 degree observations
#     arrs[0, 1].set_title("front")
#     img1 = arrs[0, 1].imshow(view_state[0])
#     arrs[0, 0].set_title("front-left")
#     img2 = arrs[0, 0].imshow(view_state[1])
#     arrs[1, 0].set_title("left")
#     img3 = arrs[1, 0].imshow(view_state[2])
#     arrs[2, 0].set_title("back-left")
#     img4 = arrs[2, 0].imshow(view_state[3])
#     arrs[2, 1].set_title("back")
#     img5 = arrs[2, 1].imshow(view_state[4])
#     arrs[1, 1].set_title("top-down")
#     top_down_img = arrs[1, 1].imshow(ndimage.rotate(my_lab.top_down_obs, 0))
#     arrs[2, 2].set_title("back-right")
#     img6 = arrs[2, 2].imshow(view_state[5])
#     arrs[1, 2].set_title("right")
#     img7 = arrs[1, 2].imshow(view_state[6])
#     arrs[0, 2].set_title("front-right")
#     img8 = arrs[0, 2].imshow(view_state[7])
#
#     # show the policy
#     max_steps = 40
#     for ep in range(episode_num):  # run #episode_num episodes
#         print(f"Episode = {ep+1}, Maze size = {size}, Init = {pos_params[0:2]}, Goal = {pos_params[2:4]}")
#         for t in range(max_steps):  # for each episode, run #max_steps time steps
#             # randomly select one action
#             act = my_lab.action_space.sample()
#             act = 4
#             # step in the environment
#             next_state, reward, done, dist, _ = my_lab.step(act)
#             print("step = {}, dist = {}, reward = {}".format(t+1, dist, reward))
#             # show the current observation and goal
#             img1.set_data(view_state[0])
#             img2.set_data(view_state[1])
#             img3.set_data(view_state[2])
#             img4.set_data(view_state[3])
#             img5.set_data(view_state[4])
#             top_down_img.set_data(ndimage.rotate(my_lab.top_down_obs, 0))
#             img6.set_data(view_state[5])
#             img7.set_data(view_state[6])
#             img8.set_data(view_state[7])
#             fig.canvas.draw()
#             plt.pause(0.0001)
#             # check terminal
#             if dist < 35:
#                 break
#             else:
#                 if view_name == "goal":
#                     view_state = goal
#                 else:
#                     view_state = next_state
#
#         state, goal = my_lab.reset(size, seed, pos_params)
#         if view_name == "goal":
#             view_state = goal
#         else:
#             view_state = state
#     plt.cla()
#
#
# env_test("state")
