import deepmind_lab
import numpy as np
from collections import defaultdict
from scipy import ndimage
import matplotlib.pyplot as plt
import IPython.terminal.debugger as Debug

plt.rcParams.update({'font.size': 8})


# actions in Deepmind
def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTION_LIST = [
    _action(0, 0, -1, 0, 0, 0, 0),  # left
    _action(0, 0, 1, 0, 0, 0, 0),  # right
    _action(0, 0, 0, 1, 0, 0, 0),  # forward
    _action(0, 0, 0, -1, 0, 0, 0),  # backward
    _action(0, 0, 0, 0, 0, 0, 0)
]

# valid observations
VALID_OBS = ['RGBD_INTERLEAVED',
             'RGB.LOOK_RANDOM_PANORAMA_VIEW',
             'RGB.LOOK_PANORAMA_VIEW',
             'RGB.LOOK_TOP_DOWN_VIEW',
             'DEBUG.POS.TRANS',
             'DEBUG.POS.ROT',
             'VEL.TRANS',
             'VEL.ROT'
            ]


# Deepmind domain for random mazes with random start and goal positions
class RandomMaze(object):
    def __init__(self, level, observations, configs, args, use_true_state=False, reward_type='sparse-0', dist_epsilon=1e-3):
        """
        Create gym-like domain interface: This is the original tile version
        :param level: name of the level (currently we only support one name "nav_random_nav")
        :param observations: list of observations [front view, panoramic view, top down view]
        :param configs: configuration of the lab [width, height, fps]
        """
        """ 
            set up the 3D maze using default settings
        """
        # flag variable for using the true state. e.g. a tuple of (x, y, z, pitch, yaw, roll) = (x, y, 0, 0, yaw, 0)
        self._use_state = use_true_state
        # set the level name
        self._level_name = level
        # set the level configuration
        self._level_configs = configs
        # check the validation of the observations
        assert set(observations) <= set(VALID_OBS), f"Observations contain invalid observations. Please check the " \
                                                    f"valid list here {VALID_OBS}."
        self._observation_names = observations + ['RGB.LOOK_RANDOM_VIEW']
        # create the lab maze environment with default settings
        # note set renderer = 'hardware' can use the GPU
        self._lab = deepmind_lab.Lab(self._level_name,
                                     self._observation_names,
                                     self._level_configs,
                                     renderer='software'
                                     )

        """ 
            states and observations from the 3-D maze
        """
        # current state = (x, y, yaw)
        self._current_state = None
        # goal state = (x_g, y_g, yaw_g)
        self._goal_state = None
        # current panoramic observation
        self._current_observation = None
        # goal observation
        self._goal_observation = None
        # top down view for debugging or policy visualization
        self._top_down_observation = None
        # current distance to the goal
        self._current_distance = None
        # position info
        self._trans = None
        # orientation info
        self._rots = None
        # velocity
        self._trans_vel = None
        self._rots_vel = None
        """ 
            Default configuration parameters for the maze on the txt map.
        """
        # map name
        self.maze_name = "default_maze"
        # map size and seed (default: 5x5 maze with random seed set to be 1234)
        self.maze_size = [5, 5]
        self.maze_seed = 1234
        # maze map txt (default: 5x5 empty room)
        # note: "P" and "G" are necessary to generate the start agent and the goal.
        # Omitting the "G" will lead to no visual goal generated in the maze.
        self.maze_map_txt = "*****\n*P  *\n*   *\n*  G*\n*****"
        # maze texture name (default)
        self.maze_texture = "MISHMASH"
        # maze wall posters proportion (default)
        self.maze_decal_freq = "0.1"
        # maze valid positions on the txt map
        self.maze_valid_positions = None

        """ 
            Default configuration parameters for the navigation
        """
        # start and goal position w.r.t. rough map
        self.start_pos_map = [1, 1, 0]
        self.goal_pos_map = [3, 3, 0]
        self.start_pos_maze = self.position_map2maze(self.start_pos_map, self.maze_size)
        self.goal_pos_maze = self.position_map2maze(self.goal_pos_map, self.maze_size)
        # global orientations: [0, 45, 90, 135, 180, 225, 270, 315]
        self.orientations = np.arange(0, 360, 45)
        # reward configurations
        self.reward_type = reward_type
        # terminal conditions
        self.dist_epsilon = dist_epsilon

        # plotting objects
        self.fig, self.arrays = None, None
        self.img_artists = []

        # parameters for sampling start positions
        self.args = args
        self.start_radius = args.start_radius

    # reset function
    def reset(self, configs):
        """ Check configurations validation """
        assert type(configs) == defaultdict, f"Invalid configuration type. It should be collections.defaultdict, " \
                                             f"but get {type(configs)}"

        """ 3D maze configurations """
        """
            if configs['update'] is true, update the whole maze. 
                -- name
                -- size
                -- random seed
                -- texture
                -- decal posters on the wall
                -- maze layout in txt file

            if configs['update'] is false, the maze won't be update
            only the start and goal positions might be updated.
        """
        # if True, create a 3-D maze with appearance controlled by the input configurations
        if configs['update']:
            # initialize the name
            self.maze_name = configs['maze_name'] if configs['maze_name'] else self.maze_name
            # initialize the size
            self.maze_size = configs['maze_size'] if configs['maze_size'] else self.maze_size
            # initialize the seed
            self.maze_seed = configs['maze_seed'] if configs['maze_seed'] else self.maze_seed
            # initialize the texture
            self.maze_texture = configs['maze_texture'] if configs['maze_texture'] else self.maze_texture
            # initialize the wall posters
            self.maze_decal_freq = configs['maze_decal_freq'] if configs['maze_decal_freq'] else self.maze_decal_freq
            # initialize the map
            self.maze_map_txt = configs['maze_map_txt'] if configs['maze_map_txt'] else self.maze_map_txt
            # initialize the valid positions
            self.maze_valid_positions = configs['maze_valid_pos'] if configs['maze_valid_pos'] \
                else self.maze_valid_positions

            """ send the parameters to lua """
            self._lab.write_property("params.maze_configs.name", self.maze_name)
            self._lab.write_property("params.maze_configs.size", str(self.maze_size[0]))
            self._lab.write_property("params.maze_configs.seed", str(self.maze_seed))
            self._lab.write_property("params.maze_configs.texture", self.maze_texture)
            self._lab.write_property("params.maze_configs.decal_freq", str(self.maze_decal_freq))
            self._lab.write_property("params.maze_configs.map_txt", self.maze_map_txt)

        """ Navigation configurations"""
        # send start position in 3-D maze
        if configs['start_pos']:
            self.start_pos_map = configs['start_pos'] if configs['start_pos'] else self.start_pos_map
            # compute the true position in 3-D maze
            self.start_pos_maze = self.position_map2maze(self.start_pos_map, self.maze_size)
            # sampling around the start position
            # Debug.set_trace()
            self.start_pos_maze = self.sampling_around(self.start_pos_maze, self.start_radius)
            # send the position
            self._lab.write_property("params.start_pos.x", str(self.start_pos_maze[0]))
            self._lab.write_property("params.start_pos.y", str(self.start_pos_maze[1]))
            self._lab.write_property("params.start_pos.yaw", str(self.start_pos_maze[2]))
        # send goal position in 3-D maze
        if configs['goal_pos']:
            self.goal_pos_map = configs['goal_pos'] if configs['goal_pos'] else self.goal_pos_map
            self.goal_pos_maze = self.position_map2maze(self.goal_pos_map, self.maze_size)
            #self._lab.write_property("params.goal_pos.x", str(self.goal_pos_maze[0]))
            #self._lab.write_property("params.goal_pos.y", str(self.goal_pos_maze[1]))
            #self._lab.write_property("params.goal_pos.yaw", str(self.goal_pos_maze[2]))

        """ update the environment """
        if configs['update']:
            self._lab.reset(episode=0)
        else:
            self._lab.reset()

        # avoid the laser cylinder when the agent is initialized
        for i in range(10):
            self._lab.step(ACTION_LIST[4], num_steps=4)

        """ initialize the 3D maze"""
        init_state = self._lab.observations()
        # initialize the current state
        self._current_state = init_state['DEBUG.POS.TRANS'].tolist()[0:2] + init_state['VEL.TRANS'].tolist()[0:2] 
        # initialize the current observations
        self._current_observation = init_state['RGB.LOOK_PANORAMA_VIEW']
        # initialize the top down view
        self._top_down_observation = init_state['RGB.LOOK_TOP_DOWN_VIEW']
        # initialize the goal observations
        self._goal_state = self.goal_pos_maze[0:2]
        self._goal_observation = self.get_random_observations_tile(self.goal_pos_map)
        # initialize the positions and orientations
        self._trans = init_state['DEBUG.POS.TRANS'].tolist()
        self._rots = init_state['DEBUG.POS.ROT'].tolist()
        self._trans_vel = init_state['VEL.TRANS'].tolist()
        self._rots_vel = init_state['VEL.ROT'].tolist()
        # initialize the distance to be infinite
        self._current_distance = np.inf

        # return observation or state based on configuration
        if self._use_state:
            return self._current_state, self._goal_state, self._trans, self._rots, self._trans_vel, self._rots_vel
        else:
            return self._current_observation, self._goal_observation, self._trans, self._rots, self._trans_vel, self._rots_vel

    # step function
    def step(self, act):
        # take the action
        self._lab.step(ACTION_LIST[act], num_steps=4)
        current_state = self._lab.observations()

        """ check the terminal and return observations"""
        if self._lab.is_running():
            # get the next observations
            self._current_observation = current_state['RGB.LOOK_PANORAMA_VIEW']
            self._top_down_observation = current_state['RGB.LOOK_TOP_DOWN_VIEW']
            # get the next state
            self._trans = current_state['DEBUG.POS.TRANS'].tolist()
            self._rots = current_state['DEBUG.POS.ROT'].tolist()
            self._trans_vel = current_state['VEL.TRANS'].tolist()
            self._rots_vel = current_state['VEL.ROT'].tolist()
            self._current_state = self._trans[0:2] + self._trans_vel[0:2]
            # check if the agent reaches the goal given the current position and orientation
            terminal, dist = self.reach_goal(self._current_state)
            # update the current distance between the agent and the goal
            self._current_distance = dist
            # update the rewards
            reward = self.compute_reward(dist)
        else:
            # set the terminal reward
            reward = 0.0
            # set the terminal flag
            terminal = True
            # set the current observation
            self._current_state = np.copy(self._current_state)
            self._current_observation = np.copy(self._current_observation)
            self._goal_observation = np.copy(self._goal_observation)
            # set the top down observation
            self._top_down_observation = np.copy(self._top_down_observation)
            # set the position and orientations
            self._trans = np.copy(self._trans)
            self._rots = np.copy(self._rots)
            self._trans_vel = np.copy(self._trans_vel)
            self._rots_vel = np.copy(self._rots_vel)
            # set the distance
            self._current_distance = np.copy(self._current_distance)

        if self._use_state:
            return self._current_state, reward, terminal, self._current_distance, self._trans, self._rots, self._trans_vel, self._rots_vel, dict()
        else:
            return self._current_observation, reward, terminal, self._current_distance, self._trans, self._rots, self._trans_vel, self._rots_vel, dict()

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
        # convert to maze positions
        pos = self.position_map2maze(pos, self.maze_size) if not self._use_state else pos
        # store observations
        ego_observations = []
        # re-arrange the observations of the agent
        ori_idx = list(self.orientations).index(pos[2])
        re_arranged_ori_indices = list(range(ori_idx, 8, 1)) + list(range(0, ori_idx, 1))
        angles = self.orientations[re_arranged_ori_indices]
        # obtain the egocentric observations
        self._lab.write_property("params.view_pos.x", str(pos[0]))
        self._lab.write_property("params.view_pos.y", str(pos[1]))
        self._lab.write_property("params.view_pos.z", str(40))
        for a in angles:
            self._lab.write_property("params.view_pos.yaw", str(a))
            ego_observations.append(self._lab.observations()['RGB.LOOK_RANDOM_VIEW'])
        return np.array(ego_observations, dtype=np.uint8)

    def get_random_observations_tile(self, pos):
        # convert to maze positions
        pos = self.position_map2maze(pos, self.maze_size) if not self._use_state else pos
        # send the parameters
        self._lab.write_property("params.view_pos.x", str(pos[0]))
        self._lab.write_property("params.view_pos.y", str(pos[1]))
        # store observations
        ego_observations = self._lab.observations()["RGB.LOOK_RANDOM_PANORAMA_VIEW"]
        return np.array(ego_observations, dtype=np.uint8)

    def reach_goal(self, current_pos):
        dist = self.compute_distance(current_pos, self.goal_pos_maze)
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

    # show the panoramic view
    def show_panorama_view(self, time_step=None, obs_type='agent'):
        assert len(self._current_observation.shape) == 4, f"Invalid observation, expected observation should be " \
                                                       f"RGB.LOOK_PANORAMA_VIEW, but get {self._observation_names[0]}." \
                                                       f" Please check the observation list. The first one should be " \
                                                       f"RGB.LOOK_PANORAMA_VIEW."
        # set the observation: agent view or goal view
        if obs_type == 'agent':
            observations = self._current_observation
        else:
            observations = self._goal_observation

        # init or update data
        if time_step is None:
            self.fig, self.arrays = plt.subplots(3, 3)
            self.arrays[0, 1].set_title("Front view")
            self.arrays[0, 1].axis("off")
            self.img_artists.append(self.arrays[0, 1].imshow(observations[0]))
            self.arrays[0, 0].set_title("Front left view")
            self.arrays[0, 0].axis("off")
            self.img_artists.append(self.arrays[0, 0].imshow(observations[1]))
            self.arrays[1, 0].set_title("Left view")
            self.arrays[1, 0].axis("off")
            self.img_artists.append(self.arrays[1, 0].imshow(observations[2]))
            self.arrays[1, 1].set_title("Top-down view")
            self.arrays[1, 1].axis("off")
            self.img_artists.append(self.arrays[1, 1].imshow(ndimage.rotate(self._top_down_observation, -90)))
            # self.img_artists.append(None)
            self.arrays[2, 0].set_title("Back left view")
            self.arrays[2, 0].axis("off")
            self.img_artists.append(self.arrays[2, 0].imshow(observations[3]))
            self.arrays[2, 1].set_title("Back view")
            self.arrays[2, 1].axis("off")
            self.img_artists.append(self.arrays[2, 1].imshow(observations[4]))
            self.arrays[2, 2].set_title("Back right view")
            self.arrays[2, 2].axis("off")
            self.img_artists.append(self.arrays[2, 2].imshow(observations[5]))
            self.arrays[1, 2].set_title("Right view")
            self.arrays[1, 2].axis("off")
            self.img_artists.append(self.arrays[1, 2].imshow(observations[6]))
            self.arrays[0, 2].set_title("Front right view")
            self.arrays[0, 2].axis("off")
            self.img_artists.append(self.arrays[0, 2].imshow(observations[7]))
        else:
            self.img_artists[0].set_data(observations[0])
            self.img_artists[1].set_data(observations[1])
            self.img_artists[2].set_data(observations[2])
            self.img_artists[3].set_data(ndimage.rotate(self._top_down_observation, -90))
            self.img_artists[4].set_data(observations[3])
            self.img_artists[5].set_data(observations[4])
            self.img_artists[6].set_data(observations[5])
            self.img_artists[7].set_data(observations[6])
            self.img_artists[8].set_data(observations[7])
        self.fig.canvas.draw()
        plt.pause(0.0001)
        return self.fig

        # show the panoramic view

    def show_panorama_view_test(self, flag, state):
        observations = state
        # init or update data
        if flag is None:
            self.fig, self.arrays = plt.subplots(3, 3)
            self.fig.set_figheight(15)
            self.fig.set_figwidth(15)
            self.arrays[0, 1].set_title("North view")
            self.arrays[0, 1].axis("off")
            self.img_artists.append(self.arrays[0, 1].imshow(observations[0]))
            self.arrays[0, 0].set_title("Northwest view")
            self.arrays[0, 0].axis("off")
            self.img_artists.append(self.arrays[0, 0].imshow(observations[1]))
            self.arrays[1, 0].set_title("West view")
            self.arrays[1, 0].axis("off")
            self.img_artists.append(self.arrays[1, 0].imshow(observations[2]))
            self.arrays[1, 1].set_title("Top-down view")
            self.arrays[1, 1].axis("off")
            self.img_artists.append(self.arrays[1, 1].imshow(ndimage.rotate(self._top_down_observation, -90)))
            self.arrays[2, 0].set_title("Southwest view")
            self.arrays[2, 0].axis("off")
            self.img_artists.append(self.arrays[2, 0].imshow(observations[3]))
            self.arrays[2, 1].set_title("South view")
            self.arrays[2, 1].axis("off")
            self.img_artists.append(self.arrays[2, 1].imshow(observations[4]))
            self.arrays[2, 2].set_title("Southeast view")
            self.arrays[2, 2].axis("off")
            self.img_artists.append(self.arrays[2, 2].imshow(observations[5]))
            self.arrays[1, 2].set_title("East view")
            self.arrays[1, 2].axis("off")
            self.img_artists.append(self.arrays[1, 2].imshow(observations[6]))
            self.arrays[0, 2].set_title("Northeast view")
            self.arrays[0, 2].axis("off")
            self.img_artists.append(self.arrays[0, 2].imshow(observations[7]))
        else:
            self.img_artists[0].set_data(observations[0])
            self.img_artists[1].set_data(observations[1])
            self.img_artists[2].set_data(observations[2])
            self.img_artists[3].set_data(ndimage.rotate(self._top_down_observation, -90))
            self.img_artists[4].set_data(observations[3])
            self.img_artists[5].set_data(observations[4])
            self.img_artists[6].set_data(observations[5])
            self.img_artists[7].set_data(observations[6])
            self.img_artists[8].set_data(observations[7])
        self.fig.canvas.draw()
        plt.pause(0.0001)
        return self.fig

    # show the front view
    def show_front_view(self, time_step=None):
        assert len(self._current_observation.shape) == 3, f"Invalid observation, expected observation should be " \
                                                       f"RGBD_INTERLEAVED, but get {self._observation_names[0]}." \
                                                       f" Please check the observation list. The first one should be" \
                                                       f" RGBD_INTERLEAVED."
        # init or update data
        if time_step is None:
            self.fig, self.arrays = plt.subplots(1, 2)
            self.img_artists.append(self.arrays[0].imshow(self._current_observation[0]))
            self.img_artists.append(self.arrays[1].imshow(ndimage.rotate(self._top_down_observation, -90)))
        else:
            self.img_artists[0].set_data(self._current_observation[0])
            self.img_artists[1].set_data(ndimage.rotate(self._top_down_observation, -90))
        self.fig.canvas.draw()
        plt.pause(0.0001)

    @staticmethod
    def compute_distance(pos_1, pos_2):
        return np.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

    @staticmethod
    def position_map2maze(pos, size):
        # convert the positions on the map to the positions in the 3D maze.
        # Note: the relation between the positions on the 2D map and the 3D maze is as follows:
        #      2D map: row, col
        #      3D maze: (y + 1 - 1) * 100 + 50, (maze_size - x) * 100 + 50
        return [pos[1] * 100 + 50, (size[1] - pos[0] - 1) * 100 + 50, pos[2]]

    @staticmethod
    def position_maze2map(pos, size):
        map_row = size[1] - (pos[1] // 100) - 1
        map_col = pos[0] // 100
        return [map_row, map_col, 0]

    @staticmethod
    def sampling_around(pos, radius):
        # store the sampled position
        sampled_pos = [0, 0, 0]
        # sample a length
        length = radius * np.random.uniform(0, 1)
        # sample an angle
        angle = np.pi * np.random.uniform(0, 2)
        # compute the offset
        x_offset = length * np.cos(angle)
        y_offset = length * np.sin(angle)
        # add the sampled pos
        sampled_pos[0] = pos[0] + x_offset
        sampled_pos[1] = pos[1] + y_offset
        sampled_pos[2] = pos[2]

        return sampled_pos
