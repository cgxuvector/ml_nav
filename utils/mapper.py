import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from pathlib import Path


from utils import searchAlg


# define a class for the rough map
class RoughMap(object):
    """
        Map coordinate system:
            I adopt the image coordinates system as the map system. Particularly, X range is (0, columns) while
            y range is (0, rows). The coordinate for one point on the image is (r, c) (i.e. adopting the same
            inference in image coordinate system). In addition, the direction
            of the map is defined as follows:
                - North: the up top
                - South: the bottom down
                - East: the far right hand side
                - West: the fat left hand side.
    """

    # init function
    def __init__(self, m_size, m_seed, loc_map_size):
        """
        Function is used to initialize the current maps
        :param m_size: size of the maze
        :param m_seed: size of the random seed
        """
        # map parameters
        self.maze_size = m_size
        self.maze_seed = m_seed
        self.loc_map_size = loc_map_size
        self.distortion_factor = [1, 1, 1, 1]  # scaling along [horizon, vertical, goal, start]

        # maps
        self.valid_pos = []
        self.raw_pos = {'init': None, 'goal': None}
        self.map2d_txt = self.load_map('txt')
        """ Note: the initial position and goal position is randomly sampled from the scaled range
        """
        self.pos, self.map2d_bw = self.load_map('bw')
        self.map2d_grid = self.load_map('grid')

        # plan path (type: list of ndarray in numpy)
        self.path, self.map2d_path = self.generate_path()
        # egocentric actions
        self.map_act, self.ego_act = self.path2egoaction(self.path)
        # egocentric local maps (size adjustable)
        self.map2d_rough = np.ones(self.map2d_grid.shape) - self.map2d_grid
        self.local_maps = self.crop_local_maps()

    # load txt map
    def load_map(self, m_type):
        """
        Function is used to load the map from ".txt" file
        :param m_type: type of the map
        :return: Maps in type: txt, bw, grid
        """
        # file name
        map_name = '_'.join(['map', str(self.maze_size), str(self.maze_seed)]) + '.txt'
        # file path
        map_path = str(Path(__file__).parent.parent) + '/maps/train/' + map_name
        # load map
        if m_type == 'txt':
            with open(map_path, 'r') as f:
                map_txt = f.readlines()
                for i_idx, l in enumerate(map_txt):
                    for j_idx, s in enumerate(l):
                        if s == ' ':
                            self.valid_pos.append([i_idx, j_idx])
                tmp_pos = np.zeros(2)
                while tmp_pos[0] == tmp_pos[1]:
                    tmp_pos = np.random.randint(0, len(self.valid_pos), 2)
                self.raw_pos['init'] = self.valid_pos[tmp_pos[0]]
                self.raw_pos['goal'] = self.valid_pos[tmp_pos[1]]
                # print('init pos = {}, goal pos = {}'.format(self.raw_pos['init'], self.raw_pos['goal']))
            return map_txt
        elif m_type == 'bw':
            map_bw = self.map_txt2bw(map_path, self.maze_size)
            return map_bw
        elif m_type == 'grid':
            map_grid = self.map_bw2grid(self.map2d_bw)
            return map_grid
        else:
            raise Exception("Map Type Error: Invalid type \"{}\". Please select from \"txt\" , \"grid\" or \"bw\"". \
                            format(m_type))

    # generate binary map
    def map_txt2bw(self, map_name, map_size):
        """
        Function is used to generated binary map.
        """
        # scale factor
        horizon_upscale = self.distortion_factor[0]
        vertical_upscale = self.distortion_factor[1]
        goal_upscale = self.distortion_factor[2]
        start_upscale = self.distortion_factor[3]

        image_h_size = map_size * horizon_upscale
        image_v_size = map_size * vertical_upscale

        # initialize white field
        bw_img_data = [[1.0 for _ in range(image_h_size)] for _ in range(image_v_size)]  # create a while background
        f = open(map_name, 'r')
        # initialize start and end position
        for i, line in enumerate(f):
            for j, char in enumerate(line):
                if char == '*':
                    for v in range(vertical_upscale):
                        for h in range(horizon_upscale):
                            # make walls black
                            bw_img_data[(i * vertical_upscale + v)][j * horizon_upscale + h] = 0

                if i == self.raw_pos['goal'][0] and j == self.raw_pos['goal'][1]:
                    # goal is set to be darker: 0.2
                    # pos['goal'] = [i, j]
                    for k in range(goal_upscale):
                        # mark goal with x
                        bw_img_data[(i * vertical_upscale + k)][j * horizon_upscale + k] = 0.2
                        bw_img_data[(i * vertical_upscale + goal_upscale - 1 - k)][j * horizon_upscale + k] = 0.2

                if i == self.raw_pos['init'][0] and j == self.raw_pos['init'][1]:
                    # init_pose.append()
                    # initial is set to be lighter: 0.8
                    # pos['init'] = [i, j]
                    for k in range(start_upscale):
                        # mark goal with x
                        bw_img_data[(i * vertical_upscale + k)][j * horizon_upscale + k] = 0.8
                        bw_img_data[(i * vertical_upscale + goal_upscale - 1 - k)][j * horizon_upscale + k] = 0.8
        f.close()

        # rescale the init and goal positions and randomly sample them
        rescaled_init = [self.raw_pos['init'][0] * vertical_upscale + np.random.randint(start_upscale), \
                         self.raw_pos['init'][1] * horizon_upscale + np.random.randint(start_upscale)]
        rescaled_goal = [self.raw_pos['goal'][0] * vertical_upscale + np.random.randint(goal_upscale), \
                         self.raw_pos['goal'][1] * horizon_upscale + np.random.randint(goal_upscale)]

        return [rescaled_init, rescaled_goal], np.asarray(bw_img_data)

    # build grid world
    @ staticmethod
    def map_bw2grid(bw_map):
        """
        Function is used to build the grid map from the binary map
        """
        grid_map = resize(bw_map, bw_map.shape)

        row = grid_map.shape[0]
        col = grid_map.shape[1]
        # obtain the grid map
        for r in range(row):
            for c in range(col):
                if grid_map[r, c] < 0.3:
                    grid_map[r, c] = 1
                else:
                    grid_map[r, c] = 0.0

        return grid_map

    # display the map: txt map or occupancy map
    def show_map(self, m_type):
        """
        Function is used to show the different types of map
        """
        # add a function that plots all the maps.
        if m_type == 'txt':
            for line in self.map2d_txt:
                print(line)
        elif m_type == 'bw':
            fig = plt.figure(figsize=(4, 4))
            fig.canvas.set_window_title("Binary Map")
            plt.imshow(self.map2d_bw, cmap=plt.cm.gray)
            plt.show()
            plt.close(fig)
        elif m_type == 'grid':
            fig = plt.figure(figsize=(4, 4))
            fig.canvas.set_window_title("Grid World Map")
            plt.imshow(self.map2d_grid, cmap=plt.cm.gray)
            plt.show()
            plt.close(fig)
        elif m_type == 'all':
            fig, fig_arr = plt.subplots(2, 2, figsize=(8, 8))
            fig.canvas.set_window_title("Maps")
            fig_arr[0, 0].set_title("Binary Map")
            fig_arr[0, 0].imshow(self.map2d_bw, cmap=plt.cm.gray)
            fig_arr[0, 1].set_title("Grid World Map")
            fig_arr[0, 1].imshow(self.map2d_grid, cmap=plt.cm.gray)
            fig_arr[1, 0].set_title("Planned Path")
            fig_arr[1, 0].imshow(self.map2d_path, cmap=plt.cm.gray)
            fig_arr[1, 1].set_title("Rough Map")
            fig_arr[1, 1].imshow(self.map2d_rough, cmap=plt.cm.gray)
            plt.show()
            plt.close(fig)
        else:
            raise Exception("Map Type Error: Invalid type \"{}\". Please select from \"txt\" , \"grid\" or \"bw\"". \
                            format(m_type))

    def generate_path(self):
        path = searchAlg.A_star(self.map2d_grid, self.pos[0], self.pos[1])
        path_map = self.show_path(path, self.map2d_grid)
        return path, path_map

    # display the path on the map
    @ staticmethod
    def show_path(path, grid_map):
        """
        Function is used to print path on the binary map
        """
        # flip the grid_map to be binary map
        grid_map = np.array(grid_map)
        grid_map = np.ones(grid_map.shape) - grid_map
        for pos in path:
            pos = list(pos)
            grid_map[pos[0], pos[1]] = 0.5
        grid_map[path[0][0], path[0][1]] = 0.8
        grid_map[path[-1][0], path[-1][1]] = 0.2

        return grid_map

    @staticmethod
    def rotate_compass(compass, act):
        """
        Function is used to rotate the current compass after executing the last action using the current compass
        :param compass: current compass (relative egocentric)
        :param act: last action in current compass
        :return: rotated compass
        """
        # This is a local compass
        # action in the current compass
        default_compass = ["up", 'right', 'down', 'left']
        # compute the rotate steps
        rotate_step = default_compass.index(act)
        # achieve the rotation using circle
        while rotate_step > 0:
            head = compass.pop()
            compass.insert(0, head)
            rotate_step -= 1
        return compass

    # convert path to egocentric actions
    def path2egoaction(self, path):
        """
        Note: the currently, I assume the rough map is accurate of the layout but inaccurate w.r.t.
        geometry of the environment. Therefore, it is impossible that the planning path will
        lead to a blocked corridor. Therefore, as for the egocentric motions, I only define
        forward, turn left, turn right. In addition, turning actions from map is equal to
        two actions in the egocentric action space. e.g. left = turn left and forward.

        Besides, the first egocentric should always be turn to the map direction.
        """
        # obtain map actions
        map_actions = []
        for i in range(len(path) - 1):
            # current position on the path
            pos = path[i]
            # next position on the path
            next_pos = path[i+1]
            # relative position vector
            pos_dir = list(next_pos - pos)

            if pos_dir == [-1, 0]:
                map_act = 'up'

            if pos_dir == [1, 0]:
                map_act = 'down'

            if pos_dir == [0, -1]:
                map_act = 'left'

            if pos_dir == [0, 1]:
                map_act = 'right'

            map_actions.append(map_act)

        # convert map actions to egocentric actions
        default_compass = ["up", "right", "down", "left"]  # initial default compass
        action_compass = [0, 1, 2, 3]  # initial action compass
        ego_actions = []
        # Idea: The compass need to be updated after taking any action in it.
        # First level: The correct local action should be decided by the default compass on the map and the
        # local compass
        # Second level: The local compass need to be updated using the local action.
        for idx, map_act in enumerate(map_actions):
            # process the initial action
            if idx == 0:
                # store the first rotating action
                ego_actions.append(map_act)
                # rotate the current compass
                action_compass = self.rotate_compass(action_compass, ego_actions[-1])
                continue
            # compute the egocentric action (w.r.t local compass)
            old_idx = default_compass.index(map_act)
            ego_act = default_compass[action_compass[old_idx]]
            # store the egocentric action
            ego_actions.append(ego_act)
            # update the local compass using the egocentric action
            self.rotate_compass(action_compass, ego_act)

        # refine the ego actions:
        # In 3D: left only rotates the agent. there for a left on the map equals to left and up in 3D
        refined_ego_actions = []
        for ego_act in ego_actions:
            if ego_act == "left" or ego_act == "right":
                refined_ego_actions.append(ego_act)
                refined_ego_actions.append("up")
            else:
                refined_ego_actions.append(ego_act)
        return map_actions, refined_ego_actions

    def crop_local_maps(self):
        # store the local maps
        local_maps = []
        # padding the rough map based on crop size
        pad_step = int((self.loc_map_size - 1) / 2)
        # create background
        pad_map_size = int(self.map2d_rough.shape[0] + 2 * pad_step)
        pad_map = np.zeros((pad_map_size, pad_map_size))
        # insert the rough map
        pad_map[pad_step:pad_map_size-pad_step, pad_step:pad_map_size-pad_step] = self.map2d_rough

        for idx, pos in enumerate(self.path):
            pad_pos = pos + [pad_step, pad_step]
            # crop one local map
            loc_map = pad_map[(pad_pos[0] - pad_step):(pad_pos[0] + pad_step + 1), \
                              (pad_pos[1] - pad_step):(pad_pos[1] + pad_step + 1)]
            local_maps.append(loc_map)
            # test draw the path on the padding map
            # pad_map[pad_pos[0], pad_pos[1]] = 0.5

            # # rotate the local map
            #
            # plt.imshow(loc_map, cmap=plt.cm.gray)
            # plt.show()

        # plt.imshow(pad_map, cmap=plt.cm.gray)
        # plt.show()

        return local_maps




