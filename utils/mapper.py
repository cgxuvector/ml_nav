import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from pathlib import Path
import random
import copy
from utils import searchAlg
import IPython.terminal.debugger as Debug
"""
    1. Using a topological graph to store the positions (currently, it is stored as a list)
    2. Fix the bug in rescaling.
"""


# define a class for the rough map
class RoughMap(object):
    """
        Map coordinate system:
            I adopt the image coordinates system as the map system. Particularly, X range is (0, rows) while
            y range is (0, columns). The coordinate for one point on the image is (r, c) (i.e. adopting the same
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
        :param loc_map_size: size of the local map
        """
        # map parameters
        self.maze_size = m_size
        self.maze_seed = m_seed
        self.distortion_factor = [1, 1, 1, 1]  # scaling along [horizon, vertical, goal, start]

        # properties
        self.loc_map_size = loc_map_size

        # txt map and primary valid positions and primary valid local maps
        self.valid_pos = []  # walkable positions in the txt map
        self.valid_loc_maps = []  # local map for the walkable positions
        self.map2d_txt = self.load_map('txt')  # map represented as the raw txt file
        self.init_pos = self.valid_pos[0]
        self.goal_pos = self.valid_pos[-1]
        self.init_pos, self.goal_pos = self.sample_start_goal_pos(True, True)

        # obtain bw map and randomly selected initial and target goals
        self.init_pos, self.goal_pos, self.map2d_bw = self.load_map('bw')

        # obtain the grid world map to do the planning
        self.map2d_grid = self.load_map('grid')
        self.path, self.map2d_path = self.generate_path()
        # egocentric actions
        self.map_act, self.ego_act = self.path2egoaction(self.path)
        # egocentric local maps (size adjustable)
        self.map2d_rough = np.ones(self.map2d_grid.shape) - self.map2d_grid
        self.local_maps, self.map2d_roughPadded = self.crop_local_maps(self.path)

    # load txt map
    def load_map(self, m_type):
        """
        Function is used to load the map from ".txt" file
        :param m_type: type of the map
        :return: Maps in type: txt, bw, grid
        """
        # map name 'map_<size>_<seed>.txt'
        map_name = '_'.join(['map', str(self.maze_size), str(self.maze_seed)]) + '.txt'
        # path to the maps folder
        map_path = str(Path(__file__).parent.parent) + '/maps/train/' + map_name
        # load map
        if m_type == 'txt':  # txt map
            with open(map_path, 'r') as f:
                map_txt = f.readlines()
                for i_idx, l in enumerate(map_txt):
                    for j_idx, s in enumerate(l):
                        if s == ' ':
                            self.valid_pos.append([i_idx, j_idx])
            return map_txt
        elif m_type == 'bw':
            rescaled_init, rescaled_goal, map_bw = self.map_txt2bw(map_path, self.maze_size)
            return rescaled_init, rescaled_goal, map_bw
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

                if i == self.goal_pos[0] and j == self.goal_pos[1]:
                    # goal is set to be darker: 0.2
                    for k in range(goal_upscale):
                        # mark goal with x
                        bw_img_data[(i * vertical_upscale + k)][j * horizon_upscale + k] = 0.2
                        bw_img_data[(i * vertical_upscale + goal_upscale - 1 - k)][j * horizon_upscale + k] = 0.2

                if i == self.init_pos[0] and j == self.init_pos[1]:
                    # init_pose.append()
                    # initial is set to be lighter: 0.8
                    for k in range(start_upscale):
                        # mark goal with x
                        bw_img_data[(i * vertical_upscale + k)][j * horizon_upscale + k] = 0.8
                        bw_img_data[(i * vertical_upscale + goal_upscale - 1 - k)][j * horizon_upscale + k] = 0.8
        f.close()

        # rescale the init and goal positions and randomly sample them
        rescaled_init = [self.init_pos[0] * vertical_upscale + np.random.randint(start_upscale), \
                         self.init_pos[1] * horizon_upscale + np.random.randint(start_upscale)]
        rescaled_goal = [self.goal_pos[0] * vertical_upscale + np.random.randint(goal_upscale), \
                         self.goal_pos[1] * horizon_upscale + np.random.randint(goal_upscale)]

        return rescaled_init, rescaled_goal, np.asarray(bw_img_data)

    # build grid world
    @ staticmethod
    def map_bw2grid(bw_map):
        """
        Function is used to build the grid map from the binary map
        """
        grid_map = np.copy(bw_map)
        row, col = grid_map.shape
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
        path = searchAlg.A_star(self.map2d_grid, self.init_pos, self.goal_pos)
        path_map = self.show_path(path, self.map2d_grid)
        return path, path_map

    # display the path on the map
    @ staticmethod
    def show_path(path, grid_map):
        """
        Function is used to print path on the binary map
        """
        # flip the grid_map to be binary map
        assert len(path) > 0
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

    def cropper(self, map_img, pos):
        # update the position based on padding
        pad_step = int((self.loc_map_size - 1) / 2)
        pad_pos = pos + [pad_step, pad_step]
        # crop one local map
        loc_map = map_img[(pad_pos[0] - pad_step):(pad_pos[0] + pad_step + 1), \
                    (pad_pos[1] - pad_step):(pad_pos[1] + pad_step + 1)]
        return loc_map

    def crop_local_maps(self, path):
        # store the local maps
        local_maps = []
        # padding the rough map based on crop size
        pad_step = int((self.loc_map_size - 1) / 2)
        # create background
        pad_map_size = int(self.map2d_rough.shape[0] + 2 * pad_step)
        pad_map = np.zeros((pad_map_size, pad_map_size))
        # insert the rough map
        pad_map[pad_step:pad_map_size-pad_step, pad_step:pad_map_size-pad_step] = self.map2d_rough

        for idx, pos in enumerate(path):
            local_maps.append(self.cropper(pad_map, pos))

        return local_maps, pad_map

    def get_start_goal_pos(self):
        pos_params = [self.init_pos[0],
                      self.init_pos[1],
                      self.goal_pos[0],
                      self.goal_pos[1],
                      0]  # [init_pos, goal_pos, init_orientation]
        return pos_params

    def sample_start_goal_pos(self, fix_init, fix_goal):
        # sample the start position
        init_positions = list(self.valid_pos)
        init_positions.remove(self.goal_pos)
        tmp_init_pos = self.init_pos if fix_init else random.sample(self.valid_pos, 1)[0]
        # sample the goal position
        goal_positions = list(self.valid_pos)
        goal_positions.remove(tmp_init_pos)
        tmp_goal_pos = self.goal_pos if fix_goal else random.sample(goal_positions, 1)[0]
        return tmp_init_pos, tmp_goal_pos

    def sample_path_goal(self, current_goal, step):
        # obtain all the positions on the path
        positions_on_path = [pos.tolist() for pos in self.path]
        # compute the from index and the to index
        current_goal_index = positions_on_path.index(current_goal)
        from_idx = 1 if current_goal_index - step < 1 else current_goal_index - step
        to_idx = len(self.path) - 1 if current_goal_index + step > len(self.path) - 1 else current_goal_index + step
        # sample a new goal
        new_goal = random.sample(self.path[from_idx:to_idx+1], 1)
        self.goal_pos = new_goal[0].tolist()
        return self.goal_pos

    def sample_path_next_goal(self, step):
        assert (step > 0), f"Invalid step size. Step should be bigger than 0."
        # obtain all the positions on the path
        positions_on_path = [pos.tolist() for pos in self.path]
        # sample the next goal
        new_goal = positions_on_path[step] if step < len(self.path) else positions_on_path[-1]
        return new_goal

    def sample_path_next_pair(self, dist):
        # length
        length = len(self.path)
        # obtain all the positions on the path
        positions_on_path = [pos.tolist() for pos in self.path]
        # sample a start position
        start_idx = np.random.choice(len(self.path), 1).item()
        # sample a pair from the path
        if np.random.sample() < 0.5:
            # select goal on the left
            if start_idx == 0:
                goal_idx = start_idx + dist if start_idx + dist < len(self.path) else len(self.path) - 1
            else:
                goal_idx = 0 if start_idx - dist <= 0 else start_idx - dist
        else:
            # select goal on the right
            if start_idx == length - 1:
                goal_idx = 0 if start_idx - dist <= 0 else start_idx - dist
            else:
                goal_idx = start_idx + dist if start_idx + dist < len(self.path) else len(self.path) - 1
        return positions_on_path[start_idx], positions_on_path[goal_idx]



