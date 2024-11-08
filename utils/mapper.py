import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import copy
from utils import searchAlg
import IPython.terminal.debugger as Debug


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
    def __init__(self, m_size, m_seed, loc_map_size=3, use_apple_reward=False):
        """
        Function is used to initialize the current map.
        :param m_size: size of the maze
        :param m_seed: seed of the maze
        :param loc_map_size: size of the local map
        """
        # flag of using intrinsic apple reward
        self.use_apple_reward = use_apple_reward

        # map parameters
        self.maze_size = m_size
        self.maze_seed = m_seed
        self.distortion_factor = [1, 1, 1, 1]  # scaling along [horizon, vertical, goal, start]

        # properties
        self.loc_map_size = loc_map_size

        # txt map and primary valid positions and primary valid local maps
        self.valid_pos = []  # walkable positions in the txt map
        self.wall_pos = []  # wall positions
        self.valid_loc_maps = []  # local map for the walkable positions
        self.map2d_txt = self.load_map('txt')  # map represented as the raw txt file
        self.init_pos = self.valid_pos[0]
        self.goal_pos = self.valid_pos[-1]

        # obtain bw map and randomly selected initial and target goals
        self.init_pos, self.goal_pos, self.map2d_bw = self.load_map('bw')

        # obtain the grid world map to do the planning
        self.map2d_grid = self.load_map('grid')
        self.path, self.map2d_path = self.generate_path(self.init_pos, self.goal_pos)
        self.dist_matrix = None
        # egocentric actionsd
        self.map_act, self.ego_act = self.path2egoaction(self.path)
        # egocentric local maps (size adjustable)
        self.map2d_rough = np.ones(self.map2d_grid.shape) - self.map2d_grid
        self.local_maps, self.map2d_roughPadded = self.crop_local_maps(self.path)

        # define the shuffle map
        self.map2d_imprecise = None
        self.map2d_rough_imprecise = None
        self.map2d_roughpad_imprecise = None
        # define the imprecise valid position
        self.imprecise_valid_pos = None
        self.imprecise_wall_pos = None

    @staticmethod
    def remove_apples(txt):
        txt_apple_removed = []
        for l in txt:
            txt_apple_removed.append(l.replace('A', ' ', len(l)))
        return txt_apple_removed

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
                map_txt = map_txt if self.use_apple_reward else self.remove_apples(map_txt)
                for i_idx, l in enumerate(map_txt):
                    for j_idx, s in enumerate(l):
                        if s != '\n' and s != '*':
                            self.valid_pos.append([i_idx, j_idx])

                        if s == '*' and 0 < i_idx < self.maze_size - 1 and 0 < j_idx < self.maze_size - 1:
                            self.wall_pos.append([i_idx, j_idx])
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
    @staticmethod
    def map_grid2bw(grid_map):
        """
        Function is used to build the grid map from the binary map
        """
        bw_map = np.copy(grid_map)
        row, col = bw_map.shape
        # obtain the grid map
        for r in range(row):
            for c in range(col):
                if bw_map[r, c] < 0.3:
                    bw_map[r, c] = 1
                else:
                    bw_map[r, c] = 0.0
        return bw_map

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

    def generate_path(self, init_pos, goal_pos):
        path = searchAlg.A_star(self.map2d_grid, init_pos, goal_pos)
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
        pad_pos = np.array(pos) + np.array([pad_step, pad_step])
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

    # get the init and goal positions
    def get_start_goal_pos(self):
        pos_params = [self.init_pos[0],
                      self.init_pos[1],
                      self.goal_pos[0],
                      self.goal_pos[1],
                      0,
                      0]  # [init_pos, goal_pos, init_orientation]
        return pos_params

    # sample the init and goal positions from the valid positions
    def sample_start_goal_pos_with_random_dist(self, fix_init, fix_goal):
        """
        Function is used to sample a global pair of init and goal positions from the valid positions.
        :param fix_init: If it is True, the init position is fixed.
        :param fix_goal: If it is True, the goal position is fixed.
        :return: new sampled init and goal positions.
        """
        # obtain valid positions
        init_positions = list(self.valid_pos)
        # candidate goal positions for fixed goal
        if fix_goal:
            init_positions.remove(self.goal_pos)
        tmp_init_pos = self.init_pos if fix_init else random.sample(init_positions, 1)[0]
        # sample the goal position
        goal_positions = list(self.valid_pos)
        goal_positions.remove(tmp_init_pos)
        tmp_goal_pos = self.goal_pos if fix_goal else random.sample(goal_positions, 1)[0]
        # update the mapper
        self.update_mapper(tmp_init_pos, tmp_goal_pos)
        return tmp_init_pos, tmp_goal_pos

    def sample_fixed_distance(self, start_pos, valid_pos_list, dist):
        # set the temporal start and goal positions
        tmp_init_pos = start_pos
        tmp_goal_pos = random.sample(valid_pos_list, 1)[0]
        # plan a new path
        pos_path = searchAlg.A_star(self.map2d_grid, tmp_init_pos, tmp_goal_pos)
        if dist != -1:
            # counter
            sample_clk = 0
            while len(pos_path) < dist + 1:
                # sample a goal position
                tmp_goal_pos = random.sample(valid_pos_list, 1)[0]
                # plan a new path
                pos_path = searchAlg.A_star(self.map2d_grid, tmp_init_pos, tmp_goal_pos)
                # increase one step
                sample_clk += 1
                if sample_clk > 100:
                    # simplify the distance by one
                    dist -= 1
                    # reset the clock
                    sample_clk = 0
        # sample the init and goal along the trajectory
        valid_path_pos = [pos.tolist() for pos in pos_path]
        goal_pos = valid_path_pos[dist] if dist < len(valid_path_pos) else valid_path_pos[-1] 
        return goal_pos

    def sample_start_goal_pos_with_maximal_dist(self, fix_init, fix_goal, dist):
        """
        Function is used to sample a random pair init and goal positions from the valid positions.
        :param fix_init: If it is True, the init position is fixed.
        :param fix_goal: If it is True, the goal position is fixed.
        :param dist: range to sample the next step
        :return: new sampled init and goal positions.
        """
        # obtain valid initial positions
        if fix_init is True and fix_goal is True:
            new_init = self.init_pos
            new_goal = self.goal_pos
        elif fix_init is True and fix_goal is False:
            new_init = self.init_pos
            valid_positions = self.valid_pos.copy()
            valid_positions.remove(new_init)
            new_goal = self.sample_fixed_distance(new_init, valid_positions, dist)
        elif fix_init is False and fix_goal is True:
            new_goal = self.goal_pos
            valid_positions = self.valid_pos.copy()
            valid_positions.remove(new_goal)
            new_init = self.sample_fixed_distance(new_goal, valid_positions, dist)
        else:
            valid_positions = self.valid_pos.copy()
            new_init = random.sample(valid_positions, 1)[0]
            valid_positions.remove(new_init)
            new_goal = self.sample_fixed_distance(new_init, valid_positions, dist)

        # update the mapper
        self.update_mapper(new_init, new_goal)
        return new_init, new_goal

    def update_mapper(self, new_init, new_goal):
        # clear the old the binary map
        self.map2d_bw[self.init_pos[0], self.init_pos[1]] = 1
        self.map2d_bw[self.goal_pos[0], self.goal_pos[1]] = 1
        # reset the init and goal positions on the map
        self.map2d_bw[new_init[0], new_init[1]] = 0.8
        self.map2d_bw[new_goal[0], new_goal[1]] = 0.2
        # update the init and goal position
        self.init_pos = new_init
        self.goal_pos = new_goal
        # update the path using the new init and goal positions
        self.path, self.map2d_path = self.generate_path(self.init_pos, self.goal_pos)
        # update the action
        self.map_act, self.ego_act = self.path2egoaction(self.path)

    def get_start_goal_pair_with_fix_distance(self, dist):
        # if distance matrix is empty, build the distance matrix
        if self.dist_matrix is None:
            # build a graph
            valid_pos_num = len(self.valid_pos)
            dist_matrix = -1 * np.ones((valid_pos_num, valid_pos_num))
            for i in range(valid_pos_num):
                for j in range(i, valid_pos_num):
                    path = searchAlg.A_star(self.map2d_grid, self.valid_pos[i], self.valid_pos[j])
                    if path is None:
                        continue
                    dist_matrix[i, j] = len(path) - 1
            self.dist_matrix = dist_matrix

        pairs = np.where(self.dist_matrix == dist)
        start_pos_list = [self.valid_pos[idx] for idx in pairs[0].tolist()] if len(pairs[0]) > 0 else None
        goal_pos_list = [self.valid_pos[idx] for idx in pairs[1].tolist()] if len(pairs[1]) > 0 else None
        return {'start': start_pos_list, 'goal': goal_pos_list}

    def shuffle_map(self, ratio, shuffle_mode='wall2corridor'):
        # copy the accurate map
        self.map2d_imprecise = copy.deepcopy(self.map2d_grid)
        self.imprecise_valid_pos = copy.deepcopy(self.valid_pos)
        self.imprecise_wall_pos = copy.deepcopy(self.wall_pos)

        # compute the shuffle number
        if shuffle_mode == 'wall2corridor':
            # shuffle number
            shuffle_num = int(len(self.imprecise_wall_pos) * ratio)
            pos_candidates = self.imprecise_wall_pos
        elif shuffle_mode == 'corridor2wall':
            # shuffle number
            shuffle_num = int(len(self.imprecise_valid_pos) * ratio)
            pos_candidates = self.imprecise_valid_pos
        elif shuffle_mode == 'mixed':
            # mixture
            shuffle_num = int(((self.maze_size - 2)**2) * ratio)
            pos_candidates = self.imprecise_valid_pos + self.imprecise_wall_pos
        else:
            raise Exception("Wrong shuffle mode")

        # sample the item
        positions = random.sample(pos_candidates, shuffle_num)
        # shuffle the positions
        for pos in positions:
            if shuffle_mode == 'wall2corridor':
                if random.uniform(0, 1) < 0.5:
                    # add into corridor
                    self.imprecise_valid_pos.append(pos)
                    # remove from wall
                    self.imprecise_wall_pos.remove(pos)
                    # change the imprecise map
                    self.map2d_imprecise[pos[0], pos[1]] = 0.0
            elif shuffle_mode == 'corridor2wall':
                if random.uniform(0, 1) < 0.5:
                    # add into corridor
                    self.imprecise_wall_pos.append(pos)
                    # remove from wall
                    self.imprecise_valid_pos.remove(pos)
                    # change the imprecise map
                    self.map2d_imprecise[pos[0], pos[1]] = 1.0
            elif shuffle_mode == 'mixed':
                if random.uniform(0, 1) < 0.5:
                    if pos in self.imprecise_valid_pos:
                        continue
                    else:
                        # add into corridor
                        self.imprecise_valid_pos.append(pos)
                        # remove from wall
                        self.imprecise_wall_pos.remove(pos)
                        # change the imprecise map
                        self.map2d_imprecise[pos[0], pos[1]] = 0.0
                else:
                    if pos in self.imprecise_wall_pos:
                        continue
                    else:
                        # add into corridor
                        self.imprecise_wall_pos.append(pos)
                        # remove from wall
                        self.imprecise_valid_pos.remove(pos)
                        # change the imprecise map
                        self.map2d_imprecise[pos[0], pos[1]] = 1.0
            else:
                raise Exception("Wrong shuffle mode")

        self.map2d_rough_imprecise = np.ones(self.map2d_imprecise.shape) - self.map2d_imprecise
        # pad the map
        pad_step = int((self.loc_map_size - 1) / 2)
        # create background
        pad_map_size = int(self.map2d_rough_imprecise.shape[0] + 2 * pad_step)
        self.map2d_roughpad_imprecise = np.zeros((pad_map_size, pad_map_size))
        # insert the rough map
        self.map2d_roughpad_imprecise[pad_step:pad_map_size-pad_step, pad_step:pad_map_size-pad_step] = self.map2d_rough_imprecise

        print(self.map2d_grid)
        print(self.map2d_imprecise)

#
# env_map = RoughMap(7, 0, 3)
# print(env_map.map_act)
# plt.imshow(env_map.map2d_rough)
# print(len(env_m ap.path))
# plt.show()


# for seed in range(20):
#     env_map = RoughMap(7, seed, 3)
#     plt.title(f"7-{seed}")
#     plt.imshow(env_map.map2d_rough)
#     plt.axis('off')
#     plt.show()
#
# for lm in env_map.local_maps:
#     plt.imshow(lm)
#     plt.show()

# path_1 = [np.array([1, 11]), np.array([1, 10]), np.array([1, 9]), np.array([2, 9]), np.array([3, 9]),
#           np.array([16, 3]), np.array([17, 3]), np.array([17, 3]), np.array([17, 4]), np.array([17, 5]),
#           np.array([17, 6]), np.array([16, 6]), np.array([17, 6]), np.array([17, 7]), np.array([17, 8]),
#           np.array([17, 9]), np.array([16, 9]), np.array([15, 9]), np.array([14, 9]), np.array([13, 9]),
#           np.array([12, 9]), np.array([11, 9]), np.array([10, 9]), np.array([9, 9]), np.array([9, 8]),
#           np.array([8, 8]), np.array([9, 8]), np.array([10, 9]), np.array([10, 10]), np.array([10, 11]),
#           np.array([9, 11]), np.array([10, 11]), np.array([9, 11]), np.array([8, 11]), np.array([7, 11]),
#           np.array([7, 12]), np.array([7, 13]), np.array([8, 13]), np.array([9, 13]), np.array([9, 14]),
#           np.array([9, 15]), np.array([8, 15]), np.array([7, 15]), np.array([6, 15]), np.array([5, 15]),
#           np.array([4, 15]), np.array([3, 15]), np.array([3, 14]), np.array([3, 13]), np.array([3, 12]),
#           np.array([3, 11]), np.array([3, 10]), np.array([3, 9]), np.array([3, 8]), np.array([3, 7]),
#           np.array([3, 6]), np.array([3, 5])]
#
# env_map.map2d_path = env_map.show_path(path_1, env_map.map2d_grid)
# plt.axis('off')

# plot avoid ill learned behavior
# env_map.update_mapper([17, 1], [3, 5])
# path_1 = [np.array([17, 1]), np.array([16, 1]), np.array([15, 1]), np.array([15, 2]), np.array([15, 3]),
#           np.array([16, 3]), np.array([17, 3]), np.array([17, 3]), np.array([17, 4]), np.array([17, 5]),
#           np.array([17, 6]), np.array([16, 6]), np.array([17, 6]), np.array([17, 7]), np.array([17, 8]),
#           np.array([17, 9]), np.array([16, 9]), np.array([15, 9]), np.array([14, 9]), np.array([13, 9]),
#           np.array([12, 9]), np.array([11, 9]), np.array([10, 9]), np.array([9, 9]), np.array([9, 8]),
#           np.array([8, 8]), np.array([9, 8]), np.array([10, 9]), np.array([10, 10]), np.array([10, 11]),
#           np.array([9, 11]), np.array([10, 11]), np.array([9, 11]), np.array([8, 11]), np.array([7, 11]),
#           np.array([7, 12]), np.array([7, 13]), np.array([8, 13]), np.array([9, 13]), np.array([9, 14]),
#           np.array([9, 15]), np.array([8, 15]), np.array([7, 15]), np.array([6, 15]), np.array([5, 15]),
#           np.array([4, 15]), np.array([3, 15]), np.array([3, 14]), np.array([3, 13]), np.array([3, 12]),
#           np.array([3, 11]), np.array([3, 10]), np.array([3, 9]), np.array([3, 8]), np.array([3, 7]),
#           np.array([3, 6]), np.array([3, 5])]
#
# env_map.map2d_path = env_map.show_path(path_1, env_map.map2d_grid)
# plt.axis('off')
# plt.imshow(env_map.map2d_path)
# plt.savefig(f'19-17_map_optimal_path_3.png', dpi=100)
# plt.show()
# for size in size_list:
#     for seed in seed_list:
#         # for i in range(2000):
#         env_map = RoughMap(size, seed, 3)
#         #     init_pos, goal_pos = env_map.sample_random_start_goal_pos(False, False, dist)
#         #     print(f"Run {i+1}: Start = {init_pos}, Goal = {goal_pos}, Target dist = {dist}, Dist = {len(env_map.path) - 1}")
#         #     if (len(env_map.path) - 1) > dist or init_pos == goal_pos:
#         #         print("Fail case", init_pos, goal_pos)
#         #         break
#         plt.title(f"{size}-{seed}-map")
#         plt.axis('off')
#         plt.imshow(env_map.map2d_rough)
#         plt.savefig(f'{size}x{seed}_map.png', dpi=100)
#         plt.show()

