import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from pathlib import Path

from utils import searchAlg


# define a class for the rough map
class RoughMap(object):

    # init function
    def __init__(self, m_size, m_seed):
        # map parameters
        self.maze_size = m_size
        self.maze_seed = m_seed
        self.distortion_factor = [4, 4, 2, 2]  # scaling along [horizon, vertical, goal, start]

        # maps
        self.map2d_txt = self.load_map('txt')
        self.map2d_bw = self.load_map('bw')
        self.pos, self.map2d_grid = self.load_map('grid')

    # load txt map
    def load_map(self, m_type):
        # file name
        map_name = '_'.join(['map', str(self.maze_size), str(self.maze_seed)]) + '.txt'
        # file path
        map_path = str(Path(__file__).parent.parent) + '/maps/train/' + map_name
        # load map
        if m_type == 'txt':
            with open(map_path, 'r') as f:
                map_txt = f.readlines()
            return map_txt
        elif m_type == 'bw':
            map_bw = self.map_txt2bw(map_path, self.maze_size)
            return map_bw
        elif m_type == 'grid':
            map_pos, map_grid = self.map_bw2grid(self.map2d_bw)
            return map_pos, map_grid
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

        image_h_size = map_size * horizon_upscale
        image_v_size = map_size * vertical_upscale

        # initalize white field
        bw_img_data = [[1.0 for _ in range(image_h_size)] for _ in range(image_v_size)]  # create a while background
        f = open(map_name, 'r')
        for i, line in enumerate(f):
            for j, char in enumerate(line):
                if char == '*':
                    for v in range(vertical_upscale):
                        for h in range(horizon_upscale):
                            # make walls black
                            bw_img_data[(i * vertical_upscale + v)][j * horizon_upscale + h] = 0

                if char == 'G':
                    for k in range(goal_upscale):
                        # mark goal with x
                        bw_img_data[(i * vertical_upscale + k)][j * horizon_upscale + k] = 0.7
                        bw_img_data[(i * vertical_upscale + goal_upscale - 1 - k)][j * horizon_upscale + k] = 0.7

                if char == 'P':
                    for k in range(goal_upscale):
                        # mark goal with x
                        bw_img_data[(i * vertical_upscale + k)][j * horizon_upscale + k] = 0.3
                        bw_img_data[(i * vertical_upscale + goal_upscale - 1 - k)][j * horizon_upscale + k] = 0.3
        f.close()

        return np.asarray(bw_img_data)

    # build grid world
    def map_bw2grid(self, bw_map):
        """
        Function is used to build the grid map from the binary map
        """
        grid_map = resize(bw_map, (bw_map.shape[0] // 2, bw_map.shape[1] // 2))

        row = grid_map.shape[0]
        col = grid_map.shape[1]
        # obtain the grid map
        for r in range(row):
            for c in range(col):
                if grid_map[r, c] < 0.3:
                    grid_map[r, c] = 1
                elif grid_map[r, c] > 0.8:
                    grid_map[r, c] = 0.0
                else:
                    grid_map[r, c] = 0.5

        # obtain the start and end position
        pos = []
        for r in range(row):
            for c in range(col):
                if grid_map[r, c] == 0.5:
                    pos.append([r, c])

        return pos, grid_map

    # display the map: txt map or occupancy map
    def show_map(self, m_type):
        """
        Function is used to show the binary map
        """
        # add a function that plots all the maps.
        if m_type == 'txt':
            for line in self.map2d_txt:
                print(line)
        elif m_type == 'bw':
            plt.imshow(self.map2d_bw, cmap=plt.cm.gray)
            plt.show()
        elif m_type == 'grid':
            plt.imshow(self.map2d_grid, cmap=plt.cm.gray)
            plt.show()
        else:
            raise Exception("Map Type Error: Invalid type \"{}\". Please select from \"txt\" , \"grid\" or \"bw\"". \
                            format(m_type))

    # display the path on the map
    def show_path(self, path, grid_map):
        # flip the grid_map to be binary map
        grid_map = np.array(grid_map)
        grid_map = np.ones(grid_map.shape) - grid_map
        for pos in path:
            pos = list(pos)
            grid_map[pos[0], pos[1]] = 0.5
        grid_map[path[0][0], path[0][1]] = 0.7
        grid_map[path[-1][0], path[-1][1]] = 0.7

        plt.imshow(grid_map)
        # plt.savefig('./plan_path.png', dpi=300)
        plt.show()

    # convert path to egocentric actions
    def path2egoaction(self, path, init_ori):
        action_list = []
        ori_list = [init_ori]

        current_ori = init_ori
        for idx, pos in enumerate(path):
            if idx < len(path) - 1:
                next_pos = path[idx + 1]
            else:
                break
            pos_dir = list(next_pos - pos)

            # potential action
            potential_act = None
            if pos_dir == [-1, 0]:
                potential_act = 'up'

            if pos_dir == [1, 0]:
                potential_act = 'down'

            if pos_dir == [0, -1]:
                potential_act = 'left-up'

            if pos_dir == [0, 1]:
                potential_act = 'right-up'

            # adjust compass
            act_name = ['up', 'right-up', 'down', 'left-up']
            normal_compass = [0, 1, 2, 3]
            current_compass = normal_compass.copy()
            move_step = act_name.index(current_ori)
            count = 0
            while count < move_step:
                head = current_compass.pop(0)
                current_compass.append(head)
                count += 1
            print("--------------------")
            print(current_ori)
            print(normal_compass)
            print(current_compass)

            # update the potential action to be the real action
            old_idx = act_name.index(potential_act)
            real_idx = current_compass.index(old_idx)
            action_list.append(act_name[real_idx])

            current_ori = potential_act
            ori_list.append(current_ori)

        # refine actions
        refined_action_list = []
        for act in action_list:
            if len(act) > 4:
                tmp = act.split('-')
                refined_action_list.append(tmp[0])
                refined_action_list.append(tmp[1])
            else:
                refined_action_list.append(act)

        print(action_list)
        return refined_action_list, ori_list

    def generate_path(self):
        path = searchAlg.A_star(self.map2d_grid, self.pos[0], self.pos[1])
        print(path)
        self.show_path(path, self.map2d_grid)

