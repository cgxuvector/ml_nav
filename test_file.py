import numpy as np
import pickle
from utils.mapper import RoughMap
import matplotlib.pyplot as plt
from utils import searchAlg


def load_pair_data(m_size, m_seed):
    path = f'./maps/2dmap/maze_{m_size}_{m_seed}.pkl'
    f = open(path, 'rb')
    return pickle.load(f)


def is_intersection(pos, m_map):
    up_pos = [pos[0]-1, pos[1]]
    down_pos = [pos[0]+1, pos[1]]
    left_pos = [pos[0], pos[1]-1]
    right_pos = [pos[0], pos[1]+1]
    count = 0
    if m_map[up_pos[0], up_pos[1]] > 0:
        count += 1
    if m_map[down_pos[0], down_pos[1]] > 0:
        count += 1
    if m_map[left_pos[0], left_pos[1]] > 0:
        count += 1
    if m_map[right_pos[0], right_pos[1]] > 0:
        count += 1
    if count > 2:
        return True
    else:
        return False


def analyze_maze_complexity_from_2d_map(m_size, m_seed):
    # load the map
    maze_mapper = RoughMap(m_size, m_seed, 3)
    plt.title(f"{maze_size}-{m_seed}")
    plt.imshow(maze_mapper.map2d_rough)
    plt.show()
    print("Maze size: ", m_size)
    # find the longest distance
    pair_data = load_pair_data(m_size, m_seed)
    dist_list = [int(k) for k in pair_data.keys()]
    max_dist = dist_list[-1]
    print("Maze longest distance: ", max_dist)
    # find the intersection
    intersection_num = 0
    for pos in maze_mapper.valid_pos:
        if is_intersection(pos, maze_mapper.map2d_bw):
            intersection_num += 1
    print("Maze intersection number: ", intersection_num)
    return max_dist, intersection_num


def draw_fail_local_pair(map_obj, m_size, m_seed, s_pos, g_pos):
    # crop the local map
    local_maps, _ = map_obj.crop_local_maps([np.array(s_pos), np.array(g_pos)])

    fig, ax = plt.subplots(1, 2)
    plt.title(f"{m_size}-{m_seed}")
    ax[0].axis('off')
    ax[0].set_title(f"{s_pos}")
    ax[0].imshow(local_maps[0])
    ax[1].axis('off')
    ax[1].set_title(f"{g_pos}")
    ax[1].imshow(local_maps[1])
    # plt.show()
    plt.savefig(f'../ml_nav_eval/corl_results/9-1/fail-case-imgs/{m_size}-{m_seed}-{s_pos}-{g_pos}.png', dpi=50)


def patch_is_exist(s_patch, g_patch, path_dict):
    # if the dict is empty
    if len(path_dict['start']) == 0:
        return False

    # loop all the existing pairs
    item_num = len(path_dict['start'])
    for idx in range(item_num):
        # check the identity
        is_s_same = np.array_equal(s_patch, path_dict['start'][idx])
        is_g_same = np.array_equal(g_patch, path_dict['goal'][idx])

        # if find return True
        if is_s_same and is_g_same:
            return True

    return False


def analyze_local_patches_diversity(m_size, m_seed):
    # load the map
    env_map = RoughMap(m_size, maze_seed, 3)
    # load the pair data
    total_pairs = load_pair_data(m_size, m_seed)
    target_pairs = total_pairs['1']
    # loop all local patterns
    local_patches = {'start': [], 'goal': []}
    for s_pos, g_pos in zip(target_pairs[0], target_pairs[1]):
        # get the local patches
        s_patch = env_map.cropper(env_map.map2d_roughPadded, s_pos)
        g_patch = env_map.cropper(env_map.map2d_roughPadded, g_pos)
        # check if the pair exists in local patches
        if patch_is_exist(s_patch, g_patch, local_patches):
            continue
        else:
            local_patches['start'].append(s_patch)
            local_patches['goal'].append(g_patch)

        # get the local patches
        g_patch = env_map.cropper(env_map.map2d_roughPadded, s_pos)
        s_patch = env_map.cropper(env_map.map2d_roughPadded, g_pos)
        # check if the pair exists in local patches
        if patch_is_exist(s_patch, g_patch, local_patches):
            continue
        else:
            local_patches['start'].append(s_patch)
            local_patches['goal'].append(g_patch)

    print(f"Total pairs = {len(target_pairs[0]) * 2},"
          f" Different pairs = {len(local_patches['start'])},"
          f" ratio = {len(local_patches['start']) / (len(target_pairs[0]) * 2)}")

    return local_patches


def get_action(s_pos, g_pos):
    # compute the difference
    diff = np.array(g_pos) - np.array(s_pos)

    if np.array_equal(diff, np.array([1, 0])):
        action = 'down'
    elif np.array_equal(diff, np.array([-1, 0])):
        action = 'up'
    elif np.array_equal(diff, np.array([0, 1])):
        action = 'right'
    else:
        action = 'left'

    return action


def analyze_local_patches_action_based_diversity(m_size, m_seed):
    # load the map
    env_map = RoughMap(m_size, maze_seed, 3)
    # load the pair data
    total_pairs = load_pair_data(m_size, m_seed)
    target_pairs = total_pairs['1']
    # loop all local patterns
    local_patches = {'left': {'start': [], 'goal': [], 'num': 0, 's_pos': [], 'g_pos': []},
                     'right': {'start': [], 'goal': [], 'num': 0, 's_pos': [], 'g_pos': []},
                     'up': {'start': [], 'goal': [], 'num': 0, 's_pos': [], 'g_pos': []},
                     'down': {'start': [], 'goal': [], 'num': 0, 's_pos': [], 'g_pos': []}}
    for s_pos, g_pos in zip(target_pairs[0], target_pairs[1]):
        # get the action
        action = get_action(s_pos, g_pos)
        # get the local patches
        s_patch = env_map.cropper(env_map.map2d_roughPadded, s_pos)
        g_patch = env_map.cropper(env_map.map2d_roughPadded, g_pos)
        # check if the pair exists in local patches
        if patch_is_exist(s_patch, g_patch, local_patches[action]):
            continue
        else:
            local_patches[action]['start'].append(s_patch)
            local_patches[action]['goal'].append(g_patch)
            local_patches[action]['num'] += 1
            local_patches[action]['s_pos'].append(s_pos)
            local_patches[action]['g_pos'].append(g_pos)

        # ge the action
        action = get_action(g_pos, s_pos)
        # get the local patches
        g_patch = env_map.cropper(env_map.map2d_roughPadded, s_pos)
        s_patch = env_map.cropper(env_map.map2d_roughPadded, g_pos)
        # check if the pair exists in local patches
        if patch_is_exist(s_patch, g_patch, local_patches[action]):
            continue
        else:
            local_patches[action]['start'].append(s_patch)
            local_patches[action]['goal'].append(g_patch)
            local_patches[action]['num'] += 1
            local_patches[action]['s_pos'].append(g_pos)
            local_patches[action]['g_pos'].append(s_pos)

    diff_patches_pair_num = local_patches['right']['num'] + local_patches['left']['num'] + local_patches['up']['num'] + local_patches['down']['num']

    print(f"Total pairs = {len(target_pairs[0]) * 2},"
          f" Different pairs = {diff_patches_pair_num},"
          f" Ratio = {diff_patches_pair_num / (len(target_pairs[0]) * 2)},"
          f" R num = {local_patches['right']['num']},"
          f" L num = {local_patches['left']['num']},"
          f" U num = {local_patches['up']['num']},"
          f" D num = {local_patches['down']['num']}")

    return local_patches


def find_identical_local_patches_in_diff_actions(local_patches):
    # action list
    action_list = ['up', 'down', 'left', 'right']
    identical_patches = {'start': [], 'goal': [], 'act': [], 'pos': []}

    # loop for actions
    for act_idx in range(3):
        # loop the elements in action
        elem_num = local_patches[action_list[act_idx]]['num']
        for idx in range(elem_num):
            # obtain the patches
            s_patch = local_patches[action_list[act_idx]]['start'][idx]
            g_patch = local_patches[action_list[act_idx]]['goal'][idx]
            # for next action
            for next_act_idx in range(act_idx + 1, 4, 1):
                if patch_is_exist(s_patch, g_patch, local_patches[action_list[next_act_idx]]):
                    identical_patches['start'].append(s_patch)
                    identical_patches['goal'].append(g_patch)
                    identical_patches['act'].append([action_list[act_idx], action_list[next_act_idx]])
                    identical_patches['pos'].append([local_patches[action_list[act_idx]]['s_pos'][idx],
                                                     local_patches[action_list[act_idx]]['g_pos'][idx]])

    return identical_patches


def visual_local_patches(patch_dict):
    # loop all the distinguished patches
    elem_num = len(patch_dict['start'])
    print(elem_num)
    for idx in range(elem_num):
        # define the figure
        fig, ax = plt.subplots(1, 2)
        # obtain a pair of patches
        s_patch = patch_dict['start'][idx]
        g_patch = patch_dict['goal'][idx]
        ax[0].axis('off')
        ax[0].set_title("start")
        ax[0].imshow(s_patch)
        ax[1].axis('off')
        ax[1].set_title('goal')
        ax[1].imshow(g_patch)
        plt.show()
        print(patch_dict['act'][idx], patch_dict['pos'])


if __name__ == '__main__':
    # """Analyze maze complexity"""
    # maze_size = 15
    # maze_seed = 0
    # maze_max_dist_list = []
    # maze_intersection_num_list = []
    # for maze_seed in range(20):
    #     print(f"{maze_size}-{maze_seed}")
    #     dist, inter_num = analyze_maze_complexity_from_2d_map(maze_size, maze_seed)
    #     maze_max_dist_list.append(dist)
    #     maze_intersection_num_list.append(inter_num)
    #     print("--------------------------")
    # print(f"Max dist = {max(maze_max_dist_list)}, min dist = {min(maze_max_dist_list)}")
    # print(f"Max intersection = {max(maze_intersection_num_list)}, min intersection = {min(maze_intersection_num_list)}")
    # # compute complexity
    # idx = 0
    # min_max_dist = min(maze_max_dist_list)
    # max_max_dist = max(maze_max_dist_list)
    # min_inter_num = min(maze_intersection_num_list)
    # max_inter_num = max(maze_intersection_num_list)
    # maze_complexity = []
    # ratio_size = (maze_size - 5) / (21 - 5)
    # for dist, inter_num in zip(maze_max_dist_list, maze_intersection_num_list):
    #     ratio_dist = (dist - min_max_dist) / (max_max_dist - min_max_dist)
    #     ratio_inter = 1 - ((inter_num - min_inter_num) / (max_inter_num - min_inter_num))
    #     print(f"Maze {idx}: {ratio_dist} - {ratio_inter}")
    #     idx += 1
    #     maze_complexity.append((0.33 * ratio_size + 0.33 * ratio_dist + 0.33 * ratio_inter))
    #     print(f'{idx - 1}: {maze_complexity[-1]}')
    # # plot the bar
    # x = list(range(len(maze_complexity)))
    # plt.title(f"Maze complexity analysis of maze size {maze_size}")
    # plt.ylabel('Maze complexity')
    # plt.ylim(0, 1)
    # plt.xlabel('Maze ID')
    # plt.bar(x, maze_complexity)
    # plt.xticks(x, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    #                '11', '12', '13', '14', '15', '16', '17', '18', '19'))
    # plt.show()

    # """ Crop wrong local map """
    # maze_size = 21
    # maze_seed = 12
    #
    # env_map = RoughMap(maze_size, maze_seed, 3)
    #
    # fail_pairs = np.load(f'../ml_nav_eval/corl_results/9-1/{maze_size}-{maze_seed}-Fail-pos.npy').tolist()
    #
    # for pair in fail_pairs:
    #     draw_fail_local_pair(env_map, maze_size, maze_seed, pair[0], pair[1])

    """ Analyze local diversity """
    maze_size = 13
    maze_seed = 17

    patch_dict = analyze_local_patches_action_based_diversity(maze_size, maze_seed)
    identical_patches = find_identical_local_patches_in_diff_actions(patch_dict)
    visual_local_patches(identical_patches)
    # visual_local_patches(patch_dict)




