import matplotlib.pyplot as plt
from skimage.transform import resize


def save_loc_maps_and_observations(size, seed, pos, loc_map, observations, observations_names):
    default_path = "/mnt/sda/dataset/ml_nav/loc_map_obs/"
    common_str = '_'.join([str(size), str(seed), str(pos[0]), str(pos[1])])
    # save the local map
    loc_map_name = '_'.join(['map', common_str]) + '.png'
    loc_map = resize(loc_map, (120, 120))
    plt.imsave(default_path + loc_map_name, loc_map, cmap=plt.cm.gray)
    # save the observations
    for idx, obs_name in enumerate(observations_names):
        if obs_name == '':
            continue
        img_name = '_'.join(['obs', common_str, obs_name]) + ".png"
        plt.imsave(default_path + img_name, observations[obs_name])