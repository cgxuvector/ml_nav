import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import torch
import IPython.terminal.debugger as Debug

plt.rcParams.update({'font.size': 15})


def save_loc_maps_and_observations(size, seed, pos, loc_map, observations, observations_names, obs_type):
    if obs_type == "random":
        default_path = "/mnt/sda/dataset/ml_nav/global_map_obs_various_texture/"
    elif obs_type == 'uniform':
        default_path = "/mnt/sda/dataset/ml_nav/global_map_obs_fixed_texture/"
    elif obs_type == 'uniform-small':
        default_path = "/mnt/sda/dataset/ml_nav/loc_map_obs_fixed_texture_small/"
    else:
        assert False, "Obs Type Error: Please input either 'various' or 'uniform'"

    common_str = '_'.join([str(size), str(seed), str(pos[0]), str(pos[1])])
    # save the local map
    loc_map_name = '_'.join(['map', common_str]) + '.png'
    io.imsave(default_path + loc_map_name, loc_map)
    # save the observations
    for idx, obs_name in enumerate(observations_names):
        if obs_name == '':
            continue
        img_name = '_'.join(['obs', common_str, obs_name]) + ".png"
        io.imsave(default_path + img_name, observations[idx])


def save_loss(loss_list, loss_name):
    root_dir = "/mnt/sda/dataset/ml_nav/VAE/plot/"
    fig, arr = plt.subplots(1, 1)
    fig.canvas.set_window_title(loss_name)
    arr.set_title(" ".join(loss_name.split("_")))
    arr.set_xlabel("Batch Iteration")
    arr.set_ylabel("Loss")
    # arr.plot(list(range(len(loss_list))), loss_list, '*r', markersize=24)
    arr.plot(list(range(len(loss_list))), loss_list, '-r', linewidth=4)
    fig.savefig(root_dir + loss_name + '.png', dpi=100)
    np.save(root_dir + loss_name + '.npy', np.array(loss_list))
    print("Loss data is saved.")
    # plt.show()


def save_metric(metric_list, metric_name):
    root_dir = "/mnt/sda/dataset/ml_nav/VAE/plot/"
    fig, arr = plt.subplots(1, 1, figsize=(8, 8))
    fig.canvas.set_window_title(metric_name)
    arr.set_title(" ".join(metric_name.split("_")))
    arr.set_xlabel("Batch Iteration")
    arr.set_ylabel("Classification Accuracy")
    arr.plot(list(range(len(metric_list))), metric_list, '*b', markersize=24)
    arr.plot(list(range(len(metric_list))), metric_list, '-b', linewidth=4)
    fig.savefig(root_dir + metric_name + '.png', dpi=100)
    np.save(root_dir + metric_name + '.npy', np.array(metric_list))
    print("Metric data is saved.")
    plt.show()


def save_model(model, model_name):
    root_dir = "/mnt/sda/dataset/ml_nav/VAE/model/"
    torch.save(model.state_dict(), root_dir + model_name + '.pt')
    print(model_name + ".pt is saved.")
