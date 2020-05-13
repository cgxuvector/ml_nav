from dataset import Dataset
from dataset.Transform import ToTensor

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import random

import argparse
import matplotlib.pyplot as plt

from model import VAE
from skimage import io
from skimage.transform import resize


def eval_train(model, val_set_loader):
    total_num = len(val_set_loader)
    accurate_num = 0.0
    for idx, val_batch in enumerate(val_set_loader):
        # load a mini-batch
        x_data_val = val_batch["observations"].float()
        y_label_val = val_batch["label"].long()
        x_data_val = x_data_val.to(torch.device("cuda:0"))
        y_label_val = y_label_val.to(torch.device("cuda:0"))
        y_predict_val = model(x_data_val)
        y_predict_list = y_predict_val.tolist()[0]
        y_predict_label = y_predict_list.index(max(y_predict_list))
        # print(y_predict_list, y_predict_label, type(y_predict_label), type(y_label_val.item()))
        if y_predict_label == y_label_val.item():
            accurate_num += 1
    return accurate_num / total_num


def image_generation(dataLoader):
    # load the trained model
    cvae = VAE.CVAE(64)
    # cvae.load_state_dict(torch.load("/mnt/sda/dataset/ml_nav/model/cvae_model_h64_c34_L2_b1_ep100_id_1.pt"))
    # cvae.load_state_dict(torch.load("/mnt/sda/dataset/ml_nav/model/cvae_BN_variance_b2_1.pt"))
    cvae.load_state_dict(torch.load("/mnt/sda/dataset/ml_nav/model/cvae_global_map_warm_1.pt"))
    cvae.eval()

    # generate the name of orientations
    tmp_ori = transformed_dataset.orientation_angle
    orientations = [torch.from_numpy(tmp_ori[key]).float() for key in tmp_ori.keys()]

    # figure
    fig, arr = plt.subplots(1, 5, figsize=(10, 2))
    fig.canvas.set_window_title("VAE Reconstruction")
    h = []
    last_z = None
    for idx, batch in enumerate(dataLoader):
        obs = batch["observation"].squeeze(0).detach().numpy().transpose(1, 2, 0)
        ori = batch["orientation"].float()
        loc_map = batch["loc_map"].float()

        if idx == 0:
            for h_idx in range(5):
                if h_idx == 0:
                    h.append(arr[h_idx].imshow(obs))
                    arr[h_idx].set_title("GT")
                else:
                    z = torch.randn(1, 64)
                    # tmp_map = torch.cat(2 * [loc_map.view(-1, 1 * 3 * 3)], dim=1)
                    tmp_map = loc_map.view(-1, 1 * 21 * 21)
                    tmp_ori = torch.cat(2 * [ori.view(-1, 1 * 1 * 8)], dim=1)
                    conditioned_z = torch.cat((z, tmp_map, tmp_ori), dim=1)
                    obs_reconstructed, _ = cvae.decoder(conditioned_z)
                    h.append(arr[h_idx].imshow(obs_reconstructed.squeeze(0).detach().numpy().transpose(1, 2, 0)))
                    arr[h_idx].set_title(str(h_idx))
        else:
            for h_idx in range(5):
                if h_idx == 0:
                    h[h_idx].set_data(obs)
                else:
                    z = torch.randn(1, 64)
                    # tmp_map = torch.cat(2 * [loc_map.view(-1, 1 * 3 * 3)], dim=1)
                    tmp_map = loc_map.view(-1, 1 * 21 * 21)
                    tmp_ori = torch.cat(2 * [ori.view(-1, 1 * 1 * 8)], dim=1)
                    conditioned_z = torch.cat((z, tmp_map, tmp_ori), dim=1)
                    obs_reconstructed, _ = cvae.decoder(conditioned_z)
                    h[h_idx].set_data(obs_reconstructed.squeeze(0).detach().numpy().transpose(1, 2, 0))
        fig.canvas.draw()
        # plt.savefig("/mnt/sda/dataset/ml_nav/cvae_reconstruction/variance/{}_val.png".format(idx+1), dpi=50)
        plt.pause(2)


def generate_panoramic_observations(input_params, dataLoader, mode='compare'):
    # load the trained model
    cvae = VAE.CVAE(64, use_small_obs=True)
    cvae.load_state_dict(torch.load("./results/vae/model/small_obs_L64_B4.pt", map_location='cpu'))
    cvae.eval()

    # generate the name of orientations
    tmp_ori = transformed_dataset.orientation_angle
    orientations = [torch.from_numpy(tmp_ori[key]).float() for key in tmp_ori.keys()]

    # obtain corresponding dataLoder
    if input_params.data_type == "trn":
        dataLoader = dataLoader[0]
    elif input_args.data_type == "val":
        dataLoader = dataLoader[1]
    elif input_args.data_type == "tst":
        dataLoader = dataLoader[2]
    else:
        assert False, "Data Type Error: Please input valid data type. (trn, val or tst)."

    if mode == 'compare':
        # generate the panoramic observations
        for idx, batch in enumerate(dataLoader):
            # load the ground truth
            loc_map = batch["loc_map"].float()
            fig_gt = transformed_dataset.visualize_batch(batch, "group")

            # generate the imagined panorama view
            reconstructed_batch = {"observation": [], "loc_map": loc_map}
            # loc_map = torch.tensor([[0, 0, 0], [0, 1, 1], [0, 1, 0]]).unsqueeze(0).unsqueeze(0).float()
            for ori in orientations:
                # sample a latent variable
                z = torch.randn(1, 64)
                # create the map feature
                tmp_map = torch.cat(2 * [loc_map.view(-1, 1 * 3 * 3)], dim=1)
                # create the orientation feature
                tmp_ori = torch.cat(2 * [ori.view(-1, 1 * 1 * 8)], dim=1)
                # construct the conditioned latent vector
                conditioned_z = torch.cat((z, tmp_map, tmp_ori), dim=1)
                # decode the observation in one direction
                obs_reconstructed, _ = cvae.decoder(conditioned_z)
                obs_reconstructed = obs_reconstructed.detach()
                # save the reconstructed observation
                reconstructed_batch["observation"].append(obs_reconstructed)
            # show the imagined observations
            fig_recon = transformed_dataset.visualize_batch(reconstructed_batch, "group")
            plt.show()
            plt.cla
            if idx > 10:
                break
    elif mode == 'variance':
        for i in range(10):
            # load the ground truth
            loc_map = torch.tensor([[0, 0, 0], [0, 1, 1], [0, 1, 0]]).unsqueeze(0).unsqueeze(0).float()
            # generate the imagined panorama view
            reconstructed_batch = {"observation": [], "loc_map": loc_map}
            for ori in orientations:
                # sample a latent variable
                z = torch.randn(1, 64)
                # create the map feature
                tmp_map = torch.cat(2 * [loc_map.view(-1, 1 * 3 * 3)], dim=1)
                # create the orientation feature
                tmp_ori = torch.cat(2 * [ori.view(-1, 1 * 1 * 8)], dim=1)
                # construct the conditioned latent vector
                conditioned_z = torch.cat((z, tmp_map, tmp_ori), dim=1)
                # decode the observation in one direction
                obs_reconstructed, _ = cvae.decoder(conditioned_z)
                obs_reconstructed = obs_reconstructed.detach()
                # save the reconstructed observation
                reconstructed_batch["observation"].append(obs_reconstructed)
            # show the imagined observations
            fig_recon = transformed_dataset.visualize_batch(reconstructed_batch, "group")
            plt.show()
            plt.cla


def is_seen(loc_map, dataLoader):
    loc_map = loc_map.view(-1, 9)
    seen_flag = False
    for idx, batch in enumerate(dataLoader):
        map = torch.round(batch["loc_map"].float().squeeze(0).squeeze(0).view(-1, 9))
        if not torch.sum((loc_map - map)**2, dim=1):
            seen_flag = True
            print(map.view(3, 3))
            break
    return seen_flag


def generate_panoramic_observations_test(dataLoader):
    # load the trained model
    cvae = VAE.CVAE(64)
    cvae.load_state_dict(torch.load("/mnt/sda/dataset/ml_nav/model/cvae_BN_variance_b2_1.pt"))
    cvae.eval()

    # generate the name of orientations
    tmp_ori = transformed_dataset.orientation_angle
    orientations = [torch.from_numpy(tmp_ori[key]).float() for key in tmp_ori.keys()]

    # generation
    loc_map = torch.tensor([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]]).float()
    if not is_seen(loc_map, dataLoader):
        reconstructed_batch = {"observation": [], "loc_map": loc_map}
        for ori in orientations:
            z = torch.randn(1, 64)
            tmp_map = torch.cat(2 * [loc_map.view(-1, 1 * 3 * 3)], dim=1)
            tmp_ori = torch.cat(2 * [ori.view(-1, 1 * 1 * 8)], dim=1)
            conditioned_z = torch.cat((z, tmp_map, tmp_ori), dim=1)
            obs_reconstructed, _ = cvae.decoder(conditioned_z)
            obs_reconstructed = obs_reconstructed.detach()
            reconstructed_batch["observation"].append(obs_reconstructed)
        fig_recon = transformed_dataset.visualize_batch(reconstructed_batch, "group")
        plt.show()
    else:
        assert False, "The map is seen in the training."


def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_num", type=int, default=4, help="number of the worker for data loader")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_type", type=str, default="trn", help="type of the evaluation data. (trn, val or tst)")

    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)
    # load parameters
    input_args = input_parser()

    # load the dataset
    transformed_dataset = Dataset.LocmapObsDataset(mode="group",
                                                   dir_path='/Users/chengguang/PycharmProjects/vae_images/loc_map_obs_fixed_texture_small',
                                                   transform=transforms.Compose([ToTensor("group")]))

    # split the dataset into training, validation and testing
    trn_ratio = 0.7
    val_ratio = 0.15
    tst_ratio = 0.15
    seed = 1234
    dataset_size = len(transformed_dataset)
    trn_sampler, val_sampler, tst_sampler = transformed_dataset.split(dataset_size,
                                                                      trn_ratio,
                                                                      val_ratio,
                                                                      seed, shuffle=True)
    # construct training, validation, and testing sets
    dataLoader_trn = DataLoader(transformed_dataset, batch_size=1, sampler=trn_sampler, num_workers=input_args.worker_num)
    dataLoader_val = DataLoader(transformed_dataset, batch_size=1, sampler=val_sampler, num_workers=input_args.worker_num)
    dataLoader_tst = DataLoader(transformed_dataset, batch_size=1, sampler=tst_sampler, num_workers=input_args.worker_num)

    # trainer
    generate_panoramic_observations(input_args, [dataLoader_trn, dataLoader_val, dataLoader_tst], 'variance')