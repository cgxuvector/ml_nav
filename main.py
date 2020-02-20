from model import DCNets
from dataset import Dataset
from dataset.Transform import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import save_data
from model import DCNTrainer
from model import DCNets
from skimage import io
import argparse


def split_train_val_tst(data_num, trn_ratio, val_ratio, tst_ratio, seed, shuffle=False):
    # obtain all the indices
    indices = list(range(data_num))
    # compute the number of samples for training, validation, and testing
    trn_num = int(np.round(data_num * trn_ratio))
    val_num = int(np.round(data_num * val_ratio))
    tst_num = data_num - trn_num - val_num
    # shuffle the indices
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    # obtain the indices for training, validation, and testing
    trn_indices = indices[:trn_num]
    val_indices = indices[trn_num:trn_num + val_num]
    tst_indices = indices[data_num - tst_num:]

    trn_sample = SubsetRandomSampler(trn_indices)
    val_sample = SubsetRandomSampler(val_indices)
    tst_sample = SubsetRandomSampler(tst_indices)

    return trn_sample, val_sample, tst_sample


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
    cvae = DCNets.CVAE(64)
    cvae.load_state_dict(torch.load("/mnt/sda/dataset/ml_nav/model/conditional_vae_vanilla_fixed_texture.pt"))
    cvae.eval()

    tmp_map = io.imread("/mnt/sda/dataset/ml_nav/loc_map_obs_fixed_texture/map_9_14_7_5.png")
    tmp_img = io.imread("/mnt/sda/dataset/ml_nav/loc_map_obs_fixed_texture/obs_9_14_7_5_RGB.LOOK_NORTH_WEST.png")



    fig, arr = plt.subplots(1, 2)
    arr[0].set_title("Ground Truth", fontsize=12)
    h1 = arr[0].imshow(tmp_img)
    arr[1].set_title("Reconstructed", fontsize=12)
    h2 = arr[1].imshow(tmp_img)
    fig.canvas.set_window_title("VAE Reconstruction")
    # plt.show()
    for idx, batch in enumerate(dataLoader):
        obs = batch["observation"].squeeze(0).detach().numpy().transpose(1, 2, 0)
        ori = batch["orientation"].float()
        loc_map = batch["loc_map"].float()

        count = 100
        while count > 0:
            z = torch.randn(1, 64)
            conditioned_z = torch.cat((z, loc_map.view(-1, 9), ori.view(-1, 8)), dim=1)
            obs_reconstructed, _ = cvae.decoder(conditioned_z)
            obs_reconstructed = obs_reconstructed.squeeze(0).detach().numpy().transpose(1, 2, 0)

            error = np.power((obs_reconstructed - obs), 2).mean()
            print("Error = ", error)

            if error < 0.01:
                break

            count -= 1

        h1.set_data(obs)
        h2.set_data(obs_reconstructed)
        fig.canvas.draw()
        plt.pause(2)


def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8, help="Size of the mini-batch")
    parser.add_argument("--epoch", type=int, default=10, help="Train using the overall dataset")
    parser.add_argument("--worker_num", type=int, default=4, help="number of the worker for data loader")
    parser.add_argument("--plot_save_name", type=str)
    parser.add_argument("--model_save_name", type=str)

    return parser.parse_args()


if __name__ == '__main__':

    input_args = input_parser()
    # load the dataset
    transformed_dataset = Dataset.LocmapObsDataset(mode="cvae",
                                                   dir_path='/mnt/sda/dataset/ml_nav/loc_map_obs_fixed_texture',
                                                   transform=transforms.Compose([ToTensor("cvae")]))

    # split the dataset into training, validation and testing
    trn_ratio = 0.7
    val_ratio = 0.15
    tst_ratio = 0.15
    seed = 1234
    dataset_size = len(transformed_dataset)
    trn_sampler, val_sampler, tst_sampler = split_train_val_tst(dataset_size,
                                                                trn_ratio,
                                                                val_ratio,
                                                                trn_ratio,
                                                                seed, shuffle=True)
    # construct training, validation, and testing sets
    dataLoader_trn = DataLoader(transformed_dataset, batch_size=input_args.batch_size, sampler=trn_sampler, num_workers=2)
    dataLoader_val = DataLoader(transformed_dataset, batch_size=1, sampler=val_sampler, num_workers=4)
    dataLoader_tst = DataLoader(transformed_dataset, batch_size=1, sampler=tst_sampler)


    # define a classification trainer
    myTrainer = DCNTrainer.CVAETrainer(input_args.hidden_size, [dataLoader_trn, dataLoader_val, dataLoader_tst], input_args.epoch)
    myTrainer.train()
    # # # # define a VAE trainer
    # # # # myTrainer = DCNTrainer.VAETrainer(64, [dataLoader_trn, dataLoader_val, dataLoader_tst], 10)
    # # # # myTrainer.train()
    save_data.save_loss(myTrainer.trn_loss_list, input_args.plot_save_name)
    save_data.save_model(myTrainer.model, input_args.model_save_name)

    # image_generation(dataLoader_tst)


