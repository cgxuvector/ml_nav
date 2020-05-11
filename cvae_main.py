from dataset import Dataset
from dataset.Transform import ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import save_data
from model import DCNTrainer
import argparse


def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8, help="Size of the mini-batch")
    parser.add_argument("--epoch", type=int, default=10, help="Train using the overall dataset")
    parser.add_argument("--worker_num", type=int, default=4, help="number of the worker for data loader")
    parser.add_argument("--plot_save_name", type=str)
    parser.add_argument("--model_save_name", type=str)
    parser.add_argument("--use_small_obs", type=bool, default=False)
    parser.add_argument("--warm_up", type=bool, default=False, help="If True, warm up is applied.")
    parser.add_argument("--device", type=str, default='cpu')

    return parser.parse_args()


if __name__ == '__main__':
    input_args = input_parser()
    # load the dataset
    transformed_dataset = Dataset.LocmapObsDataset(mode="conditional-iid",
                                                   dir_path='/mnt/sda/dataset/ml_nav/loc_map_obs_fixed_texture_small',
                                                   transform=transforms.Compose([ToTensor("conditional-iid")]))

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
    dataLoader_trn = DataLoader(transformed_dataset, batch_size=input_args.batch_size, sampler=trn_sampler, num_workers=input_args.worker_num)
    dataLoader_val = DataLoader(transformed_dataset, batch_size=1, sampler=val_sampler, num_workers=4)
    dataLoader_tst = DataLoader(transformed_dataset, batch_size=1, sampler=tst_sampler)

    # construct a conditional VAE
    myTrainer = DCNTrainer.CVAETrainer(input_args.hidden_size, [dataLoader_trn, dataLoader_val, dataLoader_tst],
                                       input_args.epoch,
                                       warm_up=input_args.warm_up,
                                       learning_rate=1e-3,
                                       use_small_obs=input_args.use_small_obs,
                                       device=input_args.device)
    myTrainer.train()

    # save the results
    save_data.save_loss(myTrainer.trn_loss_list, input_args.plot_save_name)
    save_data.save_loss(myTrainer.trn_recon_list, input_args.plot_save_name + "_recon_loss")
    save_data.save_loss(myTrainer.trn_kl_list, input_args.plot_save_name + "_kl_loss")
    save_data.save_model(myTrainer.model, input_args.model_save_name)

