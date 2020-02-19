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


def image_generation():
    # load the trained model
    vae = DCNets.VAE(64)
    vae.load_state_dict(torch.load("/mnt/sda/dataset/ml_nav/model/vae_vanilla_various_texture.pt"))
    vae.eval()

    # sample one tensor
    count = 10
    while count > 0:
        z = torch.randn(1, 64)
        img, _ = vae.decoder(z)
        img = img.squeeze(0).detach().numpy().transpose(1, 2, 0)
        plt.imshow(img)
        plt.show()
        count -= 1


if __name__ == '__main__':
    # load the dataset
    transformed_dataset = Dataset.LocmapObsDataset(mode="cls-iid",
                                                   dir_path='/mnt/sda/dataset/ml_nav/loc_map_obs_various_texture',
                                                   transform=transforms.Compose([ToTensor("cls-iid")]))

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
    dataLoader_trn = DataLoader(transformed_dataset, batch_size=8, sampler=trn_sampler, num_workers=8)
    dataLoader_val = DataLoader(transformed_dataset, batch_size=1, sampler=val_sampler, num_workers=4)
    dataLoader_tst = DataLoader(transformed_dataset, batch_size=1, sampler=tst_sampler)

    # # define a VAE trainer
    # myTrainer = DCNTrainer.VAETrainer(64, [dataLoader_trn, dataLoader_val, dataLoader_tst], 10)
    # myTrainer.train()
    # save_data.save_loss(myTrainer.trn_loss_list, "VAE_Training_Loss_various_texture")
    # save_data.save_model(myTrainer.model, "vae_vanilla_various_texture")

    image_generation()


    # # create the classifier
    # # myClassifier = DCNets.Classifier_Conv4()
    # myClassifier = DCNets.VAE(64)
    # myClassifier.to(torch.device("cuda:0"))
    # # create the loss function
    # criterion = torch.nn.CrossEntropyLoss()
    # # create optimizer
    # optimizer = torch.optim.Adam(myClassifier.parameters(),
    #                              lr=1e-4,
    #                              weight_decay=5e-4)
    # trn_loss_list = []
    # val_acc_list = []
    # for epoch in range(1):
    #     running_loss = 0.0
    #     for idx, batch in enumerate(dataLoader_trn):
    #         # fig = transformed_dataset.show(batch)
    #         # plt.show()
    #         # load a mini-batch
    #         # batch = batch.to(torch.device("cuda:0"))
    #         x_data = batch["observations"].to(torch.device("cuda:0")).float()
    #         y_label = batch["label"].to(torch.device("cuda:0")).long()
    #
    #         # feed forward
    #         y_predict = myClassifier(x_data)
    #         break
            #
            # # compute loss
            # loss = criterion(y_predict, y_label)
            #
            # running_loss += loss.item()
            #
            # if idx == 0:
            #     with torch.no_grad():
            #         val_acc_list.append(eval_train(myClassifier, dataLoader_tst))
            #     trn_loss_list.append(running_loss)
            #     print("Batch Iter = {} : Loss = {} ; Val Acc = {}".format(idx, running_loss, val_acc_list[-1]))
            #     # print("Batch Iter = {} : Loss = {} ".format(idx, running_loss / 20))
            #
            #     running_loss = 0.0
            #
            # if idx % 20 == 19:
            #     with torch.no_grad():
            #         val_acc_list.append(eval_train(myClassifier, dataLoader_val))
            #     trn_loss_list.append(running_loss / 20)
            #     print("Batch Iter = {} : Loss = {} ; Val Acc = {}".format(idx, running_loss / 20, val_acc_list[-1]))
            #     # print("Batch Iter = {} : Loss = {} ".format(idx, running_loss / 20))
            #
            #     running_loss = 0.0
            #
            # # back propagation
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()





