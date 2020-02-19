"""
    Script for training different neural network
"""
import torch
from model import DCNets
from model import LossFunction


class VAETrainer(object):
    def __init__(self, latent_dim, data_loaders, epoch, device=torch.device("cuda:0"), learning_rate=1e-3, weight_decay=5e-4):
        self.dataLoader_trn = data_loaders[0]
        self.dataLoader_val = data_loaders[1]
        self.dataLoader_tst = data_loaders[2]
        self.epoch = epoch
        self.device = device
        # model, optimizer, and loss
        self.model = DCNets.VAE(latent_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = LossFunction.VAELoss()
        # training data
        self.trn_loss_list = []

    def train(self):
        for ep in range(self.epoch):
            running_loss = 0.0
            for idx, batch in enumerate(self.dataLoader_trn):
                x_data = batch["observations"].to(self.device).float()
                # forward
                x_reconstruct, x_distribution_params, z_distribution_params = self.model(x_data)

                # compute loss
                loss = self.loss_fn(x_data, x_reconstruct, x_distribution_params, z_distribution_params)

                running_loss += loss.item()
                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if idx % 20 == 19:
                    print("Batch Iter = {} : Loss = {}".format(idx, running_loss / 20))
                    self.trn_loss_list.append(running_loss / 20)
                    running_loss = 0.0



