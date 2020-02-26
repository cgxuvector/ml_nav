"""
    Script for training different neural network
"""
import torch
from model import VAE
from model import DCNets
from model import LossFunction
from tqdm import tqdm
from utils import ml_schedule


class CNNClassifierTrainer(object):
    def __init__(self, data_loaders, epoch, device=torch.device("cuda:0"), learning_rate=1e-3, weight_decay=5e-4):
        self.dataLoader_trn = data_loaders[0]
        self.dataLoader_val = data_loaders[1]
        self.dataLoader_tst = data_loaders[2]
        self.epoch = epoch
        self.device = device
        # model, optimizer, and loss
        self.model = DCNets.Classifier_Conv4().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        # training data
        self.trn_loss_list = []
        self.val_acc_list = []

    def train(self):
        for epoch in range(self.epoch):
            running_loss = 0.0
            for idx, batch in enumerate(self.dataLoader_trn):
                # load a mini-batch
                x_data = batch["observation"].to(torch.device("cuda:0")).float()
                y_label = batch["label"].to(torch.device("cuda:0")).long()

                # feed forward
                y_predict = self.model(x_data)

                # compute loss
                loss = self.criterion(y_predict, y_label)

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if idx % 20 == 19:
                    self.trn_loss_list.append(running_loss / 20)
                    print("Batch Iter = {} : Loss = {} ".format(idx, running_loss / 20))

                    running_loss = 0.0


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
        for ep in tqdm(range(self.epoch)):
            running_loss = 0.0
            for idx, batch in enumerate(self.dataLoader_trn):
                x_data = batch["observation"].to(self.device).float()
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


class CVAETrainer(object):
    def __init__(self, latent_dim, data_loaders, epoch, device=torch.device("cuda:0"), learning_rate=1e-3, weight_decay=5e-4, warm_up=False):
        self.dataLoader_trn = data_loaders[0]
        self.dataLoader_val = data_loaders[1]
        self.dataLoader_tst = data_loaders[2]
        self.device = device
        self.epoch = epoch
        self.wp = warm_up
        # define model
        self.model = VAE.CVAE(latent_dim).to(device)

        # training the model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = LossFunction.VAELoss()

        # training statistic data
        self.trn_loss_list = []
        self.trn_recon_list = []
        self.trn_kl_list = []

    def train(self):
        beta_schedule = ml_schedule.LinearSchedule(0, 1, self.epoch)
        # for ep in tqdm(range(self.epoch)):
        for ep in range(self.epoch):
            running_loss = 0.0
            running_recon_loss = 0.0
            running_kl_loss = 0.0
            for idx, batch in enumerate(self.dataLoader_trn):
                x_obs = batch["observation"].to(self.device).float()
                y_map = batch["loc_map"].to(self.device).float()
                z_ori = batch["orientation"].to(self.device).float()

                # forward
                x_reconstruct, x_distribution_params, z_distribution_params = self.model(
                    x_obs,
                    y_map,
                    z_ori
                )

                # compute loss
                beta = beta_schedule.get_value(ep) if self.wp else None
                loss, recon_loss, kl_loss = self.criterion(x_obs, x_reconstruct, x_distribution_params, z_distribution_params, ep)

                running_loss += loss.item()
                running_recon_loss += recon_loss
                running_kl_loss += kl_loss
                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if idx % 20 == 19:
                    run_loss = running_loss / 20
                    run_recon = running_recon_loss / 20
                    run_kl = running_kl_loss / 20
                    print("Epoch = {} Batch Iter = {} : ELBO = {}, Recon Loss = {}, KL = {}, Beta = {}".format(ep,
                                                                                                               idx,
                                                                                                               run_loss,
                                                                                                               run_recon,
                                                                                                               run_kl,
                                                                                                               beta))
                    self.trn_loss_list.append(run_loss)
                    self.trn_recon_list.append(run_recon)
                    self.trn_kl_list.append(run_kl)
                    running_loss = 0.0
                    running_recon_loss = 0.0
                    running_kl_loss = 0.0



