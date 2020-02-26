"""
    VAE Model
        - Architecture:
                1. encoder:
"""
import torch
import torch.nn as nn


# Convolutional Encoder
class CNNEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(CNNEncoder, self).__init__()

        # cnn layer to encode the observations
        self.cnn_layer = nn.Sequential(
            # 3 x 64 x 64 --> 32 x 31 x 31
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32 x 31 x 31 --> 64 x 14 x 14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 x 14 x 14 --> 128 x 6 x 6
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 128 x 6 x 6 --> 256 x 2 x 2
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 256 x 2 x 2 --> 512 x 1 x 1
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # mean linear layer
        self.mu_layer = nn.Sequential(
            nn.Linear(512 + 34, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        # logarithm linear layer
        self.log_var_layer = nn.Sequential(
            nn.Linear(512 + 34, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

    def forward(self, x_obs, y_map, z_ori):
        # compute the visual embedding
        x_obs_feature = self.cnn_layer(x_obs)
        # flatten the features
        x_obs_flat = x_obs_feature.view(-1, 1 * 1 * 512)
        y_map_flat = torch.cat(2 * [y_map.view(-1, 1 * 3 * 3)], dim=1)
        z_ori_flat = torch.cat(2 * [z_ori.view(-1, 1 * 8)], dim=1)
        # compute the conditional feature
        conditional_feature = torch.cat((x_obs_flat, y_map_flat, z_ori_flat), dim=1)
        # compute the mean and variance
        latent_mu = self.mu_layer(conditional_feature)
        latent_log_var = self.log_var_layer(conditional_feature)

        return latent_mu, latent_log_var, y_map_flat, z_ori_flat


# Convolutional decoder
class CNNDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(CNNDecoder, self).__init__()
        # dense layer
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 34, 1024),
            nn.BatchNorm1d(1024),
        )
        # deconvolutional layer
        self.de_cnn_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # resize the input
        x = self.fc(x)
        # reshape from 2d 1024 x 1 to 3d 1024 x 1 x 1
        x = x.unsqueeze(2).unsqueeze(3)
        # deconvolution
        x = self.de_cnn_layer(x)
        return x, [x.view(-1, 3 * 64 * 64), torch.ones_like(x.view(-1, 3 * 64 * 64))]


class CVAE(nn.Module):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()

        self.encoder = CNNEncoder(latent_dim)
        self.decoder = CNNDecoder(latent_dim)

    def reparameterized(self, mu, log_var):
        eps = torch.randn_like(log_var)
        z = mu + torch.exp((0.5 * log_var)) * eps
        return z

    def forward(self, x_obs, y_map, z_ori):
        # visual embedding
        latent_mu, latent_log_var, latent_map, latent_ori = self.encoder(x_obs, y_map, z_ori)
        # reparameterized latent representation
        latent_z = self.reparameterized(latent_mu, latent_log_var)
        # compute the conditional latent z
        conditioned_latent_z = torch.cat((latent_z, latent_map, latent_ori), dim=1)
        # reconstruct the observation
        reconstructed_obs, reconstructed_obs_distribution_params = self.decoder(conditioned_latent_z)

        return reconstructed_obs, reconstructed_obs_distribution_params, [latent_mu, latent_log_var]

