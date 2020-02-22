# """
#     This script contains the definition of several convolutional neural networks
#         - Convolutional Neural Classifier
#         - Variational Auto-Encoder
#
#     Note:
#         Define a convolutional layer:
#                 nn.Conv2: in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, padding_model="zeros"
#                 nn.MaxPool2d:
#                 nn.ReLU:
#         H_out = ([H_in + 2 x padding[0] - dilation[0] x (kernel_size[0] - 1) - 1] / stride[0]) + 1
#         W_out = ([W_in + 2 x padding[1] - dilation[1] x (kernel_size[1] - 1) - 1] / stride[1]) + 1
#
#         Define a de-convolutional layer:
#                 nn.ConvTranspose2d:in_channels, out_channels, kernel_size, padding=0, stride=1
#                 nn.MaxPool2d:
#                 nn.ReLU:
#         H_out = (H_in - 1) x stride[0] - 2 x padding[0] + dilation[0] x (kernel_size[0] - 1) + out_put_padding[0] + 1
#         W_out = (W_in - 1) x stride[1] - 2 x padding[1] + dilation[1] x (kernel_size[1] - 1) + out_put_padding[1] + 1
# """
# import torch
# from torch import nn
#
#
# """
#     Convolutional Neural Network Classifier
# """
#
# # a simple classifier with 3 convolutional layers
# class Classifier_Conv4(nn.Module):
#     """
#         This model is a convolutional neural classifier.
#
#     """
#     def __init__(self):
#         super(Classifier_Conv4, self).__init__()
#
#         self.cnn = nn.Sequential(
#             # Input: Batch x 3 x 64 x 64
#             # Output: Batch x 32 x 32 x 32
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1,
#                       dilation=1, groups=1, bias=True, padding_mode='replicate'),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=False),
#
#             # Input: Batch x 32 x 32 x 32
#             # Output: Batch x 64 x 16 x 16
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,
#                       padding_mode='replicate'),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(),
#
#             # Input: Batch x 64 x 16 x 16
#             # Output: Batch x 128 x 8 x 8
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
#                       padding_mode='replicate'),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(),
#
#             # Input: Batch x 128 x 8 x 8
#             # Output: Batch x 256 x 4 x 4
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1,
#                       padding_mode='replicate'),
#             nn.MaxPool2d(2, 2),
#             nn.ReLU(),
#         )
#         # FC layer :
#         self.fc = nn.Sequential(
#             nn.Linear(4 * 4 * 256, 1000),
#             nn.ReLU(),
#             nn.Linear(1000, 2),
#             nn.Softmax(dim=1)
#         )
#
#     # forward
#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(-1, 4 * 4 * 256)
#         x = self.fc(x)
#
#         return x
#
#
# """
#     Variational Auto-Encoder
# """
#
#
# # VAE encoder
# class VAEEncoder(nn.Module):
#     """ This is the convolutional encoder of VAE """
#     def __init__(self, z_dim):
#         super(VAEEncoder, self).__init__()
#
#         # convolutional layer
#         self.conv_layer = nn.Sequential(
#             # 3 x 64 x 64
#             # 32 x 31 x 31
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0, stride=1),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=False),
#
#             # 32 x 31 x 31
#             # 64 x 14 x 14
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=0, stride=1),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=False),
#
#             # 64 x 14 x 14
#             # 128 x 6 x 6
#
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=1),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=False),
#
#             # 128 x 6 x 6
#             # 256 x 2 x 2
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0, stride=1),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=False)
#         )
#
#         # mean header
#         self.mu_layer = nn.Linear(2 * 2 * 256, z_dim)
#
#         # variance header
#         self.log_var_layer = nn.Linear(2 * 2 * 256, z_dim)
#
#     # forward function
#     def forward(self, x):
#         # cnn embedding
#         x = self.conv_layer(x)
#         # flatten the tensor
#         x = x.view(-1, 2 * 2 * 256)
#         # compute mean and variance
#         x_mu = self.mu_layer(x)
#         x_log_var = self.log_var_layer(x)
#         return x_mu, x_log_var
#
#
# # VAE decoder
# class VAEDecoder(nn.Module):
#     """ This is the deconvolutional decoder of VAE """
#     def __init__(self, z_dim):
#         super(VAEDecoder, self).__init__()
#         # dense layer
#         self.fc = nn.Linear(z_dim, 1024)
#         # deconvolutional layer
#         self.deconv_layer = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # resize the input
#         x = self.fc(x)
#         # reshape from 2d 1024 x 1 to 3d 1024 x 1 x 1
#         x = x.unsqueeze(2).unsqueeze(3)
#         # deconvolution
#         x = self.deconv_layer(x)
#         return x, [x.view(-1, 3 * 64 * 64), torch.ones_like(x.view(-1, 3 * 64 * 64))]
#
#
# # a simple VAE architecture
# class VAE(nn.Module):
#     def __init__(self, z_dim):
#         super(VAE, self).__init__()
#
#         # define the encoder
#         self.encoder = VAEEncoder(z_dim)
#         # define the decoder
#         self.decoder = VAEDecoder(z_dim)
#
#     def reparameterize(self, mu, log_var):
#         """
#         Function is used to do the reparameterize trick in VAE
#         :param mu: mu(X) from encoder
#         :param log_var: log(var(X)) from encoder
#         :return: z = mu(X) + exp{log(var(X)) * 0.5} * eps (sampled from a Gaussian)
#         """
#         eps = torch.randn_like(log_var)  # sampled a tensor with the same size as log_var from a Gaussian distribution
#         z = mu + torch.exp(log_var * 0.5) * eps
#         return z
#
#     def forward(self, x):
#         # encoder
#         z_mu, z_log_var = self.encoder.forward(x)
#         # reparameterize
#         z = self.reparameterize(z_mu, z_log_var)
#         # decoder
#         x_reconstruction, x_distribution_params = self.decoder(z)
#         return x_reconstruction, x_distribution_params, [z_mu, z_log_var]
#
#
# """
#     Conditional Auto-Encoder
# """
# class CVAESiameseEncoder(nn.Module):
#     def __init__(self, z_dim):
#         super(CVAESiameseEncoder, self).__init__()
#
#         self.obs_conv_layer = nn.Sequential(
#             # 3 x 64 x 64 --> 32 x 31 x 31
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.BatchNorm2d(32),
#             nn.ReLU(),
#
#             # 32 x 31 x 31 --> 64 x 14 x 14
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.BatchNorm2d(64),
#             nn.ReLU(),
#
#             # 64 x 14 x 14 --> 128 x 6 x 6
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.BatchNorm2d(128),
#             nn.ReLU(),
#
#             # 128 x 6 x 6 --> 256 x 2 x 2
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.BatchNorm2d(256),
#             nn.ReLU(),
#
#             # 256 x 2 x 2 --> 512 x 1 x 1
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2),
#             #nn.BatchNorm2d(512),
#             nn.ReLU(),
#         )
#
#         # self.map_conv_layer = nn.Sequential(
#         #     # 1 x 3 x 3 --> 32 x 3 x 3
#         #     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1),
#         #     nn.BatchNorm2d(32),
#         #     nn.ReLU(),
#         #
#         #     # 32 x 3 x 3 --> 64 x 3 x 3
#         #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         #
#         #     # 64 x 3 x 3 --> 128 x 3 x 3
#         #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
#         #     nn.BatchNorm2d(128),
#         #     nn.ReLU(),
#         #
#         #     # 128 x 6 x 6 --> 256 x 2 x 2
#         #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(),
#         #
#         #     # 256 x 2 x 2 --> 512 x 1 x 1
#         #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU()
#         # )
#         self.map_conv_layer = nn.Sequential(
#             # 1 x 64 x 64 -> 32 x 31 x 31
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.BatchNorm2d(32),
#             nn.ReLU(),
#
#             # 32 x 31 x 31 --> 64 x 14 x 14
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.BatchNorm2d(64),
#             nn.ReLU(),
#
#             # 64 x 14 x 14 --> 128 x 6 x 6
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.BatchNorm2d(128),
#             nn.ReLU(),
#
#             # 128 x 6 x 6 --> 256 x 2 x 2
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             #nn.BatchNorm2d(256),
#             nn.ReLU(),
#
#             # 256 x 2 x 2 --> 512 x 1 x 1
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2),
#             #nn.BatchNorm2d(512),
#             nn.ReLU(),
#         )
#
#         # conditioned feature: obs + map + orientation: 512 + 512 + 8 -> 64
#         self.mu_layer = nn.Linear(512 * 3, z_dim)
#         self.log_var_layer = nn.Linear(512 * 3, z_dim)
#
#         # map feature encoding: 512 -> 64
#         self.map_latent_layer = nn.Linear(512, z_dim)
#         self.ori_latent_layer = nn.Linear(512, z_dim)
#
#     def forward(self, x_obs, y_map, z_ori):
#         # compute the mu and log variance
#         x_obs = self.obs_conv_layer(x_obs)
#         y_map = self.map_conv_layer(y_map)
#         x_obs = x_obs.view(-1, 1 * 1 * 512)
#         y_map = y_map.view(-1, 1 * 1 * 512)
#         z_ori = torch.cat(64 * [z_ori.view(-1, 8)], dim=1)
#         conditioned_rep = torch.cat((x_obs, y_map, z_ori), dim=1)
#         z_mu = self.mu_layer(conditioned_rep)
#         z_log_var = self.log_var_layer(conditioned_rep)
#         z_y_map = self.map_latent_layer(y_map)
#         z_ori = self.ori_latent_layer(z_ori)
#
#         return z_mu, z_log_var, z_y_map, z_ori
#
#
# class CVAEEncoder(nn.Module):
#     def __init__(self, z_dim):
#         super(CVAEEncoder, self).__init__()
#
#         self.obs_conv_layer = nn.Sequential(
#             # 3 x 64 x 64 --> 32 x 31 x 31
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(),
#
#             # 32 x 31 x 31 --> 64 x 14 x 14
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(),
#
#
#             # 64 x 14 x 14 --> 128 x 6 x 6
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(),
#
#             # 128 x 6 x 6 --> 256 x 2 x 2
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(),
#
#             # 256 x 2 x 2 --> 512 x 1 x 1
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2),
#             nn.ReLU(),
#         )
#
#         self.mu_layer = nn.Linear(512 + 34, z_dim)
#         self.log_var_layer = nn.Linear(512 + 34, z_dim)
#
#     def forward(self, x_obs, y_map, z_ori):
#         # compute the mu and log variance
#         x_obs = self.obs_conv_layer(x_obs)
#         x_obs = x_obs.view(-1, 1 * 1 * 512)
#         y_map = torch.cat(2 * [y_map.view(-1, 1 * 3 * 3)], dim=1)
#         z_ori = torch.cat(2 * [z_ori.view(-1, 8)], dim=1)
#         conditioned_rep = torch.cat((x_obs, y_map, z_ori), dim=1)
#         z_mu = self.mu_layer(conditioned_rep)
#         z_log_var = self.log_var_layer(conditioned_rep)
#
#         return z_mu, z_log_var, y_map, z_ori
#
#
# # VAE decoder
# class CVAEDecoder(nn.Module):
#     """ This is the deconvolutional decoder of conditional VAE """
#     def __init__(self, z_dim):
#         super(CVAEDecoder, self).__init__()
#         # dense layer
#         self.fc = nn.Linear(z_dim + 34, 1024)
#         # deconvolutional layer
#         self.deconv_layer = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=1),
#             #nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
#             #nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
#             #nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
#             #nn.BatchNorm2d(3),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # resize the input
#         x = self.fc(x)
#         # reshape from 2d 1024 x 1 to 3d 1024 x 1 x 1
#         x = x.unsqueeze(2).unsqueeze(3)
#         # deconvolution
#         x = self.deconv_layer(x)
#         return x, [x.view(-1, 3 * 64 * 64), torch.ones_like(x.view(-1, 3 * 64 * 64))]
#
#
# class CVAE(nn.Module):
#     def __init__(self, z_dim):
#         super(CVAE, self).__init__()
#
#         self.encoder = CVAEEncoder(z_dim)
#         self.decoder = CVAEDecoder(z_dim)
#
#     def reparameterized(self, mu, log_var):
#         eps = torch.randn_like(log_var)
#         z = mu + torch.exp((0.5 * log_var)) * eps
#         return z
#
#     def forward(self, x_obs, y_map, z_ori):
#         # visual embedding
#         z_mu, z_log_var, z_y_map, z_ori = self.encoder(x_obs, y_map, z_ori)
#         # reparameterized latent representation
#         z = self.reparameterized(z_mu, z_log_var)
#         # addition conditioned vector
#         conditioned_z = torch.cat((z, z_y_map, z_ori), dim=1)
#         x_reconstruction, x_distribution_params = self.decoder(conditioned_z)
#         return x_reconstruction, x_distribution_params, [z_mu, z_log_var]
#
