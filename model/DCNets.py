import torch
from torch import nn


# a simple classifier with 3 convolutional layers
class Classifier_Conv4(nn.Module):
    # init function
    def __init__(self):
        super(Classifier_Conv4, self).__init__()

        """
            Define a convolutional layer:
                nn.Conv2:
                nn.ReLU:
                nn.MaxPool2d:
            H_out = ([H_in + 2 x padding[0] - dilation[0] x (kernel_size[0] - 1)] / stride[0]) + 1
            W_out = ([W_in + 2 x padding[1] - dilation[1] x (kernel_size[1] - 1)] / stride[1]) + 1
        """
        # Conv layer 1:
        # Input: Batch x 3 x 64 x 64
        # Output: Batch x 32 x 32 x 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1,
                      dilation=1, groups=1, bias=True, padding_mode='replicate'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=False)
        )
        # Conv layer 2:
        # Input: Batch x 32 x 32 x 32
        # Output: Batch x 64 x 16 x 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,
                      padding_mode='replicate'),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        # Con layer 3:
        # Input: Batch x 64 x 16 x 16
        # Output: Batch x 128 x 8 x 8
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                      padding_mode='replicate'),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        # Con layer 4:
        # Input: Batch x 128 x 8 x 8
        # Output: Batch x 256 x 4 x 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1,
                      padding_mode='replicate'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        # FC layer :
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 256, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    # forward
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 4 * 4 * 256)
        x = self.fc(x)

        return x


# VAE encoder
class VAEEncoder(nn.Module):
    def __init__(self, z_dim):
        super(VAEEncoder, self).__init__()

        # convolutional layer
        self.conv_layer = nn.Sequential(
            # 3 x 64 x 64
            # 32 x 31 x 31
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=False),

            # 32 x 31 x 31
            # 64 x 14 x 14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=False),

            # 64 x 14 x 14
            # 128 x 6 x 6

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=False),

            # 128 x 6 x 6
            # 256 x 2 x 2
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=False)
        )

        # mean header
        self.mu_layer = nn.Linear(2 * 2 * 256, z_dim)

        # variance header
        self.log_var_layer = nn.Linear(2 * 2 * 256, z_dim)

    # forward function
    def forward(self, x):
        # visual embedding
        x = self.conv_layer(x)
        # flatten
        x = x.view(-1, 2 * 2 * 256)
        # compute mean and variance
        x_mu = self.mu_layer(x)
        x_log_var = self.log_var_layer(x)
        return x_mu, x_log_var


# VAE decoder
class VAEDecoder(nn.Module):
    def __init__(self, z_dim):
        super(VAEDecoder, self).__init__()
        # dense layer
        self.fc = nn.Linear(z_dim, 1024)
        # deconvolutional layer
        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=6, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.deconv_layer(x)
        return x, [x.view(-1, 3 * 64 * 64), torch.ones_like(x.view(-1, 3 * 64 * 64))]


# a simple VAE architecture
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        # define the encoder
        self.encoder = VAEEncoder(z_dim)
        # define the decoder
        self.decoder = VAEDecoder(z_dim)

    def reparameterize(self, mu, log_var):
        """
        Function is used to do the reparameterize trick in VAE
        :param mu: mu(X) from encoder
        :param log_var: log(var(X)) from encoder
        :return: z = mu(X) + exp{log(var(X)) * 0.5} * eps (sampled from a Gaussian)
        """
        eps = torch.randn_like(log_var)  # sampled a tensor with the same size as log_var from a Gaussian distribution
        z = mu + torch.exp(log_var * 0.5) * eps
        return z

    def forward(self, x):
        # encoder
        z_mu, z_log_var = self.encoder.forward(x)
        # reparameterize
        z = self.reparameterize(z_mu, z_log_var)
        # decoder
        x_reconstruction, x_distribution_params = self.decoder(z)
        return x_reconstruction, x_distribution_params, [z_mu, z_log_var]


