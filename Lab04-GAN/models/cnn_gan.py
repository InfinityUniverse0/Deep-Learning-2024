"""
Generative Adversarial Network (GAN) Model using CNN
"""

import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator Model using CNN
    """
    def __init__(self, input_shape=(1, 28, 28)):
        """
        Initialization

        :param input_shape: input shape (C, H, W)
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward Pass

        :param x: input tensor of shape (N, C, H, W)
        :return: output tensor of shape (N, 1)
        """
        return self.model(x).view(x.size(0), -1)


class Generator(nn.Module):
    """
    Generator Model using CNN
    """
    def __init__(self, latent_dim=100, output_shape=(1, 28, 28)):
        """
        Initialization

        :param latent_dim: latent dimension
        :param output_shape: output shape (C, H, W)
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.init_size = output_shape[1] // 4
        self.linear = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size * self.init_size))

        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, output_shape[0], kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward Pass

        :param x: input tensor of shape (N, latent_dim)
        :return: output tensor of shape (N, C, H, W)
        """
        out = self.linear(x)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        return self.model(out)

    def get_latent_dim(self):
        return self.latent_dim
