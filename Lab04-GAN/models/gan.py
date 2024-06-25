"""
Generative Adversarial Network (GAN) Model using MLP
"""

import torch.nn as nn
from math import prod


class Discriminator(nn.Module):
    """
    Discriminator Model
    """
    def __init__(self, input_shape=(1, 28, 28)):
        """
        Initialization

        :param input_shape: input shape (C, H, W)
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(input_shape), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward Pass

        :param x: input tensor of shape (N, C, H, W)
        :return: output tensor of shape (N, 1)
        """
        return self.model(x)


class Generator(nn.Module):
    """
    Generator Model
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
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256, 0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, 0.8),
            nn.Linear(512, prod(output_shape)),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward Pass

        :param x: input tensor of shape (N, latent_dim)
        :return: output tensor of shape (N, C, H, W)
        """
        out = self.model(x)
        return out.view(out.size(0), *self.output_shape)

    def get_latent_dim(self):
        return self.latent_dim
