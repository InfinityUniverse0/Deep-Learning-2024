"""
MLP-Mixer Model
"""

import torch.nn as nn


class MLPMixer(nn.Module):
    """
    MLP-Mixer Model
    """
    def __init__(self, in_channels, patch_size, num_patches, hidden_dim, num_classes):
        """
        Initialization

        :param in_channels: number of input channels
        :param patch_size: patch size (int for square patch)
        :param num_patches: number of patches
        :param hidden_dim: hidden dimension
        :param num_classes: number of classes
        """
        super(MLPMixer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim

        self.patch_proj = nn.Linear(in_channels * patch_size * patch_size, hidden_dim)

        self.token_mixing = nn.Sequential(
            nn.Linear(num_patches, num_patches),
            nn.GELU(),
            nn.Linear(num_patches, num_patches),
            nn.GELU()
        )

        self.channel_mixing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward Pass

        :param x: input tensor of shape (N, C, H, W)
        :return: output tensor of shape (N, num_classes)
        """
        # Patch Embedding
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(x.size(0), x.size(1), -1, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), self.num_patches, -1)

        # Patch Projection
        x = self.patch_proj(x)

        # Token Mixing
        y = x.permute(0, 2, 1)
        y = self.token_mixing(y)
        y = y.permute(0, 2, 1)
        x = x + y  # Residual Connection

        # Channel Mixing
        y = self.channel_mixing(x)
        x = x + y  # Residual Connection

        # Global Average Pooling
        x = x.mean(dim=1)

        # Classifier
        x = self.classifier(x)
        return x
