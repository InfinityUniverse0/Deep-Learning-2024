"""
MLP Model
"""

import torch.nn as nn
from math import prod


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self, input_shape=(28, 28), num_classes=10):
        """
        Initialization

        :param input_shape: input shape (C, H, W) or (H, W)
        :param num_classes: number of classes
        """
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(prod(input_shape), 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward Pass

        :param x: input tensor of shape (N, C, H, W) or (N, H, W)
        :return:  output tensor of shape (N, num_classes)
        """
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
