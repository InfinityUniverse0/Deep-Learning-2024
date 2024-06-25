"""
MLP Base Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod


class MLPBase(nn.Module):
    """
    MLP Base Model
    """
    def __init__(self, input_shape=(28, 28), num_classes=10):
        """
        Initialization

        :param input_shape: input shape (C, H, W) or (H, W)
        :param num_classes: number of classes
        """
        super(MLPBase, self).__init__()
        self.fc1 = nn.Linear(prod(input_shape), 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 80)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(80, num_classes)

    def forward(self, x):
        """
        Forward Pass

        :param x: input tensor of shape (N, C, H, W) or (N, H, W)
        :return: output tensor of shape (N, num_classes)
        """
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        return x
