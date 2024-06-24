"""
CNN Base Model
"""

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    CNN Base Model
    """
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        """
        Initialization

        :param input_shape: input shape (C, H, W)
        :param num_classes: number of classes
        """
        super(CNN, self).__init__()
        C, H, W = input_shape
        self.conv1 = nn.Conv2d(C, 6, kernel_size=5)
        H, W = (H - 5 + 1) // 2, (W - 5 + 1) // 2  # Shape after Conv2d & MaxPool2d
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        H, W = (H - 5 + 1) // 2, (W - 5 + 1) // 2  # Shape after Conv2d & MaxPool2d
        self.fc1 = nn.Linear(16 * H * W, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        Forward Pass

        :param x: input tensor of shape (N, C, H, W)
        :return: output tensor of shape (N, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
