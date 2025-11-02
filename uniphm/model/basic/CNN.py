import torch
from torch import nn
import torch.nn.functional as F

from uniphm.model.basic.SEBlock import SEBlock1D


class CNN(nn.Module):
    def __init__(self, input_size, output_size, transpose=True, end_with_sigmoid=False):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=8)
        self.se1 = SEBlock1D(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, padding=1)
        self.se2 = SEBlock1D(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, padding=1)
        self.se3 = SEBlock1D(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size - 1184, 128)
        self.fc2 = nn.Linear(128, output_size)

        self.transpose = transpose
        self.end_with_sigmoid = end_with_sigmoid

    def forward(self, x):
        if self.transpose:
            x = x.transpose(-1, -2)
        x = self.conv1(x)
        x = self.se1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.se2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.se3(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.end_with_sigmoid:
            x = torch.sigmoid(x)
        return x
