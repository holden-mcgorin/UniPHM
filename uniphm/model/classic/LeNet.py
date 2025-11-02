from torch import nn
import torch.nn.functional as F
import torch


class LeNet(nn.Module):
    def __init__(self, input_length, output_length):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5, stride=1)

        # Calculate the flattened dimension after the convolutions and pooling
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_length)
            flattened_size = self._get_flattened_size(dummy_input)

        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_length)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _get_flattened_size(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.flatten(x)
        return x.shape[1]
