from torch import nn
import torch.nn.functional as F
import torch


class AlexNet(nn.Module):
    def __init__(self, input_length, output_length):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.flatten = nn.Flatten()
        self.flatten_size = self._get_flattened_size(input_length)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.flatten_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_length),
        )

    def _get_flattened_size(self, input_length):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            x = self.features(dummy_input)
            x = self.flatten(x)
            flattened_size = x.shape[1]
        return flattened_size

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
