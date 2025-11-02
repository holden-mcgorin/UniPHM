import torch
from torch import nn


class MyLSTM(nn.Module):

    def __init__(self, input_dim=14, hidden_dim=32, n_layers=5, dropout=0.5, bid=True):
        super(MyLSTM, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers,
                               dropout=dropout, batch_first=True, bidirectional=bid)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim * bid, hidden_dim),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, x):
        # input shape [batch_size, 时间长度, 特征维度]
        # outputs = [batch size, src sent len,  hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        self.encoder.flatten_parameters()

        encoder_outputs, (hidden, cell) = self.encoder(x)
        # encoder_outputs = F.dropout(torch.relu(encoder_outputs), p=0.5, training=self.training)

        features = encoder_outputs[:, -1:].squeeze()
        x = self.regressor(features)
        # x = torch.sigmoid(x) * 125
        return x
