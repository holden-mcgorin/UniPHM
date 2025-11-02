import torch
import torch.nn as nn
import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=8, num_layers=2,
                 transpose=False, sigmoid=False):
        super().__init__()
        self.transpose = transpose
        self.sigmoid = sigmoid

        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, output_dim)

    def forward(self, src):
        if self.transpose:
            src = src.transpose(1, 2)  # 允许用户传 [B, C, T] → [B, T, C]

        src = self.input_linear(src)
        src = self.pos_encoder(src)  # [B, T, d_model]
        encoded = self.transformer_encoder(src)  # [B, T, d_model]

        out = self.output_linear(encoded[:, -1, :])  # 每个样本最后时间步
        return torch.sigmoid(out) if self.sigmoid else out


if __name__ == '__main__':
    model = TransformerEncoderModel(input_dim=10, output_dim=1, d_model=64, nhead=8)
    x = torch.randn(32, 100, 10)  # [batch, seq_len, input_dim]
    y = model(x)
    print(y.shape)  # torch.Size([32, 1])
