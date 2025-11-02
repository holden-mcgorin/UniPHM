import torch
import torch.nn as nn

# 定义简单 RNN 模型


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, h_n = self.rnn(x)  # 输出和隐藏状态
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出

        return out

#
# if __name__ == '__main__':
#     # 模拟数据：batch_size=2，seq_len=5，input_size=10
#     x = torch.randn(2, 5, 10)  # 输入序列
#     y = torch.tensor([[1.0], [0.0]])  # 模拟输出
#
#     # 初始化模型
#     model = RNN(input_size=10, hidden_size=20, output_size=1)
#     PyTorchUtil.count_paras(model)
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#     # 训练一步
#     pred = model(x)
#     loss = loss_fn(pred, y)
#     loss.backward()
#     optimizer.step()
#
#     print(f"Predicted: {pred}")
