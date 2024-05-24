"""
@File         : 04_training_a_recurrent_neural_network.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-24 11:42:07
@Email        : cuixuanstephen@gmail.com
@Description  : 训练循环神经网络
"""

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # initial hidden state
        # 确保初始
        # 隐藏状态张量 h0 与 x 输入张量位于同一设备上。

        # 在这个 RNN 的上下文中，
        # x 预计是一个具有形状 (batch_size, sequence_length, num_features) 的 3D 张
        # 量，因此 x.size(0) 将返回 batch_size。

        # batch_size：这表示一批数据中的数量。
        # sequence_length
        # 决定了模型在每个步骤中查看过去多少天的数据。
        # num_features：此维度表示数据序列每个时间步长中的特征
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        # RNN 返回输出和最终隐藏状态
        out, _ = self.rnn(x, h0)  # get RNN output

        # pass last output to Fully Connected layer
        out = self.fc(out[:, -1, :])

        return out


rnn = RNN(10, 20, 1)  # 10 features, 20 hidden units, 1 output
print(rnn)

X = torch.randn(100, 5, 10)  # 100 samples, 5 time steps, 10 features
Y = torch.randn(100, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)

for epoch in range(100):
    output = rnn(X)

    loss = loss_fn(output, Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
