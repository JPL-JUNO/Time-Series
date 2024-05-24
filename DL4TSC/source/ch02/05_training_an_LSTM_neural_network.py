"""
@File         : 05_training_an_LSTM_neural_network.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-24 14:49:16
@Email        : cuixuanstephen@gmail.com
@Description  : 
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        # 由于 LSTM 单元的内部结构，
        # 需要初始化隐藏状态 h0 和单元状态 c0。
        # 然后，这些状态作为元组与输入 x 一起传递到 LSTM 层。
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


lstm = LSTM(10, 20, 1)
print(lstm)

# Note that PyTorch’s LSTM expects inputs to be 3D in the format batch_size, seq_length,
# and num_features:

X = torch.randn(100, 5, 10)
Y = torch.randn(100, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(lstm.parameters(), lr=0.01)

for epoch in range(100):
    output = lstm(X)
    loss = loss_fn(output, Y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
