"""
@File         : 03_training_feed_forward_neural_network.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-24 11:25:40
@Email        : cuixuanstephen@gmail.com
@Description  : 训练前馈神经网络
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        # super(Net, self) 表示要调用的基类是 Net 的父类。
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        # forward() 方法表示网络的前向传递。这是网络在将输入转换为输出时执行的计算。
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
print(net)

# Create a synthetic dataset
# 训练数据集
X = torch.randn(100, 10)
y = torch.randn(100, 1)

loss_fn = nn.MSELoss()  # 一个损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 优化器

for epoch in range(100):
    # Forward pass: compute predicted outputs by passing
    # inputs to the model
    output = net(X)  # 父类应该实现了 __call__

    # Compute loss
    loss = loss_fn(output, y)
    # Zero the gradients before running the backward pass
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss
    # with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer performs
    # an update on its parameters
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
