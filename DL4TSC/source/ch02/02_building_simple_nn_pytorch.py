"""
@File         : 02_building_simple_nn_pytorch.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 22:15:47
@Email        : cuixuanstephen@gmail.com
@Description  : 使用 PyTorch 构建简单的神经网络
"""

import torch

X = torch.randn(100, 10)
y = torch.randn(100, 1)

input_size = 10
hidden_size = 5
output_size = 1

# We use the requires_grad_() function to tell PyTorch that we want to compute gradients
# with respect to these tensors during the backward pass.

W1 = torch.randn(hidden_size, input_size).requires_grad_()
b1 = torch.zeros(hidden_size, requires_grad=True)
W2 = torch.randn(output_size, hidden_size).requires_grad_()
b2 = torch.zeros(output_size, requires_grad=True)


# Next, we define our model.
# For this simple network, we’ll use a sigmoid activation function for the
# hidden layer:
def simple_neural_net(x, W1, b1, W2, b2):
    z1 = torch.mm(x, W1.t()) + b1
    a1 = torch.sigmoid(z1)
    z2 = torch.mm(a1, W2.t()) + b2
    return z2


# Now, we’re ready to train our model.
# Let’s define the learning rate and the number of epochs:
lr = 1e-2
epochs = 100
loss_fn = torch.nn.MSELoss()

# 这个基本代码演示了神经网络的基本部分
for epoch in range(epochs):
    # 前向传递，计算预测
    y_pred = simple_neural_net(X, W1, b1, W2, b2)

    # 后向传递，计算梯度
    loss = loss_fn(y_pred.squeeze(), y)
    loss.backward()

    # 更新步骤，
    # 我们调整权重以最小化损失。
    with torch.no_grad():
        W1 -= lr * W1.grad
        b1 -= lr * b1.grad
        W2 -= lr * W2.grad
        b2 -= lr * b2.grad

    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()

    if epoch % 10 == 9:
        print(f"Epoch: {epoch} \t Loss: {loss.item()}")
