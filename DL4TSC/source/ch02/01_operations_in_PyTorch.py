"""
@File         : 01_operations_in_PyTorch.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 21:54:55
@Email        : cuixuanstephen@gmail.com
@Description  : 
"""

import torch

t1 = torch.tensor([1, 2, 3], dtype=int)
print(t1)
t2 = torch.tensor([[1, 2], [3, 4]])
print(t2)

import numpy as np

np_array = np.array([5, 6, 7])
t3 = torch.from_numpy(np_array)
print(t3)

t4 = torch.zeros(size=(3, 3))
print(t4)
t5 = torch.ones(size=(2, 2))
print(t5)
t6 = torch.eye(n=3)
print(t6)

dot_product = torch.dot(t3, t3)
print(dot_product)

matrix_product = torch.mm(t5, t5)
print(matrix_product)

t_transposed = t2.T
print(t_transposed)

# 行列式 the determinant of a matrix
det = torch.det(t5)
print(det)
# 逆矩阵 the inverse of a matrix
inverse = torch.inverse(t2)
print(inverse)
