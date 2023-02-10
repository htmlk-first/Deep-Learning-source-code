import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

x_data = torch.FloatTensor([[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]])
y_data = torch.FloatTensor([[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]])
index_y = torch.argmax(y_data,1)

W = torch.randn([4,3], requires_grad=True)
b = torch.randn([3], requires_grad=True)
optimizer = torch.optim.SGD([W,b], lr=0.01)

for step in range(50001):
  model = F.softmax(x_data.matmul(W)+b)
  cost = F.cross_entropy(model, index_y)
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

x_test = torch.FloatTensor([[1,8,8,8]])
model_test = F.softmax(x_test.matmul(W)+b)
model_test = torch.argmax(model_test,1)

if model_test.numpy() == 0:
  print("[1,8,8,8] is A.")
elif model_test.numpy() == 1:
  print("[1,8,8,8] is B.")
elif model_test.numpy() == 2:
  print("[1,8,8,8] is C.")