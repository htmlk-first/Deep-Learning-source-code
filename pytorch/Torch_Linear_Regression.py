import torch

x_data = torch.FloatTensor([[1,1],[2,2],[3,3]])
y_data = torch.FloatTensor([[10],[20],[30]])
W = torch.randn([2,1], requires_grad=True)
b = torch.randn([1], requires_grad=True)
optimizer = torch.optim.SGD([W,b], lr=0.01)

for step in range(2001):
  model = torch.matmul(x_data,W)+b
  cost = torch.mean((model-y_data)**2)
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()
print(W.detach().numpy())
print(b.detach().numpy())

x_test = torch.FloatTensor([[4,4]])
model_test = torch.matmul(x_test,W)+b
print("model for [4,4]: ", model_test.detach().numpy())