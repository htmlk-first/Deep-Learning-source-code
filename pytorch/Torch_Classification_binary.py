import torch
import numpy as np

x_data = torch.FloatTensor([[1,2],[2,3],[3,4],[4,3],[5,3],[6,2]])
y_data = torch.FloatTensor([[0],[0],[0],[1],[1],[1]])

W = torch.randn([2,1], requires_grad=True)
b = torch.randn([1], requires_grad=True)
optimizer = torch.optim.SGD([W,b], lr=0.01)

for step in range(2001):
  model = torch.sigmoid(x_data.matmul(W)+b)
  cost = ((-1)*y_data*torch.log(model)+(-1)*(1-y_data)*torch.log(1-model)).mean()
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

model_test = torch.sigmoid(x_data.matmul(W)+b)
prediction = np.round(model_test > 0.5).type(torch.float32)
accuracy = np.round(prediction == y_data).mean()
print("Model: ", model_test.detach().numpy())
print("Prediction: ", prediction.detach().numpy())
print("Accuracy: ", accuracy.detach().numpy())