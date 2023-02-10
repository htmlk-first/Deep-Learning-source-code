import torch
import numpy as np

x_data = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
y_data = torch.FloatTensor([[0],[1],[1],[0]])
W_h = torch.randn([2,3], requires_grad=True)
b_h = torch.randn([3], requires_grad=True)
W_o = torch.randn([3,1], requires_grad=True)
b_o = torch.randn([1], requires_grad=True)
optimizer = torch.optim.SGD([W_h,b_h,W_o,b_o], lr=0.1)

for step in range(40001):
  H1 = torch.sigmoid(x_data.matmul(W_h)+b_h) 
  model = torch.sigmoid(H1.matmul(W_o)+b_o)
  cost = ((-1)*y_data*torch.log(model)+(-1)*(1-y_data)*torch.log(1-model)).mean()
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

H1 = torch.sigmoid(x_data.matmul(W_h)+b_h)
model_test = torch.sigmoid(H1.matmul(W_o)+b_o)
prediction = np.round(model_test > 0.5).type(torch.float32)
accuracy = np.round(prediction == y_data).mean()
print("Model:", model_test.detach().numpy())
print("Prediction:", prediction.detach().numpy())
print("Accuracy: ", accuracy.detach().numpy())
