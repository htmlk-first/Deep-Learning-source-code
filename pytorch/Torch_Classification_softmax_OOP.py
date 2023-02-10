import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

x_data = torch.FloatTensor([[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]])
y_data = torch.FloatTensor([[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]])
index_y = torch.argmax(y_data,1)

class softmaxClassifierModel(nn.Module): 
   def __init__(self): 
     super().__init__() 
     self.linear = nn.Linear(4,3) 
   def forward(self, x): 
     return F.softmax(self.linear(x))
model = softmaxClassifierModel() 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01) 

for epoch in range(50001): 
  prediction = model(x_train)  
  cost = F.cross_entropy(prediction, index_y)  
  pred = torch.argmax(prediction,1)
  accuracy = (np.round(pred == index_y)).mean()
  optimizer.zero_grad() 
  cost.backward() 
  optimizer.step()
print("Accuracy: ", accuracy.numpy())

x_test = torch.FloatTensor([[1,8,8,8]])
model_test = model(x_test.matmul(W)+b)
model_test = torch.argmax(model_test,1)
if model_test.numpy() == 0:
  print("[1,8,8,8] is A.")
elif model_test.numpy() == 1:
  print("[1,8,8,8] is B.")
elif model_test.numpy() == 2:
  print("[1,8,8,8] is C.")