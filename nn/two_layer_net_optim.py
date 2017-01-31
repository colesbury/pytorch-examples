import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation uses the optim package to perform optimization.
"""

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

class TwoLayerNet(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(TwoLayerNet, self).__init__()
    self.linear1 = nn.Linear(D_in, H)
    self.linear2 = nn.Linear(H, D_out)

  def forward(self, x):
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred


model = TwoLayerNet(D_in, H, D_out)
loss_fn = nn.MSELoss(size_average=False)

# The optim package contains a number of data loaders, such as
# SGD, Adagrad, Adadelta, and Adam. You can optimize different parts
# of the model with different optimizers by passing a subset of
# model.parameters() to each optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
  y_pred = model(x)

  loss = loss_fn(y_pred, y)
  print(t, loss.data[0])
  
  model.zero_grad()

  loss.backward()

  # Performs one SGD step on the parameters.
  # Since the gradients are stored inside the parameter Variables,
  # `step()` requires no arguments.
  optimizer.step()
