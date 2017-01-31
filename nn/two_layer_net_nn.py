import torch
import torch.nn as nn
from torch.autograd import Variable

"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation defines the model as a custom Module subclass. Whenever you
want a model more complex than a simple sequence of existing Modules you will
need to define your model this way.
"""

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

class TwoLayerNet(nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = nn.Linear(D_in, H)
    self.linear2 = nn.Linear(H, D_out)

  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must return
    a Variable of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Variables.
    """
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = nn.MSELoss(size_average=False)

learning_rate = 1e-4
for t in range(500):
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # override the __call__ operator so you can call them like functions. When
  # doing so you pass a Variable of input data to the Module and it produces
  # a Variable of output data.
  y_pred = model(x)

  # Compute and print loss. We pass Variables containing the predicted and true
  # values of y, and the loss function returns a Variable containing the loss.
  loss = loss_fn(y_pred, y)
  print(t, loss.data[0])
  
  # Zero the gradients before running the backward pass.
  model.zero_grad()

  # Backward pass: compute gradient of the loss with respect to all the learnable
  # parameters of the model. Internally, the parameters of each Module are stored
  # in Variables with requires_grad=True, so this call will compute gradients for
  # all learnable parameters in the model.
  loss.backward()

  # Update the weights using gradient descent. Each parameter is a Variable, so
  # we can access its data and gradients like we did before.
  for param in model.parameters():
    param.data -= learning_rate * param.grad.data
