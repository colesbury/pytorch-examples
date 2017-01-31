This repository introduces the fundamental concepts of
[PyTorch](https://github.com/pytorch/pytorch)
through self-contained examples.

At its core, PyTorch provides two main features:
- An n-dimensional Tensor, similar to numpy but can run on GPUs
- Automatic differentiation for building and training neural networks

We will use a fully-connected ReLU network as our running example. The network
will have a single hidden layer, and will be trained with gradient descent to
fit random data by minimizing the Euclidean distance between the network output
and the true output.

### Table of Contents

:CONTENTS

## PyTorch: Tensors

Numpy is a great framework, but it cannot utilize GPUs to accelerate its
numerical computations. For modern deep neural networks, GPUs often provide
speedups of [50x or greater](https://github.com/jcjohnson/cnn-benchmarks), so
unfortunately numpy won't be enough for modern deep learning.

Here we introduce the most fundamental PyTorch concept: the **Tensor**. A PyTorch
Tensor is conceptually identical to a numpy array: a Tensor is an n-dimensional
array, and PyTorch provides many functions for operating on these Tensors. Like
numpy arrays, PyTorch Tensors do not know anything about deep learning or
computational graphs or gradients; they are a generic tool for scientific
computing.

However unlike numpy, PyTorch Tensors can utilize GPUs to accelerate their
numeric computations. To run a PyTorch Tensor on GPU, you simply need to cast it
to a new datatype.

Here we use PyTorch Tensors to fit a two-layer network to random data. Like the
numpy example above we need to manually implement the forward and backward
passes through the network:

```python
:INCLUDE tensor/two_layer_net_tensor.py
```

## PyTorch: Variables and autograd

In the above examples, we had to manually implement both the forward and
backward passes of our neural network. Manually implementing the backward pass
is not a big deal for a small two-layer network, but can quickly get very hairy
for large complex networks.

Thankfully, we can use
[automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
to automate the computation of backward passes in neural networks. 
The **autograd** package in PyTorch provides exactly this functionality.
When using autograd, the forward pass of your network will define a
**computational graph**; nodes in the graph will be Tensors, and edges will be
functions that produce output Tensors from input Tensors. Backpropagating through
this graph then allows you to easily compute gradients.

This sounds complicated, it's pretty simple to use in practice. We wrap our
PyTorch Tensors in **Variable** objects; a Variable represents a node in a
computational graph. If `x` is a Variable then `x.data` is a Tensor, and
`x.grad` is another Variable holding the gradient of `x` with respect to some
scalar value.

PyTorch Variables have the same API as PyTorch Tensors: (almost) any operation
that you can perform on a Tensor also works on Variables; the difference is that
using Variables defines a computational graph, allowing you to automatically
compute gradients.

Here we use PyTorch Variables and autograd to implement our two-layer network;
now we no longer need to manually implement the backward pass through the
network:

```python
:INCLUDE autograd/two_layer_net_autograd.py
```

## PyTorch: nn
Computational graphs and autograd are a very powerful paradigm for defining
complex operators and automatically taking derivatives; however for large
neural networks raw autograd can be a bit too low-level.

When building neural networks we frequently think of arranging the computation
into **modules**, some of which have **learnable parameters** which will be
optimized during learning.

In PyTorch, the `nn` package defines a set of
**Modules**, which are roughly equivalent to neural network layers. A Module receives
input Variables and computes output Variables, but may also hold internal state such as
Variables containing learnable parameters. The `nn` package also defines a set of useful
loss functions that are commonly used when training neural networks.

The **Module** class is also used to assemble layers into larger structures. Here we
define our two-layer network as a **Module** containing several submodules.
**Module** must only implement the `forward` method; the backwards computation is
handled automatically by autograd.

```python
:INCLUDE nn/two_layer_net_nn.py
```

## PyTorch: optim
Up to this point we have updated the weights of our models by manually mutating the
`.data` member for Variables holding learnable parameters. This is not a huge burden
for simple optimization algorithms like stochastic gradient descent, but in practice
we often train neural networks using more sophisiticated optimizers like AdaGrad,
RMSProp, Adam, etc.

The `optim` package in PyTorch abstracts the idea of an optimization algorithm and
provides implementations of commonly used optimization algorithms.

In this example we will use the `nn` package to define our model as before, but we
will optimize the model using the Adam algorithm provided by the `optim` package:

```python
:INCLUDE nn/two_layer_net_optim.py
```

## PyTorch: RNNs

TODO

## Data Loading
We often want to load inputs and targets from files, instead of using random inputs. We also often want to do any preprocessing in the background to avoid slowing down the training loop. PyTorch provides two classes `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` to help with data loading. `DataLoader` implements batching and shuffling. It will load the data in background processes if you set `num_workers`.

```python
:INCLUDE nn/data_loading.py
```

PyTorch also provides a number of implementations for common datasets in the `vision` and `text` packages:
- Vision: MNIST, LSUN, COCO, CIFAR, and generic "ImageFolder"
- Text: SNLI, SST, and generic "Translation" and "LanguageModeling"

## PyTorch for Torch Users

The non-autograd parts of pytorch will be quite familiar to torch users, but there are
a few important changes to be aware of:

**Inplace / Out-of-place**

The first difference is that ALL operations on the tensor that operate in-place on it will have an **_** postfix.
For example, `add` is the out-of-place version, and `add_` is the in-place version.

```python
a.fill_(3.5)
# a has now been filled with the value 3.5

b = a.add(4.0)
# a is still filled with 3.5
# new tensor b is returned with values 3.5 + 4.0 = 7.5
```

Some operations like narrow do not have in-place versions, and hence, `.narrow_` does not exist. 
Similarly, some operations like `fill_` do not have an out-of-place version, so `.fill` does not exist.

 **Zero Indexing**

Another difference is that Tensors are zero-indexed. (Torch tensors are one-indexed)

```python
b = a[0,3] # select 1st row, 4th column from a
```

Tensors can be also indexed with Python's slicing

```python
b = a[:,3:5] # selects all rows, columns 3 to 5
```

**No camel casing**

The next small difference is that all functions are now NOT camelCase anymore.
For example `indexAdd` is now called `index_add_`

```python
x = torch.ones(5, 5)
print(x)
z = torch.Tensor(5, 2)
z[:,0] = 10
z[:,1] = 100
print(z)
x.index_add_(1, torch.LongTensor([4,0]), z)
print(x)
```

**Numpy Bridge**

Converting a torch Tensor to a numpy array and vice versa is a breeze.
The torch Tensor and numpy array will share their underlying memory, and changing one will change the other.

*Converting torch Tensor to numpy Array*

```python
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b) # see how the numpy array changed in value
```

*Converting numpy Array to torch Tensor*

```python
import numpy as np
a = np.ones(5)
b = torch.DoubleTensor(a)
np.add(a, 1, out=a)
print(a)
print(b) # see how changing the np array changed the torch Tensor automatically
```

All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

**CUDA Tensors**

CUDA Tensors are nice and easy in pytorch, and they are much more consistent as well.
Transfering a CUDA tensor from the CPU to GPU will retain it's type.

```python
# creates a LongTensor and transfers it 
# to GPU as torch.cuda.LongTensor
a = torch.LongTensor(10).fill_(3).cuda()
print(type(a))
b = a.cpu()
# transfers it to CPU, back to 
# being a torch.LongTensor
```

**Multiprocessing vs multithreading**

In Lua, CPU parallelism for data loading and HOGWILD typically used multithreading
via the torch `threads` package. In Python, CPU parallelism is achieved through the
torch `multiprocessing` package. This is a simple extension of the Python `multiprocessing`
package, that causes tensor storages to be passed between processes in shared memory.

Unlike the torch threads library, arbitrary objects (e.g. whole models) can be shared
between Python processes for HOGWILD training.

The [MNIST HOGWILD example](https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py) and the
[PyTorch data loader](https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py)
are good examples of how to use torch multiprocessing.

# Advanced Topics 

## PyTorch: Defining new autograd functions
Under the hood, each primitive autograd operator is really two functions that
operate on Tensors. The **forward** function computes output Tensors from input
Tensors. The **backward** function receives the gradient of the output Tensors
with respect to some scalar value, and computes the gradient of the input Tensors
with respect to that same scalar value.

In PyTorch we can easily define our own autograd operator by defining a subclass
of `torch.autograd.Function` and implementing the `forward` and `backward` functions.
We can then use our new autograd operator by constructing an instance and calling it
like a function, passing Variables containing input data.

In this example we define our own custom autograd function for performing the ReLU
nonlinearity, and use it to implement our two-layer network:

```python
:INCLUDE autograd/two_layer_net_custom_function.py
```


## TensorFlow: Static Graphs
PyTorch autograd looks a lot like TensorFlow: in both frameworks we define
a computational graph, and use automatic differentiation to compute gradients.
The biggest difference between the two is that TensorFlow's computational graphs
are **static** and PyTorch uses **dynamic** computational graphs.

In TensorFlow, we define the computational graph once and then execute the same
graph over and over again, possibly feeding different input data to the graph.
In PyTorch, each forward pass defines a new computational graph.

Static graphs are nice because you can optimize the graph up front; for example
a framework might decide to fuse some graph operations for efficiency, or to
come up with a strategy for distributing the graph across many GPUs or many
machines. If you are reusing the same graph over and over, then this potentially
costly up-front optimization can be amortized as the same graph is rerun over
and over.

One aspect where static and dynamic graphs differ is control flow. For some models
we may wish to perform different computation for each data point; for example a
recurrent network might be unrolled for different numbers of time steps for each
data point; this unrolling can be implemented as a loop. With a static graph the
loop construct needs to be a part of the graph; for this reason TensorFlow
provides operators such as `tf.scan` for embedding loops into the graph. With
dynamic graphs the situation is simpler: since we build graphs on-the-fly for
each example, we can use normal imperative flow control to perform computation
that differs for each input.

To contrast with the PyTorch autograd example above, here we use TensorFlow to
fit a simple two-layer net:

```python
:INCLUDE autograd/tf_two_layer_net.py
```

# PyTorch: HOGWILD

TODO @sgross

## PyTorch: Control Flow and Weight Sharing
As an example of dynamic graphs and weight sharing, we implement a very strange
model: a fully-connected ReLU network that on each forward pass chooses a random
number between 1 and 4 and uses that many hidden layers, reusing the same weights
multiple times to compute the innermost hidden layers.

For this model can use normal Python flow control to implement the loop, and we
can implement weight sharing among the innermost layers by simply reusing the
same Module multiple times when defining the forward pass.

We can easily implement this model as a Module subclass:

```python
:INCLUDE nn/dynamic_net.py
```
