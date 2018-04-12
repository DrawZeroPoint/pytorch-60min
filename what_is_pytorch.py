from __future__ import print_function
import torch
import numpy as np

# Getting Started
# Tensors
# Tensors are similar to NumPy's ndarrays, with the addition being that
# Tensors can also be used on a GPU to accelerate computing.

# Construct a 5x3 matrix, uninitialized:
x = torch.Tensor(5, 3)  # Tensor is automatically transferred into FloatTensor
print(x)

# Construct a randomly initialized matrix:
x = torch.rand(5, 3)
print(x)

# Get its size:
print(x.size())
print(torch.Size([5, 3]))


# Operations
# There are multiple syntaxes for operations.
# In the following example, we will take a look at the addition operation.

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

# Addition: providing an output tensor as argument
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition: in-place
# Any operation that mutates a tensor in-place is post-fixed with an _.
# For example: x.copy_(y), x.t_(), will change x.
y.add_(x)
print(y)

# You can use standard NumPy-like indexing with all bells and whistles!
# The columns' idx start from 0, and increase from left to right, so -1
# represent the column on the first column's left, i.e., the right most one
# note that the idx ranges from -3 to 2
print(y[:, -1])

# Resizing: If you want to resize/reshape tensor, you can use torch.view:
y = x.view(15)  # The original elements are traversed in row-wise
z = x.view(-1, 5)
print(y, z)


# NumPy Bridge
# The Torch Tensor and NumPy array will share their underlying memory locations,
# and changing one will change the other.

# Converting a Torch Tensor to a NumPy Array
a = torch.ones(5, 3)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# Converting NumPy Array to Torch Tensor
a = np.ones(5)  # We cannot use ones(5,5) to declare a matrix
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# CUDA Tensors
# Tensors can be moved onto GPU using the .cuda method.
if torch.cuda.is_available():
    with torch.cuda.device(0):
        print("CUDA available")
