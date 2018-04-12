# Variable
# autograd.Variable is the central class of the package. It wraps a Tensor,
# and supports nearly all of operations defined on it. Once you finish your
# computation you can call .backward() and have all the gradients computed automatically.
#
# You can access the raw tensor through the .data attribute, while the gradient w.r.t.
# this variable is accumulated into .grad.

# There's one more class which is very important for autograd implementation - a Function.
#
# Variable and Function are interconnected and build up an acyclic graph,
# that encodes a complete history of computation. Each variable has a .grad_fn attribute
# that references a Function that has created the Variable (except for Variables created by
# the user - their grad_fn is None).
#
# If you want to compute the derivatives, you can call .backward() on a Variable.
# If Variable is a scalar (i.e. it holds a one element data), you don't need to specify
# any arguments to backward(), however if it has more elements, you need to specify a
# gradient argument that is a tensor of matching shape.

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

y = 2 * x + 2
print(y)
print(y.grad_fn)

# grad can be implicitly created only for scalar outputs
# so if y is not a scalar but an array, we cannot use y.backward()

z = y * y * 3
out = z.mean()  # out = z.max() all have the same gradient
print("z and out: ", z, out)

out.backward()
print(x.grad)

x = torch.randn(3)
print(x)
x = Variable(x, requires_grad=True)
y = x * 2
print("data of y:", y.data)

# data.norm is different from norm
# http://pytorch.org/docs/stable/torch.html#torch.norm
while y.data.norm() < 10:
    y = y * 3;

print(y)

grad = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(grad)
print(x.grad)  # 3*3*2
