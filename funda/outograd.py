"""
Theory

Autograd = automatic differentiation engine
It:

Records tensor operations
Builds a computation graph
Applies chain rule automatically
Computes gradients

differentiation is fundamentally about finding the instantaneous rate of change of a function
(dy/dx)

gradient measures the rate and direction of the steepest change in a function or physical quantity
"""
import torch as to
"""
x=to.tensor(3.0,requires_grad=True)  #Track everything that happens to x

If requires_grad=False:

tensor is invisible to autograd

no gradients

 Only tensors with requires_grad=True participate in learning
"""
x=to.tensor(3.0,requires_grad=True)  #Track everything that happens to x
y=x*x  #y=x^2
y.backward()
print(x.grad)

"""
y = x²
dy/dx = 2x
at x=3 → 6
👉 This is backpropagation
output
tensor(6.)
"""

#Computation Graph  ->Every operation creates a node.

x = to.tensor(2.0, requires_grad=True)
y = x * x  #y=x^2
z = y + 3   #z=y+3 = x^2
z.backward()
print(z.grad_fn)
#If someone asks for gradients, I know how to backprop through multiplication.
print(x.grad)
#x → square → y → add → z
"""
PyTorch stores:

    operations
    dependencies
    backward functions
Graph is built dynamically (define-by-run) ->This is why PyTorch is flexible.
"""
#what does .backward() does?
"""
z.backward()
Compute ∂z / ∂(everything that has requires_grad=True)
Results stored in: x.grad

.backward() works only on scalar outputs  beacuse Gradient must flow from one number

"""
