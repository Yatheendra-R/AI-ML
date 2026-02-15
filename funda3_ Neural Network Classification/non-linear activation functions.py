import torch
from torch import nn
import matplotlib.pyplot as plt
import math
import numpy as np
X=np.arange(-5,6,1)
print("X:",X)
print()
#Sigmoid
"""
Y=1/(1+e^-x)
ranges 0,1
converts:
-inf =0
inf = 1
"""
print("Sigmoid")
Y=1/(1+math.e**(-1*X))
print(Y)
print(type(Y))

plt.plot(X,Y,label="Sigmoid")
plt.legend()
plt.xlabel("Input (X)")      
plt.ylabel("Output (Y)") 
plt.title("Sigmoid Function")   
plt.show()
#Relu
#Y=max(0,X)  makes negative to zero
print("Relu")
#Y=torch.max(np.array(0),X)
Y=[]
for i in X:
    Y.append(max(np.int64(0),i))

Y=np.array(Y)
print(Y)
print(type(Y))


plt.plot(X,Y,label="Relu")
plt.legend()
plt.xlabel("Input (X)")      
plt.ylabel("Output (Y)") 
plt.title("Relu Function")   
plt.show()

"""
ReLU function: f(x)=max(0,x)
Rectified Linear Unit
    f(x)=max(0,x)

Which means:
    If x > 0 → return x
    If x ≤ 0 → return 0

Left side → flat at 0
Right side → straight line
That “bend” at 0 is what introduces non-linearity.

It introduces a bend at 0.
That bend breaks linearity.
Stack many of them → complex shapes.

Linear model can only draw this:  ------------

Non-linear model can draw this:   ~~~~~~~

That’s why:
    Linear regression → straight line
    Neural networks → curves, shapes, complex patterns


ReLU Makes Network Non-Linear

Without ReLU: Linear → Linear → Linear   = still linear.

With ReLU:   Linear → ReLU → Linear → ReLU → Linear

    Negative values are cut off.
    Information flow becomes conditional.
    The function gains the ability to bend.

That break at zero makes the network powerful.

| Activation | Problem                          |
| ---------- | -------------------------------- |
| Sigmoid    | Vanishing gradient               |
| Tanh       | Still shrinks gradients          |
| ReLU       | Fast & avoids vanishing gradient |


"""
"""
Derivative of sigmoid:

σ'(x)=σ(x)/(1−σ(x))

Important fact:

    Maximum derivative = 0.25
    Near extremes → derivative ≈ 0

That means:

    When x is very positive or negative:
    gradient ≈ 0

Why That Kills Deep Networks
Backpropagation multiplies gradients layer by layer.

If each layer gives:  0.1 × 0.1 × 0.1 × 0.1 × 0.1

After 5 layers: 0.00001
After 20 layers: Almost zero.

This is the vanishing gradient problem.

Early layers stop learning.

Tanh — Better But Still Shrinks
Formula: tanh(x)=(e^x-e^-x)/(e^x+e^-x)

Improvements Over Sigmoid
    Output range: (-1, 1)
    Zero-centered (better for optimization)

Derivative:  1−tanh^2(x)

Max derivative = 1
Better than sigmoid (0.25).

But Still a Problem

    For large |x|:  tanh ≈ -1 or 1
    Derivative → 0
    So saturation still happens.
    It shrinks gradients — just slower than sigmoid.
            ​
ReLU — Why It Avoids Vanishing Gradient
    Formula: f(x)=max(0,x)

For positive inputs: gradient = 1

Not 0.25
Not shrinking
Not saturating

So during backprop:  1 × 1 × 1 × 1 × 1

Gradient stays strong.
That’s why deep networks became possible after ReLU was introduced.
"""

