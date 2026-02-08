import torch
"""
Manipulating Tensors (tensor operations)
Tensor opertions include:
(element-wise)
+ Addition
- Subtraction
/ Division
* Multiplication 


Matrix multiplication

One of the most common operations in machinelearning and deep learning algorithms
(like neural networks) is matrix multiplication
"""
tor=torch.tensor([1,2,3])
print("Tensor: ",tor)
MIN=tor.min()
print("MIN: ",MIN)
pos_min=torch.argmin(tor)
print("index of min value in the tensor: ",pos_min)
min_val, min_pos = torch.min(tor, dim=0)
print("Both min val and min index",min_val, min_pos)
"""
Why dim=-1 also works?
    Negative dimensions in PyTorch
    -1 means last dimension
    -2 means second last, etc.
"""


MAX=tor.max()
print("MAX: ",MAX)
pos_max=torch.argmax(tor)
print("index of max value in the tensor: ",pos_max)
max_val, max_pos = torch.max(tor, dim=0)
print("Both max val and max index",max_val, max_pos)

MEAN=torch.mean(tor.to(dtype=torch.float64))
#MEAN=torch.mean(tor.type(torch.float64))

print("MEAN: ",MEAN)
"""
RuntimeError: mean(): could not infer output dtype.
Input dtype must be either a floating point or complex dtype.Got: Long
"""
SUM=tor.sum()
print("SUM: ",SUM)
print()

x = torch.tensor([1, 2, 3])
y = torch.tensor([2, 2, 2])
print("X",x)
print("Y",y)
print("X>Y: ",x > y)   # tensor([False, False, True])
print("X==Y: ",x == y)  # tensor([False, True, False])


print()

print("2D")
A = torch.tensor([
    [1, 9, 3],
    [4, 2, 8]
])
print("Tensor A",A)
# Row-wise max (dim=1)
vals, pos = torch.max(A, dim=1)
print("max val for each row wise: ",vals)
print("max ind: ",pos)

# Column-wise min (dim=0)
vals, pos = torch.min(A, dim=0)
print("min val for each column wise: ",vals)
print("min ind: ",pos)

#Global max position (flattened index)
pos = torch.argmax(A)
max_val = torch.max(A)    #do not mention the dim
print("flattened index: ",pos)
print("Max value of flattened: ",max_val)

#Position where condition is true

tup=torch.where(A > 5)
print("Position where condition(A>5) is true",tup)
print("Type: ",type(tup))

print()

import torch as to

a=to.tensor([1,2,4])
print("A: ",a)
print("Type of A: ",a.dtype)
print()

b=to.tensor([1.,2.,4.])
print("B: ",b)
print("Type of B: ",type(b))
print()

print("These are element-wise operations, not matrix ops.")
print("Addition A+B: ",a+b)
print("SUB A-B: ",a - b)
print("MUL A*B: ",a * b)
print("Power A**B",a**b)
print("DIV A/B: ",a / b)
print("MOD A%B: ",a % b)

print()

print("Scalar–tensor operations")
print("A*2: ",a*2)
print("A+2: ",a+2)

print()

A = to.tensor([[1., 2.],
                  [3., 4.]])

B = to.tensor([[5., 6.],
                  [7., 8.]])

print("A: ",A)
print("Shape of A: ",A.shape)
print("Dimension of A: ",A.dim())
print()
print("B: ",B)
print()

print("Element-wise multiplication A*B")
print(A*B)
print()

print("Matrix multiplication  A@B")
print(A@B)
# or print(to.matmul(A,B))
print()

"""
matrix mul
shape
a=(2,3)  b=(3,4)
3==3 so can mul the two matrix
c=a@b
c=(2,4)


shape
a=(2,3)  b=(4,3)
3!=4 so can not  mul the two matrix
 so we can transpose b
 x=b.T
 x shape (3,4)
 shape of b will not change
c= a@x
 c=(2,3)

"""




"""
Rule to remember forever

* → element-wise

@ or matmul → matrix multiply
"""


print("Broadcasting")
print()
a1=to.tensor([1.,2.,3.]) #1D
print("a1: ",a1)
print()
b1=to.tensor(5.2)  #scalar 
print("b1: ",b1)
print()
print("a1+b1: ",a1+b1)
print()
"""
[1.,2.,3.]+[5.2,5.2,5.2]=tensor([6.2000, 7.2000, 8.2000])
"""
a2=to.tensor([[1.,2.,3.],
             [4.,5.,6.]]
                )
print("a2: ",a2)
print()
#2D+ scalar
print("a2+b1: ",a2+b1)
print()
"""
[                           [
[1.,2.,3.],             + [5.2,5.2,5.2],
[4.,5.,6.]               [5.2,5.2,5.2]
]                            ]
"""
#Example 2 (2D + 1D)
print("a2+a1: ",a2+a1)
print()
"""
a2                         a1
[
[1.,2.,3.],             +[[1.,2.,3.]
[4.,5.,6.]              [1.,2.,3.]]
]
"""
"""
Broadcasting rules 

Two dimensions are compatible if:

They are equal

One of them is 1

Example:

(2,3)
(1,3)  ✅

(2,3)
(2,1)  ✅

(2,3)
(3,2)  ❌



"""
print("In-place operations (_) ")
x = to.tensor([1., 2., 3.])
print("Tensor x:",x)
x.add_(5)
print("Tensor x+5:",x)
print()
#_ means modifies the tensor in memory
#Dangerous when using gradients 

print("Reshaping tensors") 
t = to.arange(12)
print("Tensor using arange:",t)
print("shape: ",t.shape)

print("Reshape: ",t.reshape(3, 4))
#reshaping does not cahnge the shape of the original tensor (t)
print("view tensor : ",t.view(4, 3))    #works like reshape

print()

M = to.tensor([[1, 2, 3],
                  [4, 5, 6]])
print("Tensor M",M)
print("Transposed tensor M",M.T)
#no change in original M
#(2,3) → (3,2)

print()

z = torch.bernoulli(torch.rand(3, 3))  # 0 or 1 randomly
print("Tensor z",z)
"""
Performance tip (important later)

Avoid Python loops ❌
Use tensor ops ✅

Bad:

for i in range(1000):
    x[i] = x[i] * 2


Good:

x = x * 2
"""


"""
RAND
torch.rand()

Uniform distribution

Range: [0, 1)    #0<=x<1

torch.rand(3)

output:
tensor([0.12, 0.83, 0.45])


"""

"""
RANDN
randn = random numbers from a normal distribution
torch.randn()

Normal (Gaussian) distribution

You are NOT just getting “random numbers”.
Mean (μ) = 0
Standard deviation (σ) = 1
What is a Normal (Gaussian) distribution?

It’s the classic bell curve:

    Most values are near 0

    Fewer values far from 0

    Symmetric left and right
Rough rule:

    ~68% values in [-1, 1]

    ~95% values in [-2, 2]

    ~99.7% values in [-3, 3]
torch.randn(3)
tensor([-0.2, 1.5, -0.7])

Why randn is used more in ML

    Neural network weights work best with zero-mean

    Helps gradient flow

    Matches theory assumptions

📌 Rule of thumb:

Data simulation → rand

Weights → randn

"""
"""
Compare rand vs randn (intuition)

torch.rand()
    Uniform distribution

    Every number in [0,1) equally likely

    No negatives

0 ───────── 1


torch.randn()

    Bell curve

    Most near 0

    Includes negatives
      ^
      |      ***
      |    *******
      |  ***********
------------------>

Why ML prefers randn for weights

    Neural network learning depends on gradients.

    Problem with only positive weights (rand)

    Activations become biased

    Gradients flow poorly

    Learning becomes slow or unstable
Why randn helps

    Zero mean → balanced positive & negative

    Symmetry breaking

    Better gradient flow

 This is why almost all weight initializations are based on randn.
"""
