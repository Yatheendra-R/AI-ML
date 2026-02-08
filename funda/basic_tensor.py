import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#print(torch.__version__)

#scalar single val
print("Scalar")
scalar=torch.tensor(5)
print(scalar)
print("dim:",scalar.ndim)
print("shape:",scalar.shape)

print()
#vector 1-D
print("Vector")
vector=torch.tensor([1,2,3])
print(vector)
print("dim: ",vector.ndim)
print("shape:",vector.shape)
print()
#matrix 2-D
matrix=torch.tensor([[1,2,3],
                    [1,2,3],
                    [1,2,3]])
print(matrix)
print("dim:",matrix.ndim)  #number of open brackets at the staring 
print("shape:",matrix.size)
"""
size is built in function or method, which is callable just printing size gives
shape: <built-in method size of Tensor object at 0x00000255265E0EF0> is a method reference
size should be called to give the size
"""

"""
shape is not callable, shape is an attribute (property) 
"""
#inside a pair of square bracket , cnt the number of square brackets(3,3)  first 3 is due to 3 pair of square brackets,
#second 3 is due to number of element in the inner square bracket

#Tensor
print()
print("TENSOR")
TENSOR=torch.tensor(
    [
        [                                                 #1
            [1,2,3,4],                                 #2   #4  is the shape
            [1,2,3,4]    
        ]
    ]                                                    #3 is dim
)
print("dim:",TENSOR.dim())
#ndim → attribute → number of dimensions
#dim() → method → number of dimensions
print("shape: ",TENSOR.shape)

#rand
print()
print("Random")
RandTensor=torch.rand(1,2,3)     #value in is dim
#or RandTensor=torch.(size=rand(1,2,3))     #value in is dim

print(RandTensor)
print("dim:",RandTensor.ndim)  #number of value/ element inside 
print("Shape:",RandTensor.shape)



