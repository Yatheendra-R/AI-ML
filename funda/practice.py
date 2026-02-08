import torch
import numpy as np
#Create a random tensor with shape (7, 7).
ten1=torch.rand(7,7)
print("Random tensor of shape 7,7: ",ten1)
#Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7)
ten2=torch.rand(1,7)
print("Random tensor of shape 1,7: ",ten2)
transpose_ten2=ten2.T
print("transpose_ten2  (1,7)->(7,1): ",transpose_ten2)

mul=ten1@transpose_ten2
print("mul (7,7)*(7,1)=(7,1): ",mul)

#Find the maximum and minimum values 
print("max value,",torch.max(mul))
print("min value,",torch.min(mul))
#Find the maximum and minimum index
print("max value index,",mul.argmax())
print("min value index,",mul.argmin())

"""Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor
with all the 1 dimensions removed to be left with a tensor of shape (10).
Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and
it's shape."""

seed=7
torch.manual_seed(seed)
tens=torch.rand(1,1,1,10)
print("Tensor: ",tens)
print("dim: ",tens.ndim)
print("shape: ",tens.shape)
tens_sq=tens.squeeze()
print("tensor_squeezed: ",tens_sq)
print("dim: ",tens_sq.ndim)
print("shape: ",tens_sq.shape)
