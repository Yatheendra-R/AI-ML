import torch
import numpy as np
arr=np.arange(1.0,5.0)
print("Numpy array: ",arr)
print("Array type: ",arr.dtype)   #default type of numpy array float64
TENSOR=torch.from_numpy(arr)
print("Array to Tensor: ",TENSOR)
print("Array to Tensor type : ",TENSOR.dtype)
arr1=TENSOR.numpy()
print("Tensor to Array: ",arr1)
print("Tensor to Array type: ",arr1.dtype)
"""
Converting numpy array to tensor , converted tensor has type float64
"""
print()

TEN=torch.arange(1.0,5.0)
print("Tensor :",TEN)
print("Tensor type: ",TEN.dtype)   #default type of Tensor array float32
arr2=TEN.numpy()
print("Tensor to numpy array: ",arr2)
print("Tensor to numpy array type: ",arr2.dtype)
TEN1=torch.from_numpy(arr2)
print("Tensor to Array : ",TEN1)
print("Tensor to Array TYPE: ",TEN1.dtype)
"""
Converting tensor to numpy array ,converted array has type float32 
"""

