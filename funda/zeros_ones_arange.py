import torch
print("Ones")
ones1=torch.ones(3)

print(ones1)
print()
oness=torch.ones((3,3,3))
"""
Struct
[
    [
        [  , , , ]
        []
        []
    ]
    [
        [  , , , ]
        []
        []
    ]
    [
        [  , , , ]
        []
        []
    ]
]                                   
"""

                    
print(oness)
print("dim: ",oness.ndim)
print("Shape: ",oness.shape)
print()

#Zeroes
print("Zeroes")
zeroes=torch.zeros((1,3,2))
"""
struct
[
  [
        [,]
        []
        []
    ]
]
"""
print(zeroes)
print("dim: ",zeroes.ndim)
print("shape: ",zeroes.shape)
print()

#arange
range_val=torch.arange(start=1,end=37,step=3)  #end is exclusive
#dtype is int 
#if only one parameter is passed , it takes it has end , default start and step value is 0
print("Rnage values: ",range_val)
print()
reshaped_range_val=range_val.reshape(1,3, 4)
"""
To find the total number of value in the tensor mul all each shape , here 1*3*4=12
when you reshape the array of n dim ,number of value in n dim should match
with total number of values in reshape (mul all parameter of reshape)
"""
print("reshaped_range_val:",reshaped_range_val)
print("dim:",reshaped_range_val.dim)
print("shape:",reshaped_range_val.shape)
print()

#_like
print("_like")
#This will keep the dim and shape of inputed tensor
print("zeroes_LK")
zeroes_LK=torch.zeros_like(input=reshaped_range_val)
print(zeroes_LK)
print("dim: ",zeroes_LK.ndim)
print("shape: ",zeroes_LK.shape)

print()

print("ones_LK")
ones_LK=torch.ones_like(input=reshaped_range_val)
print(ones_LK)
print()

"""
ones_like and zeros_like can work with dtype int, but for randn and rand it needs float,
generally tensor gives dtype as float , but in arange by default it gives dtypes as int 

"""
print("rand_LK")
rand_LK=torch.rand_like(input=reshaped_range_val.float())
print(rand_LK)
print()

print("randn_LK")
randn_LK=torch.randn_like(input=reshaped_range_val.float())
print(randn_LK)
print()

