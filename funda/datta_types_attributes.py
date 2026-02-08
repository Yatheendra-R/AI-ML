import torch
tens=torch.randn(2,3,4)
print("Tensor: ",tens)
print("Data type: ",tens.dtype)  #dtype is a attribute
print()

print("conversion")
tens_int=tens.int()
print("Data type: ",tens_int.dtype)
tens_long=tens.long()
print("Data type: ",tens_long.dtype)

tens_int32 = tens.to(dtype=torch.int32)  #same as tens_int=tens.int()
#tens_int32 = tens.type(dtype=torch.int32)  
print("Data type: ",tens_int32.dtype)
print()

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

#float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device

#assign dtype while creating
tens_crt=torch.tensor([1.2,3.4,5.5],
                      dtype=torch.int32,     # defaults to None, which is torch.float32 or whatever datatype is passed
                      device=None,            # defaults to None(cpu), which uses the default tensor type 
                      requires_grad=False) # if True, operations performed on the tensor are recorded 
print(tens_crt)
print()

"""
default data type is float32
32 byte

there are
16byte    half pression
32byte   single pression         
64byte   double pression
has byte increase more space is required and more pression(less data is lost)
has byte decreases  less space is required and less pression(more data is lost) , faster

float16 → half → very fast, low precision
float32 → default → balanced speed & precision
float64 → double → very precise, slower

| dtype                                       | Description                   | Example                                                                       
| ------------------------- | -------------------- | --------------------------------------- |
| torch.float32 / torch.float     | 32-bit floating point     |  x = torch.tensor([1.,2.])
| torch.float64 / torch.double | 64-bit float                    |  x = torch.tensor([1.,2.], dtype=torch.double)
| torch.float16 / torch.half       | 16-bit float                     |  x = torch.tensor([1.,2.], dtype=torch.half)
| torch.int64/ torch.long          | 64-bit integer               |  torch.tensor([1,2])                    
| torch.int32 / torch.int            | 32-bit integer               |  torch.tensor([1,2], dtype=torch.int32)
| torch.int16 / torch.short        | 16-bit integer                |  torch.tensor([1,2], dtype=torch.short)                   
| 8-bit integer                          |8-bit integer                   | torch.tensor([1,2], dtype=torch.int8)         
| torch.uint8                             | 8-bit unsigned integer | torch.tensor([0,255], dtype=torch.uint8)                
| torch.bool                               | Boolean values             | torch.tensor([True, False])


| Task                                                      | Recommended dtype |
| --------------------------------  | ----------------- |
| Standard neural networks                 | float32           |
| Mixed precision training (fast GPU)   |  float16          |
| Scientific simulations                          | float64          |

float32 → ~7 decimal digits precision
float64 → ~16 decimal digits precision



Note: Tensordatatypes is one of the 3 big errors you'll run into with PyTorch & deep learning:
1. Tensors not right datatype
2. Tensors not right shape
3. Tensors not on the right device
"""
