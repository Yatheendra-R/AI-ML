#Reproducibility
import torch
import numpy as np
ten_int=torch.randint(0,10,(2,3)) #range , (2,3) shape
print("tensor with int ",ten_int)

ten1=torch.rand(2,3)  #shape , default type is float 32 0<= rand <1

ten2=torch.rand(2,3)  
print("tensor with float ",ten1)
print("tensor with float ",ten2)
print(ten1==ten2)
#creates  different random value 
#print(np.where(ten1>0.3))

print()
#to create a same random value use seed

seed=42
torch.manual_seed(seed)
ten1=torch.rand(2,3)  #shape , default type is float 32 0<= rand <1
torch.manual_seed(seed)
ten2=torch.rand(2,3)
print("tensor with float ",ten1)
print("tensor with float ",ten2)
print(ten1==ten2)
"""
What is reproducibility?

Reproducibility means:
If you run the same code, with the same data, you should get the same results every time.

Without it:
    model accuracy changes
    loss curves look different
    debugging becomes a nightmare

Changing the seed:
    Changes random values
    Does NOT change the distribution
"""







