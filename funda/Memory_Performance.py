import torch as to
"""
Where does a tensor actually live?

Every tensor occupies real memory:

CPU tensor → RAM

GPU tensor → VRAM (GPU memory)
"""
x = to.tensor([1., 2., 3.])
print(x.device)
"""
y = x.to("cuda")
print(y.device)
cuda:0
CPU RAM and GPU VRAM are separate physical memories.
Why .to("cuda") is expensive

When you do:

x = x.to("cuda")


What happens:

Data copied from RAM → VRAM

Synchronization happens

GPU kernel launched

⛔ Doing this inside loops is very slow

❌ Bad:

for i in range(1000):
    x = x.to("cuda")


✅ Good:

x = x.to("cuda")
for i in range(1000):
    x = x * 2
"""

print("Contiguous vs non-contiguous memory")
"""
This matters for speed.
Contiguous tensor
Stored sequentially in memory.
"""
a = to.randn(3, 4)
print(a.is_contiguous())   # True

#Non-contiguous tensor
#Created after transpose/slicing.

b = a.T
print(b.is_contiguous())   # False
"""
Why this matters:

GPU kernels expect contiguous memory

Non-contiguous tensors are slower
"""
#Fix:
b = b.contiguous()

print()
#Views vs Copies (CRITICAL)
#View (shares memory)
#Same memory, no extra allocation
x = to.arange(6)
y = x.view(2, 3)

y[0, 0] = 999
print(x)
print()


#Copy (new memory)
#x is unchanged.
print("copy unchanged")
z = x.clone()
z[0] = -1
print(z)
print(x)
print()
"""
view() → shares memory
clone() → new memory
"""
print("In-place operations & memory")
x = to.tensor([1., 2., 3.])
y = x

x.add_(10)
print(y)

"""
Why?

x and y point to the same memory
This is why in-place ops can silently break things.
"""

"""
GPU memory is NOT freed immediately

This surprises everyone.

x = torch.randn(10000, 10000, device="cuda")
del x


VRAM still looks used.

Why?

PyTorch uses memory caching

For faster future allocations

Free cache manually (only if needed):

torch.cuda.empty_cache()


Avoid unnecessary tensor creation

❌ Bad:

for i in range(1000):
    x = torch.tensor([i])


✅ Good:

x = torch.empty(1000)
for i in range(1000):
    x[i] = i


Tensor creation is expensive.


Batch operations = speed

GPU loves large batches, hates small ops.

❌ Bad:

for i in range(1000):
    y = x[i] * 2


✅ Good:

y = x * 2


This is why ML uses batching.
"""


