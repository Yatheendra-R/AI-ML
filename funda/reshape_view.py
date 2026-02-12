import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print("x:" ,x)

y = x.view(3, 2)

print("y:" ,y)
print("x:" ,x)
"""
view:

does NOT copy memory

only changes shape + strides

x and y share the same memory
"""

#Changing y changed x → same memory.
y[0, 0] = 999

print("y:" ,y)
print("x:" ,x)

z = x.t()      # transpose
print("t:",z)

print(z.reshape(6))
print(z.reshape(2,3))

"""
reshape:

tries to return a view

if impossible → makes a copy

Always works (if element count matches)
"""

print(x.reshape(-1)) #flatten tensor

print(x.reshape(2, -1)) #PyTorch computes -1 automatically.



"""
| Operation      | Use when                                       |
| -------------- | ---------------------------------------------- |
| `view()`       | You **know** tensor is contiguous & want speed |
| `reshape()`    | You want **safe code** (recommended)           |
| `flatten()`    | You want 1D                                    |
| `contiguous()` | Fix memory layout before `view()`              |
"""
"""
#########
#my understanding

Stride tells PyTorch how many memory steps to jump when you move by 1 along a dimension.

Memory index:  0  1  2  3  4  5
Values:              1  2  3  4  5  6

#before Transpose
 [[1, 2, 3],
[4, 5, 6]]

from 1 to 4(to next row) ,3 jumps(0 to 1 to 2 to 3 memory)
from 1 to 2 (to next column), 1 jump (0 to 1  memory)
 
Stride(3,1)
shape (2,3)


#after transpose
shape and stride changes but not the memory index
[
 [1, 4]
 [2, 5]
 [3,6]
 ]
 from 1 to 2 (to next row), 1 jump (0 to 1  memory)
from 1 to 4(to next column) ,3 jumps(0 to 1 to 2 to 3 memory)

Stride(1,3)
shape (3, 2)


memory_index = row_index * row_stride
             + col_index * col_stride



Why view() cares about strides

view() assumes: “Elements are laid out in memory in a simple, increasing order.”
                            Transpose violates this assumption.
                            z’s strides don’t match a flat walk.

So PyTorch says ❌ no.
so,
z = x.t()      # transpose
print(z.view(6))
print(z.view(3,2))
gives error


Why reshape() works

reshape() tries to return a view.
If that fails, it silently creates a copy.
reshape():
    if possible without copying:
        return view
    else:
        make new contiguous tensor
        copy data
        return new tensor



Danger

Copying memory means:
slower
more RAM

onle when else case is triggered:
breaks shared storage

Example:

a = z.reshape(6)
a[0] = 999
z[0,0]   # unchanged


Because they no longer share memory.


Ultimate mental model (memorize this)

view() = change shape without touching memory
reshape() = change shape, copy memory if needed


To Check contiguity
tensor.is_contiguous()

To Check memory sharing
a.storage().data_ptr() == b.storage().data_ptr()




view() fails when the tensor is non-contiguous, meaning the stride pattern cannot represent the new shape without reordering memory.

✔ Not “index changes”
✔ It’s about stride pattern not matching a linear walk



permute() does not move data.
It changes the stride, so PyTorch uses a different mathematical mapping from indices to memory addresses.
permute() uses pure index math — no data movement.
permute() only changes how indices are translated into memory addresses

x.shape   = (2, 3)
x.stride  = (3, 1)
Access:
x[i, j] → memory[i*3 + j*1]

After permute
Copy code
p = x.permute(1, 0)

p.shape  = (3, 2)
p.stride = (1, 3)
Access:
p[i, j] → memory[i*1 + j*3]

permute() does ONLY this:
changes shape
changes stride
keeps the same memory pointer
Access uses a formula (not a lookup table):
address = base_ptr
        + i0 * stride0
        + i1 * stride1
        + ...


q)
x.shape  = (6, 7, 8)
x.stride = (56, 8, 1)
x.permute(1, 2, 0)

permute(1, 2, 0) means:

new dim 0 ← old dim 1
new dim 1 ← old dim 2
new dim 2 ← old dim 0

(7, 8, 6) <- shape
(8, 1, 56) <-stride


permute() is fast because it only changes stride metadata and uses simple index arithmetic, without copying data.
contiguous() is slow because it physically copies and rearranges all elements in memory.
######

Stride tells PyTorch how many memory steps to jump when you move by 1 along a dimension.
That’s it. Nothing more mystical than that.
Start from something familiar: a flat array in memory

Take this tensor:
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
1 4
2 5 
3 6
Physically in memory (1D)
Even though it looks 2D, memory is flat:

Memory index:  0  1  2  3  4  5
Values:              1  2  3  4  5  6

Shape tells what, stride tells how

Shape: (2, 3) → 2 rows, 3 columns

Stride: (3, 1)

Now let’s decode (3, 1).
Meaning of stride = (3, 1)

Stride always matches the number of dimensions.

x.shape   = (2, 3)
x.stride()= (3, 1)

So:

First number → row stride = 3
To move down 1 row, jump 3 elements in memory

Example:

x[0, 0] → memory index 0
x[1, 0] → memory index 0 + 3 = 3

Which is correct:
x[1,0] = 4

Second number → column stride = 1

To move right 1 column, jump 1 element

Example:

x[0,0] → index 0
x[0,1] → index 0 + 1 = 1
x[0,2] → index 0 + 2 = 2

Visual rule (lock this in 🧠)
memory_index = row_index * row_stride
             + col_index * col_stride
             
For x[1,2]:= 1 * 3 + 2 * 1= 5
Memory[5] = 6

Now why transpose breaks things
z = x.t()

Shape becomes:(3, 2)

But memory is unchanged: [1, 2, 3, 4, 5, 6]

Stride becomes:(1, 3)

Meaning:
Move down rows → jump 1
Move across columns → jump 3

So PyTorch reads memory in a different pattern, without copying data.

This is why:
transpose is fast
but view() fails (memory order is weird)
Why view() cares about strides

view() assumes:
“Elements are laid out in memory in a simple, increasing order.”
Transpose violates this assumption.

z’s strides don’t match a flat walk.

So PyTorch says ❌ no.
"""

"""
Q)
x = torch.tensor([
    [10, 20, 30, 40],
    [50, 60, 70, 80]
])
# shape = (2, 4)
# stride = (4, 1)
(x[:, ::2])

[[10, 30],
 [50, 70]]
 
 shape = (2, 2)
 y.stride = (4, 2)


torage vs data_ptr

Storage = the raw contiguous memory block of all elements

data_ptr() = pointer to the first element in memory

Example
x = torch.tensor([[1,2],[3,4]])
x.storage()      # 1D memory: [1,2,3,4]
x.data_ptr()     # memory address of '1'


x[1,1] → memory index = 3 → value = 4

No matter how you slice or permute, storage() shows the underlying flat memory

"""
