"""
squeeze only removes dimensions whose size is 1
example [2,3] no change
[1,3]= [3]
here :1 is number of row
3 is number of element in each row , or column
[3,1]=[3]

here 3 is number of row
1 is number of element in each row , or column

squeeze:
❌ does NOT touch data
❌ does NOT recompute values
✅ updates shape & stride

Usually returns a view (no copy)

So it is O(1) — very fast.

nternal decision rule (this is exact)

For each dimension i:

if size[i] == 1:
    remove dimension i
else:
    keep dimension i
"""
import torch
l=[[1,2,3],[4,5,6]]
t=torch.tensor(l)
print(t.dim())
print(t.size())

print(t)
print(torch.squeeze(t,dim=0))
print(torch.squeeze(t,dim=1))   #Nothing changes

"""
Does t have any dimension with size = 1?
How many rows?
→ 2
dim 0 → size 2 ❌
How many elements in each row?
→ 3
dim 1 → size 3 ❌
shape = [2, 3]

"""
print(torch.unsqueeze(t,0))

print(torch.unsqueeze(t,1))


l1=[[1],[2],[3],[4],[5],[6]]

t1=torch.tensor(l1)
print(t1.dim())
print(t1)
print(torch.squeeze(t1,dim=0))
print(torch.squeeze(t1,dim=1))


l2=[[1,2,3,4,5,6]]

t2=torch.tensor(l2)
print(t2.dim())
print(t2)
print(torch.squeeze(t2,dim=0))
print(torch.squeeze(t2,dim=1))

t3=torch.tensor(
    [
        [
            [1,2,3],
            [4,5,6]
        ]
    
    ]
)
print(t3.dim())
print(t3.size())  #([1,2,3])  # 1 number of block inside first bracket , # 2 in second bracket ...
print(torch.squeeze(t3,dim=0))
"""
What does “dimension” actually mean?
A dimension = one axis of data.
Think in terms of how many brackets you see.

0-D tensor (scalar)
5
Shape: []
Dimensions: 0
Just a single number


1-D tensor (vector)
[1, 2, 3]
Shape: [3]
Dimensions: 1

One axis → length = 3
“3 features” or “3 values”


2-D tensor (matrix)
[
 [1, 2, 3],
 [4, 5, 6]
]

Shape: [2, 3]
Dimensions: 2
Axis-0 → rows (2)
Axis-1 → columns (3)

“2 samples, each with 3 features”



The batch dimension


Without batch (single sample)
[1, 2, 3]
Shape: [3]


With batch
[
 [1, 2, 3]
]
Shape: [1, 3]

Batch size = 1
Features   = 3

What unsqueeze(0) really does

It adds a new axis at position 0.

Before: [3]
After : [1, 3]

[1, 2, 3]
↓ unsqueeze(0)
[[1, 2, 3]]


What squeeze(0) does?
It removes an axis ONLY if its size is 1.

[[1, 2, 3]]
↓ squeeze(0)
[1, 2, 3]

Before: [1, 3]
After: [3]


[N, C, H, W]
N → batch size (how many samples)
C → channels (like RGB = 3, grayscale = 1)
H → height
W → width
[H, W]
→ [1, H, W]      (batch)
→ [1, 1, H, W]   (channel)
dim 0 → batch
dim 1 → features / channels
rest  → structure (H, W, etc.)


Example:
[
  [1,2,3],
  [4,5,6]
]

Outer brackets → 1st dimension
Inner brackets → 2nd dimension
2-D structure

Number of rows = 2
Number of elements in each row = 3
Shape = [2, 3]

2 things
each having 3 things
dim 0 → which row
dim 1 → which element inside the row

Batch size = 2
Features   = 3
[batch, features]->[2,3]

Sample 1 → [1,2,3]
Sample 2 → [4,5,6]

Single sample (no batch):
[1,2,3]        → Shape [3]

Your example (batched):
[[1,2,3],
 [4,5,6]]      → Shape [2,3]




"""

"""
[
    [
        [,],
        [,]
    ],
    [
        [,],
        [,]
    ]
]

shape=[2,2,2]
Shape = [4, 1, 2]



Number of dimensions = 3

dim 0 has size 4
dim 1 has size 1
dim 2 has size 2


Here is a valid tensor with shape [4,1,2]:

[   
    [
        [1, 2]
    ],
    [
        [3, 4]
    ],
    [
        [5, 6]
    ],
    [
        [7, 8]
    ]
]


Let’s check carefully.

Verify each dimension
🔹 dim 0 (outermost)

How many big blocks? 4

dim 0 size = 4
These are:

[[1,2]]
[[3,4]]
[[5,6]]
[[7,8]]

🔹 dim 1 (inside each block)
Inside EACH block:[[1,2]]

How many inner lists? 1
dim 1 size = 1

dim 2 (innermost values)

Inside:[1,2]

How many values? 2
dim 2 size = 2

Final confirmed shape
[4, 1, 2]
dim 0 → batch (N)
dim 1 → channels / features (C)
dim 2 → length
So this means:

Copy code
N = 4  → 4 samples
C = 1  → 1 channel per sample
Each channel has 2 values
In words:

Copy code
4 samples
each sample has 1 feature vector
each vector has length 2

"""

print("##################")
ten=torch.tensor(
[
  [
    [
        [1, 2, 3]
    ],
    [
        [4, 5, 6]
    ]
  ]
]
)
print(ten)
print(ten.dim())  #4
print(ten.size())  #[1,2,1,3]
print()
ten1=torch.squeeze(ten,dim=0)
print(ten1)  
print(ten1.dim())  #3
print(ten1.size())  #[2,1,3]
print()
ten2=torch.squeeze(ten1,dim=1)  #dim is the index of the size 
print(ten2)  
print(ten2.dim())  #2
print(ten2.size())  #[2,3]
print()

ten3=ten.squeeze()
print(ten3)  
print(ten3.dim())  #2
print(ten3.size())  #[2,3]
print()



"""
What is unsqueeze?
Short definition

unsqueeze(dim) adds a new dimension of size 1 at the given position.

If squeeze removes size-1 dimensions,
then unsqueeze creates size-1 dimensions.
"""

x1 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
#Shape[2, 3]  2 rows 3 columns
print(x1.unsqueeze(0)) #Insert a new dimension at position 0
"""
Before: [2, 3]
After : [1, 2, 3]

[
  [
    [1, 2, 3],
    [4, 5, 6]
  ]
]

"""

print(x1.unsqueeze(1))
""""
Before: [2, 3]
After : [2, 1, 3]
Output
[
  [
    [1, 2, 3]
  ],
  [
    [4, 5, 6]
  ]
]
"""
print(x1.unsqueeze(2))
"""
Before: [2, 3]
After : [2, 3, 1]
Output

[
  [[1], [2], [3]],
  [[4], [5], [6]]
]

Each element becomes a column vector
"""
"""
| Operation                   | Effect                  |
| --------------------------- | ----------------------- |
| `unsqueeze(dim)`            | add size-1 dimension    |
| `squeeze(dim)`              | remove size-1 dimension |
| `x.unsqueeze(d).squeeze(d)` | returns original        |
| `x.squeeze().unsqueeze()`   | not always reversible   |

"""

print(x1.unsqueeze(-1))


print(x1.unsqueeze(x1.dim()))

"""
Final mental model (EXAM GOLD)
unsqueeze = add a wrapper
squeeze   = remove a wrapper
"""

