import torch

x = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
"""
This should work but it is not working
#print(x[::-1, :]) #all rows, but in reverse order 

#print(x[::-1, ::-1]) #Reverse both rows and columns.flip upside down and left-right

"""
print(x)
print(x[0][0])

"""
First [0] → selects row 0

Second [0] → selects column 0

Result is a scalar tensor, not a Python int
"""
print(x[1, 2])

print(x[0])

"""
Behind the scenes

PyTorch returns a 1D tensor

Shape becomes (3,)
"""


#slice
"""
(rows, columns)
x[row_index , column_index]

, -> is  to differentiate row and column

start:end:step
end is not inclusive
"""
print(x[:, 1])
print(x[0:2, 1:3])
print(x[:, 0:2])
print(x[-1])
"""
-1 → last row
Same rule as Python lists
"""


"""
start : end   (end NOT included)

"""

y = x[0]
y[0] = 999
print(x)
"""
y is a view, not a copy

Changing y changes x
"""

#: Make a copy (safe way) 
z = x[0].clone()
z[0] = 111
print(x)
print(z)

mask = x > 50
print(mask)
"""
tensor([[ True, False, False],
        [False, False,  True],
        [ True,  True,  True]])

"""
print(x[mask])
"""
only True elements
tensor([999, 60, 70, 80, 90])

"""
print(x[:, :])  #everything

print(x[::2, :]) #Every alternate row, all columns

print(x[:,1])  #only first column,tensor([20, 50, 80])

print(x[:,1:2])
"""
only first column but with the range ,
tensor([[20],
        [50],
        [80]])
"""
print(x[:,1:])
"""
tensor([[20, 30],
        [50, 60],
        [80, 90]])

"""

"""
x[ rows , columns ]
rows    → start:stop:step
columns → start:stop:step
    
"""
"""
What if tensor is 3D?

Then you’ll see two commas:

x[batch, height, width]


Each comma = move to next dimension.
"""




