import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles  #getting data sets
#sklearn is simple and efficient tools for predictive data analysis and is built on top of other scientific libraries like NumPy, SciPy, and Matplotlib.
#Sklearn is mostly used for machine learning, PyTorch is designed for deep learning.
import pandas as pd
n_samples=1000
X,Y=make_circles(n_samples,  
                 noise=0.03,              #with out this it makes perfect circle
                 random_state=42)
"""
X tells the exact position of each dot in 2D space.
 Y tells which group that dot belongs to.
"""
print("First 10 sample of X(features):\n ",X[:10])
print("First 10 sample of Y(Labels):\n ",Y[:10])
#two X values per one y value.
table_circles =pd.DataFrame({"X1":X[:,0],"X2":X[:,1],"Labels":Y})
"""
: means → all rows
0 means → column index 0
"""
#print(table.head(10))  #first 10
print(table_circles.head(10))

print("Count of unique label(y):")
print(table_circles["Labels"].value_counts())

plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.RdYlBu)
plt.show()

"""
c = Y
This controls color of each dot.
    If label is 0 → one color
    If label is 1 → another color

cmap = plt.cm.RdYlBu

This means:
    Rd → Red
    Yl → Yellow
    Bu → Blue

It is a color gradient map.
    Label 0 might appear red,
    Label 1 might appear blue.
"""

print("Shape of X features: ",X.shape)
print("Shape of Y Labels: ",Y.shape)
print("Type of X features and Y Lables: ",type(X)," ",type(Y))
#it is in numpy , need to converted into the tensor for learning
X=torch.from_numpy(X).to(dtype=torch.float)  #converted to tensor , type float32
Y=torch.from_numpy(Y).to(dtype=torch.float)
print("Coverted Type of X features and Y Lables(For learning): ",type(X)," ",type(Y))




