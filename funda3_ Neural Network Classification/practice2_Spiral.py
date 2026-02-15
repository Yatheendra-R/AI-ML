"""
Create a multi-class dataset using the spirals data creation function from CS231n (see below for the code).
Construct a model capable of fitting the data (you may need a combination of linear and non-linear layers).
Build a loss function and optimizer capable of handling multi-class data (optional extension: use the Adam optimizer instead of SGD, you may have to experiment with different values of the learning rate to get it working).
Make a training and testing loop for the multi-class data and train a model on it to reach over 95% testing accuracy (you can use any accuracy measuring function here that you like).
Plot the decision boundaries on the spirals dataset from your model predictions, the plot_decision_boundary() function should work for this dataset too.
"""
# Code for creating a spiral dataset from CS231n
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import random
SEED=42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
n_sample=1000

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

print(X[:5])
X=torch.from_numpy(X).to(dtype=torch.float)
Y=torch.from_numpy(y).to(dtype=torch.long)
unique_elements, counts = torch.unique(
    Y, return_inverse=False, return_counts=True
)
print(unique_elements)
print(counts)

class spiral_multi_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Linear(in_features=2,out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8,out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16,out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8,out_features=3),
            )
    def forward(self,x):
        return self.seq(x)
SMM=spiral_multi_model()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

epochs=200
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=SMM.parameters(),lr=0.1)

def accuracy(Y_train,Y_pred):
    return ((torch.eq(Y_pred,Y_train).sum().item())/len(Y_pred))*100


for epoch in range(epochs):
    SMM.train()
    Y_train_logits=SMM(X_train) 
    Y_train_prob=torch.softmax(Y_train_logits,dim=1)
    Y_train_lab=torch.argmax(Y_train_prob,dim=1)
    Y_loss_train=loss_fn(Y_train_logits,Y_train)
    Y_train_acc=accuracy(Y_train,Y_train_lab)

    optimizer.zero_grad()
    Y_loss_train.backward()
    optimizer.step()

    SMM.eval()
    with torch.inference_mode():
         Y_test_logits=SMM(X_test)
         Y_test_prob=torch.softmax(Y_test_logits,dim=1)
         Y_test_lab=torch.argmax(Y_test_prob,dim=1)
         Y_loss_test=loss_fn(Y_test_logits,Y_test)
         Y_test_acc=accuracy(Y_test,Y_test_lab)
    if(epoch%20==0):
        print(f"epoch {epoch} | Loss Train {Y_loss_train:.5f}   |  Loss Test {Y_loss_test:.5f}   ||   ACC Train {Y_train_acc:.2f}%  | Acc Train {Y_test_acc:.2f}%")


    


from helper_functions import plot_predictions, plot_decision_boundary   #using the help from other files 
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(SMM, X_train, Y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(SMM, X_test, Y_test)
plt.show() 
    






