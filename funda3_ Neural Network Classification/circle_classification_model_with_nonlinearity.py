import torch
from torch import nn
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import pandas as pd

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)
n_Sample=1000
X,Y=make_circles(n_Sample,noise=0.03,random_state=123)
print(X[:5])
print(X.shape)
print(Y[:5])
print(Y.shape)

plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.RdYlBu)
plt.show()

X=torch.from_numpy(X).to(dtype=torch.float)
Y=torch.from_numpy(Y).to(dtype=torch.float)
print(type(Y))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=123)
print(len(X_train))
print(X_train[:5])
print(len(X_test))
print(X_test[:5])
print(len(Y_train))
print(Y_train[:5])
print(len(Y_test))
print(Y_test[:5])

class Circle_classi_nonlinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=2,out_features=8)
        self.layer2=nn.Linear(in_features=8,out_features=16)
        self.layer3=nn.Linear(in_features=16,out_features=8)
        self.layer4=nn.Linear(in_features=8,out_features=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.layer4(self.relu(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))))

CCN=Circle_classi_nonlinear()

def accuracy(Y_pred_lab,Y_train):
    return ((torch.eq(Y_pred_lab,Y_train).sum().item())/len(Y_train))*100
epochs=3300
loss_fn=nn.BCEWithLogitsLoss() 
optimizer=torch.optim.SGD(params=CCN.parameters(),lr=0.1)
Y_test_loss_arr=[]
Y_train_loss_arr=[]
Y_test_acc_arr=[]
Y_train_acc_arr=[]
epoch_cnt=[]
for epoch in range(epochs):
    CCN.train()
    Y_pred_logit=CCN(X_train).squeeze()
    Y_pred_lab=torch.round(torch.sigmoid(Y_pred_logit))
    Y_loss_train=loss_fn(Y_pred_logit,Y_train)
    Y_train_acc=accuracy(Y_pred_lab,Y_train)
    optimizer.zero_grad()
    Y_loss_train.backward()
    optimizer.step()
    CCN.eval()
    with torch.inference_mode():
        Y_pred_logit_test=CCN(X_test).squeeze()
        Y_pred_lab_test=torch.round(torch.sigmoid(Y_pred_logit_test))
        Y_loss_test=loss_fn(Y_pred_lab_test,Y_test)
        Y_test_acc=accuracy(Y_pred_lab_test,Y_test)
    if(epoch%100==0):
        print(f"epoch: {epoch} |Train Loss: {Y_loss_train:.5f}  | Test Loss: {Y_loss_test:.5f}  | |  Train accuracy: {Y_train_acc:.2f}% |Test accuracy: {Y_test_acc:.2f}%")
        epoch_cnt.append(epoch)
        Y_test_loss_arr.append(Y_loss_test.detach().numpy())   #to remove grad
        Y_train_loss_arr.append(Y_loss_train.detach().numpy())
        Y_test_acc_arr.append(Y_test_acc)
        Y_train_acc_arr.append(Y_train_acc)


plt.figure(figsize=(8,5))
plt.plot(epoch_cnt,Y_test_loss_arr,label="Test loss")
plt.plot(epoch_cnt,Y_train_loss_arr,label="Train loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(epoch_cnt,Y_test_acc_arr,label="Test Acc")
plt.plot(epoch_cnt,Y_train_acc_arr,label="Train Acc")
plt.title("Training and test ACC curves")
plt.ylabel("ACC")
plt.xlabel("Epochs")
plt.legend()
plt.grid(True)
plt.show()
        



from helper_functions import plot_predictions, plot_decision_boundary   #using the help from other files 
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(CCN, X_train, Y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(CCN, X_test, Y_test)
plt.show()


"""
What Is Linearity?

A model is linear if the output is just a weighted sum of inputs:
y=w1​x1​+w2​x2​+b
That’s a straight line.
No matter how many linear layers you stack: Linear → Linear → Linear
It is STILL linear.
So stacking linear layers does NOT increase power.

What Is Non-Linearity?

A function is non-linear if the relationship is NOT a straight line.

Examples:

    Curves
    Waves
    Parabolas
    Step functions

Non-Linearity Is Needed in Neural Networks

Without activation functions:


    self.layer1 = nn.Linear(1,16)
    self.layer2 = nn.Linear(16,8)
    self.layer3 = nn.Linear(8,1)
    return self.layer3(self.layer2(self.layer1(x)))

Just a linear transformation.

3-layer network behaves like: y = ax + b


That means it CANNOT learn:

    Circles
    Spirals
    Images
    Speech patterns
    XOR problem

Where Non-Linearity Comes From

We add activation functions like:
    ReLU
    Sigmoid
    Tanh

Linear → ReLU → Linear → ReLU → Linear
Now the model can bend space.

| Without Non-Linearity         | With Non-Linearity              |
| ----------------------------- | ------------------------------- |
| Just linear regression        | Universal function approximator |
| Cannot model complex patterns | Can model almost anything       |
| No hidden power               | True deep learning              |


A neural network without activation functions is just a fancy linear regression.

"""











