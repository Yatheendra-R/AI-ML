"""
Make a binary classification dataset with Scikit-Learn's make_moons() function.
For consistency, the dataset should have 1000 samples and a random_state=42.
Turn the data into PyTorch tensors. Split the data into training and test sets using train_test_split with 80% training and 20% testing.
"""
import torch
from torch import nn
import numpy as np
from  sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

SEED=42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
n_sample=1000
X,Y=make_moons(n_samples=n_sample,
               noise=0.03,
               random_state=42
               )
print(X[:5])
print(Y[:5])
print(type(X))
print(X.dtype)
print(type(Y))
print(Y.dtype)
plt.scatter(X[:,0],X[:,1],c=Y, cmap=plt.cm.RdYlBu)
plt.show()

X=torch.from_numpy(X).to(dtype=torch.float)
Y=torch.from_numpy(Y).to(dtype=torch.float)

class binary_class_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=2,out_features=8)
        self.layer2=nn.Linear(in_features=8,out_features=16)
        self.layer3=nn.Linear(in_features=16,out_features=8)
        self.layer4=nn.Linear(in_features=8,out_features=1)
        self.relu=nn.ReLU()
    def forward(self,x):
        return self.layer4(self.relu(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

BCM=binary_class_model()
BCM.eval()
with torch.inference_mode():
    Y_train_pred_logit=BCM(X_train).squeeze()
print(Y_train[:5])
print(Y_train_pred_logit[:5])
Y_train_prob=torch.sigmoid(Y_train_pred_logit)
Y_train_label=torch.round(Y_train_prob)
print(Y_train_label[:5])


from helper_functions import plot_predictions, plot_decision_boundary   #using the help from other files 
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(BCM, X_train, Y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(BCM, X_test, Y_test)
plt.show()

def accuracy(Y_train,Y_pred):
    return ((torch.eq(Y_pred,Y_train).sum().item())/len(Y_pred))*100

#train
loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(params=BCM.parameters(),lr=0.1)
epochs=620

for epoch in range(epochs):
    BCM.train()
    Y_train_pred_logits=BCM(X_train).squeeze()
    Y_train_pred_prob=torch.sigmoid(Y_train_pred_logits)
    Y_train_pred_lab=torch.round(Y_train_pred_prob)
    Y_train_loss=loss_fn(Y_train_pred_logits,Y_train)
    Y_train_acc=accuracy(Y_train_pred_lab,Y_train)
    optimizer.zero_grad()
    Y_train_loss.backward()
    optimizer.step()


    BCM.eval()
    with torch.inference_mode():
        Y_test_pred_logit=BCM(X_test).squeeze()
    Y_test_prob=torch.sigmoid(Y_test_pred_logit)
    Y_test_label=torch.round(Y_test_prob)
    Y_test_loss=loss_fn(Y_test_pred_logit,Y_test)
    Y_test_acc=accuracy(Y_test_label,Y_test)

    if(epoch%20==0):
        print(f"epoch {epoch} | Loss Train {Y_train_loss:.5f}   |  Loss Test {Y_test_loss:.5f}   ||   ACC Train {Y_train_acc:.2f}%  | Acc Train {Y_test_acc:.2f}%")


    


from helper_functions import plot_predictions, plot_decision_boundary   #using the help from other files 
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(BCM, X_train, Y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(BCM, X_test, Y_test)
plt.show() 




