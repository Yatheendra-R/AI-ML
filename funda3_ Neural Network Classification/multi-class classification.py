import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
n_sample=1000
X_blob,Y_blob=make_blobs(n_samples=n_sample,
                         n_features=2,  # X features
                         
                         #Each feature = one axis. Data lives in 2D: (x1, x2)
                        #Feature 0 → x-axis
                        #Feature 1 → y-axis
                        
                         centers=4, # y labels 
                         cluster_std=1.5, # give the clusters a little shake up (try changing this to 1.0, the default)
                         random_state=42)
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=Y_blob, cmap=plt.cm.RdYlBu);
plt.show()
print(Y_blob.dtype)
X_blob=torch.from_numpy(X_blob).to(dtype=torch.float32)
Y_blob=torch.from_numpy(Y_blob).to(dtype=torch.long)

X_train,X_test,Y_train,Y_test=train_test_split(X_blob,Y_blob,test_size=0.2,train_size=0.8,random_state=42)
print(len(X_train))
print(type(X_test))


class multi_class_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=2,out_features=8)
        self.layer2=nn.Linear(in_features=8,out_features=16)
        self.layer3=nn.Linear(in_features=16,out_features=8)
        self.layer4=nn.Linear(in_features=8,out_features=4)

        #we can add relu and work but does not make any difference becuase this dataset and be trainned only using linear fn
    def forward(self,x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))
MCM=multi_class_model()

MCM.eval()
with torch.inference_mode():
    Y_train_pred_logits=MCM(X_train)
Y_train_pred_prob=torch.softmax(Y_train_pred_logits,dim=1)
Y_train_pred_lab=torch.argmax(Y_train_pred_prob,dim=1)

"""
We use dim=1 because:

    dim=0 → down columns
    dim=1 → across classes (row)
"""

print(Y_train_pred_lab[:5])
print(Y_train[:5])


from helper_functions import plot_predictions, plot_decision_boundary   #using the help from other files 
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(MCM, X_train, Y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(MCM, X_test, Y_test)
plt.show()
#train

epochs=50
loss_fn=nn.CrossEntropyLoss() #uses logits/raw data
optimizer=torch.optim.SGD(params=MCM.parameters(),lr=0.1)   

def accuracy(Y_true,Y_test):
    return ((torch.eq(Y_true,Y_test).sum().item())/len(Y_true))*100

epochs=30

Y_test_loss_arr=[]
Y_train_loss_arr=[]
Y_test_acc_arr=[]
Y_train_acc_arr=[]
epoch_cnt=[]
for epoch in range(epochs):
    MCM.train()
    Y_train_pred_logits=MCM(X_train)
    Y_train_pred_prob=torch.softmax(Y_train_pred_logits,dim=1)
    Y_train_pred_lab=torch.argmax(Y_train_pred_prob,dim=1)
    Y_train_loss=loss_fn(Y_train_pred_logits,Y_train) 
    Y_train_acc=accuracy(Y_train,Y_train_pred_lab)
    optimizer.zero_grad()

    Y_train_loss.backward()
    optimizer.step()
    MCM.eval()
    with torch.inference_mode():
        Y_test_pred_logits=MCM(X_test)
    Y_test_pred_prob=torch.softmax(Y_test_pred_logits,dim=1)
    Y_test_pred_lab=torch.argmax(Y_test_pred_prob,dim=1)
    Y_test_loss=loss_fn(Y_test_pred_logits,Y_test)
    Y_test_acc=accuracy(Y_test,Y_test_pred_lab)

    if(epoch%10==0):
        print(f"epoch: {epoch} |Train loss: {Y_train_loss.item():.5f} | "
              f"Test loss: {Y_test_loss.item():.5f} | "
              f"Train acc: {Y_train_acc:.2f}% | "
              f"Test acc: {Y_test_acc:.2f}%")
        epoch_cnt.append(epoch)
        Y_test_loss_arr.append(Y_test_loss.detach().numpy())   #to remove grad
        Y_train_loss_arr.append(Y_train_loss.detach().numpy())
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

        

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(MCM, X_train, Y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(MCM, X_test, Y_test)
plt.show()    

"""
You only need .squeeze() when:

Your model outputs (N,1)
But your targets are (N)


You do NOT need squeeze when:

Your model outputs (N,C)
And targets are (N)


That is the multi-class standard format.
"""





