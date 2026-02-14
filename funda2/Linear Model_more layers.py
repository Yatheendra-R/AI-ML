import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
import random

weight =0.3
bias=0.7
random.seed(123)
torch.manual_seed(123)
np.random.seed(123)
start =0
end=1
step=0.01

X=torch.arange(start,end,step).unsqueeze(1)
Y=X*weight+bias
class linear_model_multi_layer(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1=nn.Linear(in_features=1,out_features=16)
        self.layer2=nn.Linear(in_features=16,out_features=8)
        self.layer3=nn.Linear(in_features=8,out_features=1)
    def forward(self,x):
        return self.layer3(self.layer2(self.layer1(x))) #self.seq(x) -> using Sequential restrict the complete usage of layers in forward 

LML=linear_model_multi_layer()
print(X[:5])
print(X.shape)
print(Y[:5])
print(Y.shape)

epochs=100
epochs_cnt=[]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=123)



def ploting(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,predict=None):

    plt.scatter(X_train,Y_train,c="b",s=4,label="Training data")
    plt.scatter(X_test,Y_test,c="g",s=4,label="Testing data")
    if(predict is not None):
            plt.scatter(X_test,predict,c="r",s=4,label="Predict data")
    plt.legend();
    plt.show()

LML.eval()
with torch.inference_mode():
    Y_pred=LML(X_test)
print(Y_pred)
print(Y_test)
ploting(predict=Y_pred)


loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(params=LML.parameters(),lr=0.01)
epochs=300
loss_train_arr=[]
loss_pred_arr=[]
epo_cnt=[]
for epoch in range(epochs):
    LML.train()

    Y_train_pred=LML(X_train)

    loss_train=loss_fn(Y_train_pred,Y_train)

    optimizer.zero_grad()
    loss_train.backward()

    optimizer.step()

    LML.eval()
    with torch.inference_mode():
        Y_pred_test=LML(X_test)
        loss_test=loss_fn(Y_pred_test,Y_test)


    if(epoch%10==0):
        loss_train_arr.append(loss_train.item())
        loss_pred_arr.append(loss_test.item())
        epo_cnt.append(epoch)
        print(f"epoch: {epoch} | loss_train: {loss_train} |  loss_test: {loss_test}")

plt.plot(epo_cnt,loss_train_arr,label="loss_train")
plt.plot(epo_cnt,loss_pred_arr,label="test_train")
plt.legend()
plt.show()

LML.eval()
with torch.inference_mode():
    Y_pred=LML(X_test)
print(Y_pred)
print(Y_test)
ploting(predict=Y_pred)
#print(LML.state_dict())
learned_weight = LML.layer3.weight.detach().numpy()
learned_bias = LML.layer3.bias.detach().numpy()




