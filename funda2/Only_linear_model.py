import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


weight=0.7
bias=0.3
#y=wx+b

X=torch.arange(0,1,0.02).unsqueeze(dim=1)
print(X)

Y=weight*X+bias
print(Y)

plt.scatter(X,Y,c="b",s=4)
plt.show()


split=int(len(X)*0.8)
X_train=X[:split]
X_test=X[split:]
Y_train=Y[:split]
Y_test=Y[split:]

def ploting(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,predict=None):

    plt.scatter(X_train,Y_train,c="b",s=4,label="Training data")
    plt.scatter(X_test,Y_test,c="g",s=4,label="Testing data")
    if(predict is not None):
            plt.scatter(X_test,predict,c="r",s=4,label="Predict data")
    plt.legend(prop={"size": 14});
    plt.show()

ploting()


class learn(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, 
                                      out_features=1)
    
    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor): #-> torch.Tensor:
        return self.linear_layer(x)


torch.manual_seed(123)
PLM=learn()
print(PLM.state_dict())

with torch.inference_mode():
    Y_pred=PLM(X_test)

print(Y_pred)
ploting(predict=Y_pred)

# Check the predictions
print(f"Predicted values:\n{Y_pred}")
print(f"required values:\n{Y_test}")
print("difference b/w y_test-y_preds",Y_test-Y_pred)


torch.manual_seed(123)

epochs=5500
test_loss_y=[]
train_loss_y=[]
epochs_cnt=[]

loss_fn = nn.L1Loss()   
optimizer = torch.optim.SGD(PLM.parameters(), lr=0.001) 

for epoch in range(epochs):
    PLM.train()
    Y_train_pred=PLM(X_train)

    train_loss=loss_fn(Y_train_pred,Y_train)
    optimizer.zero_grad()

    train_loss.backward()
    optimizer.step()

    PLM.eval()

    with torch.inference_mode():
        Y_test_pred=PLM(X_test)
        test_loss=loss_fn(Y_test_pred,Y_test)

    if epoch % 10 == 0:
            epochs_cnt.append(epoch)
            train_loss_y.append(train_loss.detach().numpy())
            test_loss_y.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {train_loss} | MAE Test Loss: {test_loss} ")

        
plt.plot(epochs_cnt, train_loss_y, label="Train loss")
plt.plot(epochs_cnt, test_loss_y, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
        
print(PLM.state_dict())

PLM.eval()

with torch.inference_mode():
 
  test_Y_pred = PLM(X_test)
print(Y_test_pred)
print(Y_test)
ploting(predict=Y_test_pred)

    

