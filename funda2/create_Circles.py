import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



"""
Why use 2 * np.pi?

Because:
    2π radians=360
    And a full circle is 360°.

So:
    0 radians → start of circle
    2π radians → complete one full rotation
That’s why we generate angles from: 0→2π

"""
theta=np.linspace(0,2*np.pi,100)
#Generate evenly spaced numbers over a specified interval
radius=1
"""
Convert polar → Cartesian coordinates
x=rcos(θ)
y=rsin(θ)
"""
x=radius*np.cos(theta)
y=radius*np.sin(theta)
plt.axis('equal') 
plt.plot(x,y,color="green")
plt.title("Circle")
plt.show()

"""
split_test=0.2 #per
split_val=int(split_test*len(x))

x=torch.from_numpy(x).to(dtype=torch.float32).unsqueeze(1)
y=torch.from_numpy(y).to(dtype=torch.float32).unsqueeze(1)


x_train, x_test=x[0:len(x)-split_val],x[len(x)-split_val:]
y_train, y_test=y[0:len(x)-split_val],y[len(x)-split_val:]

"""

theta = theta / (2*np.pi)
theta_tensor = torch.from_numpy(theta).float().unsqueeze(1)

xy_tensor = torch.stack(
    (torch.from_numpy(x).float(),
     torch.from_numpy(y).float()),
    dim=1
)

split_test = 0.2
split_val = int(split_test * len(theta_tensor))

x_train = theta_tensor[:-split_val]
x_test  = theta_tensor[-split_val:]

y_train = xy_tensor[:-split_val]
y_test  = xy_tensor[-split_val:]


print("Shape: ",x_train.shape)
print("Length: ",len(x_train))


print("Shape: ",y_test.shape)
print("Length: ",len(y_test))


class Create_Circle_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Linear(in_features=1,out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64,out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128,out_features=64),
            nn.Tanh(),

            nn.Linear(in_features=64,out_features=2)
            )
    def forward(self,x):
        return self.seq(x)


CCM=Create_Circle_model()
CCM.eval()
with torch.inference_mode():
    full_pred = CCM(theta_tensor)

pred_x = full_pred[:, 0].numpy()
pred_y = full_pred[:, 1].numpy()

plt.axis('equal')
plt.plot(x, y, color="green", label="True Circle")
plt.plot(pred_x, pred_y, color="red", label="Predicted Circle")
plt.legend()
plt.title("Circle before train")
plt.show()
epochs=2210
epochs_cnt=[]
train_loss_arr=[]
test_loss_arr=[]

loss_fn=nn.MSELoss()  
optimizer=torch.optim.Adam(params=CCM.parameters(),lr=0.005)
for epoch in range(epochs):
    CCM.train()

    y_train_pred=CCM(x_train)
    train_loss=loss_fn(y_train_pred,y_train)
    train_loss_arr.append(train_loss)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    CCM.eval()
    with torch.inference_mode():
        y_test_pred=CCM(x_test)
        test_loss=loss_fn(y_test_pred,y_test)
        test_loss_arr.append(test_loss)
    if(epoch%100==0):
        print(f"epoch : {epoch} | Train loss: {train_loss} | Test loss: {test_loss}")

CCM.eval()
with torch.inference_mode():
    full_pred = CCM(theta_tensor)

pred_x = full_pred[:, 0].numpy()
pred_y = full_pred[:, 1].numpy()

plt.figure(figsize=(6,6))
plt.axis('equal')
plt.plot(x, y, color="green", label="True Circle")
plt.plot(pred_x, pred_y, color="red", label="Predicted Circle")
plt.legend()
plt.title("Circle After Training")
plt.show()
