import torch
from torch import nn
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader

SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


train_data=datasets.FashionMNIST(root="data",train=True,transform=ToTensor())
test_data=datasets.FashionMNIST(root="data", train=False,transform=ToTensor())

print(f"Number of  train data: {len(train_data)}")
print(f"Number of test data: {len(test_data)}")

label_class_names=train_data.classes
print("Number of  Unique items:",label_class_names)
print("Unique items:",len(label_class_names))


image,label=train_data[0]
print(f"Image shape: {image.shape}")
#print(f"Label shape: {label.shape}")  error , label has single scalar value
plt.imshow(image.squeeze())
plt.title(label_class_names[label])
plt.axis(False)
plt.show()

plt.imshow(image.squeeze(),cmap="grey")
plt.title(label_class_names[label])
plt.axis(False)
plt.show()


fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(label_class_names[label])
    plt.axis(False);
plt.show()

BATCH_SIZE=32
train_data_load=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
test_data_load=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)


print(f"Dataloaders: {train_data_load, test_data_load}") 
print(f"Length of train dataloader: {len(train_data_load)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_data_load)} batches of {BATCH_SIZE}")
train_features_batch, train_labels_batch=next(iter(train_data_load))  #takes first batch
print(train_features_batch.shape)  #torch.Size([32, 1, 28, 28]
print(train_labels_batch.shape) # torch.Size([32])  , 32 items in each batch - each has one label.

#The nn.Flatten() layer took our shape from [color_channels, height, width] to [color_channels, height*width].
#28*28=784
class Model_MNIST_baseline(nn.Module):
    def __init__(self):
        super(). __init__()
        self.seq=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=10)
            )

    def forward(self,x):
            return self.seq(x)
#max  accuracy without ReLu Train Loss=0.4536 | Test Loss=0.4823 | Train Acc=84.16% | Test Acc=82.47%
#max  accuracy with ReLu  Train Loss=0.3700 | Test Loss=0.3968 | Train Acc=86.57% | Test Acc=85.51%


"""
Original image shape

For a grayscale image in FashionMNIST:

Each image is [1, 28, 28]
1 → color channel (grayscale)
28 → height
28 → width
So the image is a 2D grid of pixels (28×28) with 1 channel.

What nn.Flatten() does

nn.Flatten()
This takes all the pixels from height × width and turns them into a 1D vector.
So [1, 28, 28] becomes [1*28*28] = [784].
If batch size is included: [batch_size, 1, 28, 28] → [batch_size, 784].

Analogy:

    Think of the 28×28 pixel grid like a spread-out chessboard.
    Flattening = taking all the squares and laying them in a single row.

Why do we flatten before nn.Linear()?

Linear (fully connected) layers expect input in the shape: [batch_size, num_features]
Here, num_features = 28*28 = 784
They cannot take 2D images directly, because the weights in a linear layer are just a matrix multiplication.

Formula of Linear layer: Y=XW^T+b

X must be [batch_size, num_features]
W is [out_features, in_features]

Without flattening, PyTorch would not know how to multiply your 28×28 image with the weight matrix.
"""
MMB=Model_MNIST_baseline()
print(MMB)

def accuracy(y_true,y_pred):
    return ((torch.eq(y_true,y_pred).sum().item())/(len(y_true)))*100
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(params=MMB.parameters(),lr=0.1)


epochs=3

for epoch in range(epochs):
    print(f"Epoch: {epoch}\n-------")

    train_acc=0;    #python var
    train_loss=0;
    #(X, y) Each batch from the DataLoader returns a tuple
    #enumerate() is a Python function that returns both the index and the value when looping.
    MMB.train()
    for batch, (X, y) in enumerate(train_data_load):
        y_pred_logits=MMB(X)
        
        loss=loss_fn(y_pred_logits,y)
        train_pred_loss=torch.softmax(y_pred_logits,dim=1)
        optimizer.zero_grad()                   
        loss.backward()                    
        optimizer.step()
        train_loss += loss.item()           #Do NOT use .item() before loss.backward(),.item() extracts the scalar value as a plain Python float from a 1-element tensor.
        train_acc+=accuracy(y,torch.argmax(train_pred_loss,dim=1))
        #Mixing Python numbers and tensors is not allowed directly in older PyTorch versions or can create subtle bugs.
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_data)} samples")

    train_acc=train_acc/len(train_data_load)
    train_loss=train_loss/len(train_data_load)

    MMB.eval()
    test_acc=0;  
    test_loss=0;
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_data_load):
            y_pred_logits=MMB(X)
            loss = loss_fn(y_pred_logits, y)

            test_loss+=loss.item()
            test_pred_loss=torch.softmax(y_pred_logits,dim=1)
            test_acc+=accuracy(y,torch.argmax(test_pred_loss,dim=1))

            

        test_acc=test_acc/len(test_data_load)
        test_loss=test_loss/len(test_data_load)
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Test Loss={test_loss:.4f} | Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%\n")


"""
Lower loss is better. The train loss is slightly smaller than the test loss, which is normal
— it means the model is learning but hasn’t overfit yet.

From the looks of things, it seems like our model is overfitting on the training data.

Overfitting means our model is learning the training data well but those patterns aren't generalizing to the testing data.

Two of the main ways to fix overfitting include:

Using a smaller or different model (some models fit certain kinds of data better than others).
Using a larger dataset (the more data, the more chance a model has to learn generalizable patterns).
"""

    
