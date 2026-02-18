import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

SEED=42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
print("imp")

train_data=datasets.FashionMNIST(root="data",download=False,train=True,transform=ToTensor())
test_data=datasets.FashionMNIST(root="data",download=False,train=False,transform=ToTensor())
print(f"train_data => len: {len(train_data)}")
print(f"test_data => len: {len(test_data)}")
#print(test_data) speacial object

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

#Input layer -> [Convolutional layer -> activation layer -> pooling layer] -> Output layer
class Model_MNIST_CNN(nn.Module):
    def __init__(self,input_shape,hidden_shape,output_shape):
        super().__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                     kernel_size=(3,3),
                     padding=1,
                     stride=1,
                     out_channels=hidden_shape),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape,
                     kernel_size=(3,3),
                     padding=1,
                     stride=1,
                     out_channels=hidden_shape),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )

        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_shape,
                     kernel_size=(3,3),
                     padding=1,
                     stride=1,
                     out_channels=hidden_shape),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape,
                     kernel_size=(3,3),
                     padding=1,
                     stride=1,
                     out_channels=hidden_shape),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
            

        #works likemulticlassifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_shape*7*7,
                      out_features=output_shape))
    
    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        #print(x.shape)  #torch.Size([10, 14, 14])
        x = self.block2(x)
        #print(x.shape) #torch.Size([10, 7, 7])

        x = self.classifier(x)
       # print(x.shape)
        return x

MMC=Model_MNIST_CNN(1,10,len(label_class_names))

"""
x_random_in=torch.rand(size=(1,1,28,28))
print(x_random_in.shape)

torch.Size([10, 14, 14])
torch.Size([10, 7, 7])
used find the shape of in_feautes after flattening

y_pred=MMC(x_random_in)
"""

# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), 
                             lr=0.1)
