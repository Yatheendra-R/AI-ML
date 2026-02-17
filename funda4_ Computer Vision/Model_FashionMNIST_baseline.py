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
