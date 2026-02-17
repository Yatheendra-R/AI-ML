import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms  import ToTensor
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.utils.data import DataLoader

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
"""
root: str - which folder do you want to download the data to?
train: Bool - do you want the training or test split?
download: Bool - should the data be downloaded?
transform: torchvision.transforms - what transformations would you like to do on the data?
target_transform - you can transform the targets (labels) if you like too.
Many other datasets in torchvision have these parameter options.

"""
"""
# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)


test_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=False, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)
"""
train_data = datasets.FashionMNIST(root="data", train=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, transform=ToTensor())
print("Number of train image: ",len(train_data))
print("Number of test image: ",len(test_data))
"""
What ToTensor()

Before ToTensor():
Image is a PIL Image
Pixel values range: 0 → 255

After ToTensor():
Image becomes a torch.Tensor
Pixel values become: 0 → 1 (float values)
It also changes shape to: (C, H, W)

Grayscale
28 × 28
After ToTensor(), what will be the shape of: (1, 28, 28)

Because ToTensor():
    Converts PIL image → torch.Tensor
    Adds channel dimension
    Reorders to PyTorch format → (C, H, W)


"""

image, label = train_data[0]
"""
Return one image and its label
It loads the 0th image
Applies any transform (like ToTensor())
Returns: image tensor ,label (integer)
"""
#print(image)
#print("labels: ",label)  output is 9, means 9th Unique element is present 
print("Shape of the image: ",image.shape) #[color_channels=1, height=28, width=28] ,I due to grayscale
class_names=train_data.classes
print("Number of  Unique items:",len(class_names))
print("Unique items:",class_names)


print(f"Image shape: {image.shape}")
plt.imshow(image.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
#squeeze(), bcs  imshow does not accept colour channels
plt.title(class_names[label])
plt.axis(False)
plt.show()

plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
plt.show()


# Plot more images
fig = plt.figure(figsize=(9, 9))

rows, cols = 4, 4
"""
This creates:
4 rows
4 columns
Total = 16 images
 16 cells, 4*4 grid 
"""
for i in range(1, rows * cols + 1):
    #rows * cols+1 = 16+1=17,loop from 1 to 16
    
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    """
    len(train_data) → 60000
    torch.randint(0, 60000, size=[1]) → random number
    .item() → convert tensor to normal integer
    size=[1] ?

    Because torch.randint() must know:
    How many random numbers do you want?
    If you say: size=[1] -> You are saying: “Give me ONE random number.”

    size does NOT mean “how many numbers”.

    It means: “What should be the SHAPE of the output tensor?”

    So:

        size=(5,) → 1D tensor with 5 values
        size=(2,3) → 2D tensor (2 rows, 3 columns)
        size=(4,3,2) → 3D tensor
    """
    #random_idx = 45231  (for example)

    img, label = train_data[random_idx]
    #img → (1,28,28)
    #label → number 0–9
    fig.add_subplot(rows, cols, i)
    #Put this image in position i of the 4×4 grid.
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False);
    """
    .squeeze() → remove channel dimension
    cmap="gray" → show correctly
    plt.axis(False) → remove x/y numbers
    """
plt.show()
"""
Dataset: Gives 1 sample at a time

DataLoader:  Takes Dataset → Creates batches → Makes it iterable

Think of it like this:
    Dataset = book of images
    DataLoader = person who hands you 32 pages at once

Shuffling -> Every epoch, it mixes the data.

Why?
    Because if data is ordered like:  0 0 0 0 0 1 1 1 1 1 2 2 2 2 ...
    The model may learn class order patterns.

Shuffle prevents bias.
Dataset = Data Storage
DataLoader = Data Feeding Mechanism
"""
BATCH_SIZE=32
train_data_load=DataLoader(train_data,  # dataset to turn into iterable
                      batch_size=BATCH_SIZE,  #Samples per batch
                      shuffle=True
                      )
test_data_load=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False)

print(f"Dataloaders: {train_data_load, test_data_load}") 
print(f"Length of train dataloader: {len(train_data_load)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_data_load)} batches of {BATCH_SIZE}")
train_features_batch, train_labels_batch=next(iter(train_data_load))  #takes first batch
print(train_features_batch.shape)  #torch.Size([32, 1, 28, 28]
print(train_labels_batch.shape) # torch.Size([32])  , 32 items in each batch - each has one label.

"""
iter(train_dataloader) → gives an iterator over batches.
next(...) → gives the first batch of the iterator.

train_features_batch → images of the batch, shape: [BATCH_SIZE, 1, 28, 28]
32 → number of images in batch
1 → number of channels (grayscale)
28, 28 → image height × width

train_labels_batch → labels of the batch, shape: [BATCH_SIZE]
Each value is 0–9 → class of the corresponding image
"""
