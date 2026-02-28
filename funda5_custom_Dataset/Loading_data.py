import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt 

"""
datasets

datasets provides ready-made dataset classes.

What Is a Dataset Pipeline?
    Data preparation system
    A dataset pipeline is the full process that takes:

    Raw data (images/files)
            ↓
    Loads them
            ↓
    Transforms them
            ↓
    Creates batches
            ↓
    Feeds them to the model
"""
data_path=Path("data/")
image_path=data_path / "pizza_steak_sushi"

train_dir = r"E:\My AI ML\AI-ML\funda5_custom_Dataset\data\Pizza_steak_sushi\train"
test_dir=r"E:\My AI ML\AI-ML\funda5_custom_Dataset\data\Pizza_steak_sushi\test"

data_transform=transforms.Compose(
    [transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()]
    )
train_data=datasets.ImageFolder(root=train_dir,  # Target folder of images
                                transform=data_transform,   # Transforms to perform on data (images)
                                target_transform=None #Transforms to perform on labels (if necessary)
                                )

test_data=datasets.ImageFolder(root=test_dir,  # Target folder of images
                                transform=data_transform,   # Transforms to perform on data (images)
                                target_transform=None #Transforms to perform on labels (if necessary)
                                )
print(f"Train data:\n{train_data}\nTest data:\n{test_data}")



print("Class Names")
#Class names in list
class_names=train_data.classes
print(class_names)

#class names in dict
class_names_dict=train_data.class_to_idx
print(class_names_dict)

print(f"Number of data in train data is {len(train_data)} and Number of data in test data is {len(test_data)}")

#taking first image and its label from training data set
img, label=train_data[0][0],train_data[0][1]
print(f"Image vector {img}")
print(f"Image Shape {img.shape} and  Image dim {img.ndim} and Type {img.dtype}")

print(f"Label {label} and Name {class_names[label]}")
print(f"Label Type {type(label)}")
img_per=img.permute(1,2,0)
plt.imshow(img_per)
plt.title(class_names[label])
plt.axis(False)
plt.show()

train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=1, # how many samples per batch?
                              num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             num_workers=0, 
                             shuffle=False) # don't usually need to shuffle testing data

print(train_dataloader)
print(test_dataloader)
img, label = next(iter(train_dataloader))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

