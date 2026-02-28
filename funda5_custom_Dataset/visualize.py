import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from  PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from  torchvision import datasets, transforms 

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
"""
data/pizza_steak_sushi
Now you get:

.glob()
.is_dir()
.parent
.stem
.name

So Path is more powerful and safer.
"""

train_dir="E:\\My AI ML\\AI-ML\\funda5_custom_Dataset\\data\\Pizza_steak_sushi\\train"
test_dir="E:\\My AI ML\\AI-ML\\funda5_custom_Dataset\\data\\Pizza_steak_sushi\\test"


print(train_dir)
print(test_dir)
#String Object
#image_path="data\\pizza_steak_sushi"   , will give error in using with glob() , bcs Because string has no .glob() method.
print(image_path)  #data/pizza_steak_sushi  is Path Object

#random.seed(42)


#Get all image paths (* means "any combination")
image_path_list=list(image_path.glob("*/*/*.jpg"))
#glob() is a method from Path that: Searches for files matching a pattern.
print(image_path_list)

#Get random image path
random_image_path=random.choice(image_path_list)
print(random_image_path)

#Get image class from path name (the image class is the name of the directory where the image is stored)
image_class=random_image_path.parent.stem
print(image_class)
"""

random_image_path: data/pizza_steak_sushi/train/pizza/img1.jpg

img1.jpg          ← file
pizza             ← parent folder
train             ← grandparent
pizza_steak_sushi ← great-grandparent
data              ← root

Break it down:
random_image_path.parent → pizza folder
.stem → folder name as string

.parent gives the immediate folder containing the file.
.stem removes extension if it’s a file OR gives folder name if it's a folder.

| Command       | Output       |
| ------------ | ----------|
| .name             | img1.jpg     |
| .stem              | img1           |
| .parent           | pizza folder |
| .parent.stem  | pizza            |
| .parent.parent.stem | train |

"""

#Open image

img=Image.open(random_image_path)


print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")
print(img)

# Turn the image into an array
img_array=np.asarray(img)

# Plot the image with matplotlib
plt.figure(figsize=(10,7))
plt.imshow(img_array)
plt.title(f"Image class: {image_class} | Image shape: {img_array.shape} -> [height, width, color_channels]")
plt.axis(False)
plt.show()


#Transforming data
"""
Converting image into tesnor(number on which we can apply ML Algo)

-> Resize the images using transforms.Resize() (from about 512x512 to 64x64, the same shape as the images on the CNN Explainer website).
-> Flip our images randomly on the horizontal using transforms.RandomHorizontalFlip() (this could be considered a form of data augmentation because it will artificially change our image data).
-> Turn our images from a PIL image to a PyTorch tensor using transforms.ToTensor().
"""



# Write transform for image
#ombines multiple image transforms into one single transform pipeline.
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])


def plot_transform(image_paths,transform,n=3,Seed=None):
    """
    Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """

    random.seed(Seed)
    random_image_paths=random.sample(image_paths,k=n)
    """
| Function                              | Returns     | Allows duplicates? |
| -------------------         | ----------- | ------------------ |
| random.choice(list)           | One element |  No (only 1)      |
| random.choices(list, k=n) | n elements  |  Yes              |
| random.sample(list, k=n)  | n elements  |  No               |

    """
    #Creates a list of n image path taken in random
    for each_img_path in random_image_paths:
        with Image.open(each_img_path) as img_f:  #img_f becomes a PIL Image object
            
            fig, ax=plt.subplots(1,2)
            """
            Create 1 row
            Create 2 columns
            So we get 2 side-by-side plots
            ax[0] → Left plot
            ax[1] → Right plot
            """
            ax[0].imshow(img_f)
            ax[0].set_title(f"Original \nSize: {img_f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(img_f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {each_img_path.parent.stem}", fontsize=20)
        plt.show()


plot_transform(image_path_list, 
                        transform=data_transform, 
                        n=3)
"""
plt.figure(figsize=(8,4))  # create new figure
plt.subplot(rows, columns, position)
plt.subplot(1, 2, 1)  # 1 row, 2 columns, position 1

def plot_all_in_one(image_paths, transform, n=3, Seed=None):
    
    random.seed(Seed)
    random_image_paths = random.sample(image_paths, k=n)

    # Create ONE figure with n rows and 2 columns
    plt.figure(figsize=(8, 4*n))

    for i, each_img_path in enumerate(random_image_paths):
        with Image.open(each_img_path) as img_f:

            # ---- Original Image ----
            plt.subplot(n, 2, 2*i + 1)
            plt.imshow(img_f)
            plt.title(f"Original\n{img_f.size}")
            plt.axis("off")

            # ---- Transformed Image ----
            transformed_image = transform(img_f).permute(1, 2, 0)

            plt.subplot(n, 2, 2*i + 2)
            plt.imshow(transformed_image)
            plt.title(f"Transformed\n{transformed_image.shape}")
            plt.axis("off")

    plt.tight_layout()
    plt.show()

plot_all_in_one(image_path_list, 
                        transform=data_transform, 
                        n=3)
"""
