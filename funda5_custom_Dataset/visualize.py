import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from  PIL import Image
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

train_dir="E:\\My AI ML\\AI-ML\\funda5_custom_Dataset\\data\\Pizza_steak_sushi\\train"
test_dir="E:\\My AI ML\\AI-ML\\funda5_custom_Dataset\\data\\Pizza_steak_sushi\\test"


print(train_dir)
print(test_dir)

#image_path="data\\pizza_steak_sushi"
print(image_path)

random.seed(42)


#Get all image paths (* means "any combination")

image_path_list=list(image_path.glob("*/*/*.jpg"))
print(image_path_list)

#Get random image path
random_image_path=random.choice(image_path_list)
print(random_image_path)

#Get image class from path name (the image class is the name of the directory where the image is stored)
image_class=random_image_path.parent.stem
print(image_class)

#Open image

img=Image.open(random_image_path)


print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")
print(img)

# Turn the image into an array
img_array=np.asarray(img)



plt.figure(figsize=(10,7))
plt.imshow(img_array)
plt.title(f"Image class: {image_class} | Image shape: {img_array.shape} -> [height, width, color_channels]")
plt.axis(False)
plt.show()
