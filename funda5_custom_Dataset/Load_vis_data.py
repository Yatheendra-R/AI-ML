import numpy as np
import random
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#Creating a folder if not available
data_path=Path("data/")
image_path=data_path/ "Pizza_steak_sushi"

if image_path.is_dir():
    print(f"{image_path} is present")
else:
    image_path.mkdir(parents=True ,exist_ok=True)
    
