import numpy as np
import random
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#Creating a folder if not available
data_path=Path("data/")
image_path=data_path/ "Pizza_steak_sushi"
"""
Path("data/")   ==> creates a Path object for folder data

data_path / "Pizza_steak_sushi"   ==> joins paths
Result: "data/Pizza_steak_sushi"

So image_path represents: data/Pizza_steak_sushi
"""

if image_path.is_dir():
    print(f"{image_path} is present")
else:
    image_path.mkdir(parents=True ,exist_ok=True)
    #parents=True   if parent is not present it does not throw error it creates one
    #here data/ is a parent (if it is not present first it will create one )
    #exist_ok=True  if parent which is data/ (here) if already present it does not give error
    #mkdir() means make directory (create folder).
    #wb  open the file and write in binary mode

    #open can not open folder , it can only open files
    """
    You cannot write data directly into a folder.

    If you try: with open("data/Pizza_steak_sushi", "wb") as f:

    Will get an error like:
    IsADirectoryError: [Errno 21] Is a directory
    Because Python expects a file, not a directory.

    =>File Path
    This is a file. data/pizza_steak_sushi.zip
    This works:

    with open("data/pizza_steak_sushi.zip", "wb") as f:
    Because you're writing into a file.

    It will create pizza_steak_sushi.zip if it is not present.
    """

    # Download pizza, steak, sushi data
        
    with open(data_path / "pizza_steak_sushi.zip","wb") as f:
        request=requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        """
        This sends an HTTP request to GitHub.
        The returned object request contains:
            status code (200 if successful)
            headers
            file data
        """

        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)
        """
        request.content contains the ZIP file data in bytes
        f.write() writes those bytes into your opened file
        """
        
    #Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip","r")  as zip_ref:
        print("Unzipping pizza, steak, sushi data...") 
        zip_ref.extractall(image_path)
        #Takes all files inside the zip
        #Extracts them into the folder image_path
    

    

